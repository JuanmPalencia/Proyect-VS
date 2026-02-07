"""Blockchain adapter: BSV on-chain (OP_RETURN via bsv-sdk + ARC) + local ledger."""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

# ── Protocol prefix for OP_RETURN data ────────────────────────────────
APP_PREFIX = "TRAFFIC_EVIDENCE"
APP_VERSION = "v1.0"


class BlockchainAdapter(ABC):
    """Abstract interface for evidence registration."""

    @abstractmethod
    def register(self, evidence_record: dict[str, Any]) -> dict[str, Any]:
        """Register evidence. Returns {tx_id, status, ...}."""
        ...

    @abstractmethod
    def verify(self, analysis_hash: str) -> dict[str, Any] | None:
        """Look up registered evidence by analysis_hash."""
        ...

    @abstractmethod
    def list_records(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent evidence records."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# BSV ON-CHAIN ADAPTER (bsv-sdk + ARC broadcaster)
# ═══════════════════════════════════════════════════════════════════════

class BSVAdapter(BlockchainAdapter):
    """Real BSV blockchain adapter using bsv-sdk for OP_RETURN transactions.

    Uses ARC (https://arc.gorillapool.io) for broadcasting and
    WhatsOnChain API for UTXO lookups and tx verification.
    All records are also saved to local ledger for fast lookups.
    """

    def __init__(self):
        self.private_key_wif = config.BSV_PRIVATE_KEY
        self.network = config.BSV_NETWORK  # "main" or "testnet"
        self.arc_url = config.ARC_URL
        self.woc_base = config.WOC_BASE
        self._local = LocalLedgerAdapter()  # always keep local copy

        self._key = None
        self._address = None

        if not self.private_key_wif:
            logger.warning("BSV_PRIVATE_KEY not set - BSVAdapter in stub mode")
        else:
            self._init_key()

    def _init_key(self):
        """Initialize bsv-sdk PrivateKey from WIF."""
        try:
            from bsv import PrivateKey
            self._key = PrivateKey(self.private_key_wif)
            self._address = self._key.address()
            logger.info("[BSV] Initialized key, address: %s", self._address)
        except Exception as e:
            logger.error("[BSV] Failed to init PrivateKey: %s", e)
            self._key = None

    @property
    def is_configured(self) -> bool:
        return self._key is not None

    @property
    def address(self) -> str | None:
        return self._address

    # ── UTXO fetching via WhatsOnChain ────────────────────────────────

    def _fetch_utxos(self) -> list[dict]:
        """Fetch unspent outputs for our address from WhatsOnChain."""
        url = f"{self.woc_base}/address/{self._address}/unspent"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        utxos = resp.json()
        logger.info("[BSV] Found %d UTXOs for %s", len(utxos), self._address)
        return utxos

    def _fetch_raw_tx(self, txid: str) -> str:
        """Fetch raw transaction hex from WhatsOnChain."""
        url = f"{self.woc_base}/tx/{txid}/hex"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text.strip()

    # ── Transaction building ──────────────────────────────────────────

    def _build_op_return_tx(self, analysis_hash: str, scene_id: str) -> Any:
        """Build a signed OP_RETURN transaction using bsv-sdk."""
        from bsv import (
            Transaction, TransactionInput, TransactionOutput,
            P2PKH, OpReturn,
        )

        utxos = self._fetch_utxos()
        if not utxos:
            raise RuntimeError(
                f"No UTXOs available for address {self._address}. "
                "Fund the address to broadcast on-chain."
            )

        # Pick the first UTXO with enough satoshis (OP_RETURN txs are cheap)
        utxo = max(utxos, key=lambda u: u["value"])
        source_txid = utxo["tx_hash"]
        source_vout = utxo["tx_pos"]
        source_satoshis = utxo["value"]

        logger.info(
            "[BSV] Using UTXO %s:%d (%d sats)",
            source_txid, source_vout, source_satoshis,
        )

        # Fetch the source transaction for input construction
        raw_hex = self._fetch_raw_tx(source_txid)
        source_tx = Transaction.from_hex(raw_hex)

        # Build transaction
        tx = Transaction()

        # Input: spend the UTXO
        tx_input = TransactionInput(
            source_transaction=source_tx,
            source_output_index=source_vout,
            unlocking_script_template=P2PKH().unlock(self._key),
        )
        tx.add_input(tx_input)

        # Output 1: OP_RETURN with evidence data
        op_return_data = [
            APP_PREFIX,
            analysis_hash,
            scene_id,
            APP_VERSION,
        ]
        data_output = TransactionOutput(
            locking_script=OpReturn().lock(op_return_data),
            satoshis=0,
        )
        tx.add_output(data_output)

        # Output 2: Change back to our address
        change_output = TransactionOutput(
            locking_script=P2PKH().lock(self._address),
            change=True,
        )
        tx.add_output(change_output)

        # Calculate fee and sign
        tx.fee()
        tx.sign()

        logger.info(
            "[BSV] Built tx %s (%d bytes, fee=%d sats)",
            tx.txid(), tx.byte_length(), tx.get_fee(),
        )
        return tx

    def _broadcast_tx(self, tx: Any) -> str:
        """Broadcast transaction via ARC. Returns txid."""
        from bsv import ARC

        broadcaster = ARC(self.arc_url)
        # Use sync_broadcast for synchronous integration
        response = broadcaster.sync_broadcast(tx, timeout=30)

        if response.status == "success":
            logger.info("[BSV] Broadcast success: %s", response.txid)
            return response.txid
        else:
            raise RuntimeError(
                f"ARC broadcast failed: {response.status} - "
                f"{getattr(response, 'description', getattr(response, 'message', 'unknown'))}"
            )

    def _explorer_url(self, txid: str) -> str:
        """Build WhatsOnChain explorer URL."""
        if self.network == "testnet":
            return f"https://test.whatsonchain.com/tx/{txid}"
        return f"https://whatsonchain.com/tx/{txid}"

    # ── Public API ────────────────────────────────────────────────────

    def register(self, evidence_record: dict[str, Any]) -> dict[str, Any]:
        # Always save locally first (fast, resilient)
        local_result = self._local.register(evidence_record)

        if not self.is_configured:
            logger.info("[BSV] No private key - registered locally only")
            return {**local_result, "status": "local_only"}

        analysis_hash = evidence_record["analysis_hash"]
        scene_id = evidence_record.get("scene_id", "unknown")

        try:
            tx = self._build_op_return_tx(analysis_hash, scene_id)
            txid = self._broadcast_tx(tx)

            # Update local record with real txid
            self._local._update_record_txid(analysis_hash, txid)

            return {
                "tx_id": txid,
                "status": "on_chain",
                "network": self.network,
                "explorer_url": self._explorer_url(txid),
                "evidence_id": local_result.get("evidence_id"),
                "address": self._address,
            }

        except RuntimeError as e:
            # Expected failures (no UTXOs, broadcast rejected)
            logger.warning("[BSV] %s", e)
            return {
                **local_result,
                "status": "local_fallback",
                "warning": str(e),
                "address": self._address,
            }

        except Exception as e:
            logger.error("[BSV] Unexpected error: %s", e, exc_info=True)
            return {
                **local_result,
                "status": "error",
                "error": str(e),
            }

    def verify(self, analysis_hash: str) -> dict[str, Any] | None:
        # Check local ledger first (fast)
        record = self._local.verify(analysis_hash)
        if not record:
            return None

        txid = record.get("tx_id", "")
        if txid and not txid.startswith("local_"):
            # Verify on-chain via WhatsOnChain
            try:
                resp = requests.get(
                    f"{self.woc_base}/tx/{txid}", timeout=10,
                )
                if resp.status_code == 200:
                    tx_data = resp.json()
                    record["on_chain_verified"] = True
                    record["confirmations"] = tx_data.get("confirmations", 0)
                    record["explorer_url"] = self._explorer_url(txid)

                    # Try to verify OP_RETURN data matches
                    hex_resp = requests.get(
                        f"{self.woc_base}/tx/{txid}/hex", timeout=10,
                    )
                    if hex_resp.status_code == 200:
                        record["raw_tx_available"] = True
                else:
                    record["on_chain_verified"] = False
            except Exception as e:
                record["on_chain_verified"] = False
                record["verify_error"] = str(e)

        return record

    def list_records(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._local.list_records(limit=limit)


# ═══════════════════════════════════════════════════════════════════════
# LOCAL JSONL LEDGER (always active, also used as cache for BSV)
# ═══════════════════════════════════════════════════════════════════════

class LocalLedgerAdapter(BlockchainAdapter):
    """Local JSONL file ledger for demo/offline mode and as BSV cache."""

    def __init__(self, ledger_path: str | None = None):
        self.ledger_path = Path(ledger_path) if ledger_path else config.LEDGER_PATH
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, evidence_record: dict[str, Any]) -> dict[str, Any]:
        evidence_id = str(uuid.uuid4())
        entry = {
            "evidence_id": evidence_id,
            **evidence_record,
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n")
        logger.info("[LOCAL] Registered %s → %s", evidence_record["analysis_hash"][:16], evidence_id[:8])
        return {
            "evidence_id": evidence_id,
            "tx_id": f"local_{evidence_id[:8]}",
            "status": "registered",
        }

    def verify(self, analysis_hash: str) -> dict[str, Any] | None:
        if not self.ledger_path.exists():
            return None
        for line in self.ledger_path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            if record.get("analysis_hash") == analysis_hash:
                return record
        return None

    def list_records(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        lines = self.ledger_path.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(line) for line in lines[-limit:]]
        return list(reversed(records))

    def _update_record_txid(self, analysis_hash: str, txid: str):
        """Update a local record with the on-chain txid."""
        if not self.ledger_path.exists():
            return
        lines = self.ledger_path.read_text(encoding="utf-8").strip().splitlines()
        updated = []
        for line in lines:
            record = json.loads(line)
            if record.get("analysis_hash") == analysis_hash:
                record["tx_id"] = txid
            updated.append(json.dumps(record, sort_keys=True, ensure_ascii=True))
        self.ledger_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════

def get_blockchain_adapter() -> BlockchainAdapter:
    """Returns BSVAdapter (always - it falls back internally to local ledger)."""
    return BSVAdapter()
