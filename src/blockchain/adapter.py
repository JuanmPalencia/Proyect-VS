"""Blockchain adapter: BSV on-chain (OP_RETURN) + local ledger."""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
# BSV ON-CHAIN ADAPTER (bsvlib + WhatsOnChain)
# ═══════════════════════════════════════════════════════════════════════

class BSVAdapter(BlockchainAdapter):
    """Real BSV blockchain adapter using bsvlib for OP_RETURN transactions.

    Requires BSV_PRIVATE_KEY in WIF format in .env.
    Uses WhatsOnChain API to verify/query transactions.
    All records are also saved to local ledger for fast lookups.
    """

    def __init__(self):
        self.private_key_wif = config.BSV_PRIVATE_KEY
        self.network = config.BSV_NETWORK
        self.woc_base = (
            "https://api.whatsonchain.com/v1/bsv/test"
            if self.network == "testnet"
            else "https://api.whatsonchain.com/v1/bsv/main"
        )
        self._local = LocalLedgerAdapter()  # always keep local copy
        self._wallet = None

        if not self.private_key_wif:
            logger.warning("BSV_PRIVATE_KEY not set - BSVAdapter in stub mode")

    @property
    def is_configured(self) -> bool:
        return bool(self.private_key_wif)

    def _get_wallet(self):
        """Lazy-init bsvlib wallet."""
        if self._wallet is None:
            from bsvlib import Wallet
            from bsvlib.constants import Chain
            chain = Chain.TEST if self.network == "testnet" else Chain.MAIN
            self._wallet = Wallet(chain=chain)
            self._wallet.add_key(self.private_key_wif)
        return self._wallet

    def register(self, evidence_record: dict[str, Any]) -> dict[str, Any]:
        # Always save locally first
        local_result = self._local.register(evidence_record)

        if not self.is_configured:
            logger.info("[BSV STUB] No private key - registered locally only")
            return {**local_result, "status": "local_only"}

        try:
            wallet = self._get_wallet()
            analysis_hash = evidence_record["analysis_hash"]
            scene_id = evidence_record.get("scene_id", "unknown")

            # OP_RETURN data: prefix | hash | scene_id | version
            pushdatas = [
                APP_PREFIX.encode(),
                analysis_hash.encode(),
                scene_id.encode(),
                APP_VERSION.encode(),
            ]

            # Create OP_RETURN-only transaction (no value outputs needed)
            tx = wallet.create_transaction(outputs=[], pushdatas=pushdatas, combine=True)
            txid = tx.broadcast()

            if txid:
                explorer_url = (
                    f"https://test.whatsonchain.com/tx/{txid}"
                    if self.network == "testnet"
                    else f"https://whatsonchain.com/tx/{txid}"
                )
                logger.info("[BSV] Registered %s → tx %s", analysis_hash[:16], txid)
                # Update local record with txid
                self._local._update_record_txid(analysis_hash, txid)
                return {
                    "tx_id": txid,
                    "status": "confirmed",
                    "network": self.network,
                    "explorer_url": explorer_url,
                    "evidence_id": local_result.get("evidence_id"),
                }
            else:
                logger.warning("[BSV] Broadcast returned empty txid")
                return {**local_result, "status": "broadcast_failed"}

        except Exception as e:
            logger.error("[BSV] Transaction failed: %s", e)
            return {
                **local_result,
                "status": "error",
                "error": str(e),
            }

    def verify(self, analysis_hash: str) -> dict[str, Any] | None:
        # Check local ledger first (fast)
        record = self._local.verify(analysis_hash)
        if record and record.get("tx_id") and not record["tx_id"].startswith("local_"):
            # Verify on-chain via WhatsOnChain
            try:
                import requests
                txid = record["tx_id"]
                resp = requests.get(f"{self.woc_base}/tx/{txid}", timeout=10)
                if resp.status_code == 200:
                    record["on_chain_verified"] = True
                    record["confirmations"] = resp.json().get("confirmations", 0)
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
