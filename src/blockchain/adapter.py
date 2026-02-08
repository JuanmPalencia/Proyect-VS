"""Adaptador de Blockchain: BSV On-Chain (OP_RETURN via bsv-sdk + ARC) + registro local."""

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

# ── Prefijo del protocolo para datos OP_RETURN ────────────────────────────────
APP_PREFIX = "TRAFFIC_EVIDENCE"
APP_VERSION = "v1.0"


class BlockchainAdapter(ABC):
    """Interfaz abstracta para el registro de evidencia."""

    @abstractmethod
    def register(self, evidence_record: dict[str, Any]) -> dict[str, Any]:
        """Registra la evidencia. Devuelve {tx_id, status, ...}."""
        ...

    @abstractmethod
    def verify(self, analysis_hash: str) -> dict[str, Any] | None:
        """Busca evidencia registrada por su analysis_hash."""
        ...

    @abstractmethod
    def list_records(self, limit: int = 50) -> list[dict[str, Any]]:
        """Lista los registros de evidencia más recientes."""
        ...



class BSVAdapter(BlockchainAdapter):
    """Adaptador real de blockchain BSV usando bsv-sdk para transacciones OP_RETURN.

    Usa ARC (https://arc.gorillapool.io) para transmisión y
    la API de WhatsOnChain para consultas UTXO y verificación de transacciones.
    Todos los registros también se guardan en un libro local para consultas rápidas.
    """

    def __init__(self):
        self.private_key_wif = config.BSV_PRIVATE_KEY
        self.network = config.BSV_NETWORK  # "main" or "testnet"
        self.arc_url = config.ARC_URL
        self.woc_base = config.WOC_BASE
        self._local = LocalLedgerAdapter()  # siempre mantener una copia local

        self._key = None
        self._address = None

        if not self.private_key_wif:
            logger.warning("BSV_PRIVATE_KEY not set - BSVAdapter in stub mode")
        else:
            self._init_key()

    def _init_key(self):
        """Inicializa la PrivateKey de bsv-sdk desde WIF."""
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

    # ── Obtención de UTXO vía WhatsOnChain ────────────────────────────────

    def _fetch_utxos(self) -> list[dict]:
        """Obtiene las salidas no gastadas para nuestra dirección desde WhatsOnChain."""
        url = f"{self.woc_base}/address/{self._address}/unspent"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        utxos = resp.json()
        logger.info("[BSV] Found %d UTXOs for %s", len(utxos), self._address)
        return utxos

    def _fetch_raw_tx(self, txid: str) -> str:
        """Obtiene el hexadecimal de la transacción raw desde WhatsOnChain."""
        url = f"{self.woc_base}/tx/{txid}/hex"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text.strip()

    # ── Construcción de transacciones ──────────────────────────────────────────

    def _build_op_return_tx(self, analysis_hash: str, scene_id: str) -> Any:
        """Construye una transacción OP_RETURN firmada usando bsv-sdk."""
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

        # Elegir el primer UTXO con suficientes satoshis (no necesita muchos)
        utxo = max(utxos, key=lambda u: u["value"])
        source_txid = utxo["tx_hash"]
        source_vout = utxo["tx_pos"]
        source_satoshis = utxo["value"]

        logger.info(
            "[BSV] Using UTXO %s:%d (%d sats)",
            source_txid, source_vout, source_satoshis,
        )

        # Obtener la transacción fuente para construir el input
        raw_hex = self._fetch_raw_tx(source_txid)
        source_tx = Transaction.from_hex(raw_hex)

        # Construir transacción
        tx = Transaction()

        # Input: gastar el UTXO
        tx_input = TransactionInput(
            source_transaction=source_tx,
            source_output_index=source_vout,
            unlocking_script_template=P2PKH().unlock(self._key),
        )
        tx.add_input(tx_input)

        # Output 1: OP_RETURN con los datos de evidencia
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

        # Output 2: Cambio de vuelta a nuestra dirección
        change_output = TransactionOutput(
            locking_script=P2PKH().lock(self._address),
            change=True,
        )
        tx.add_output(change_output)

        # Calcular comisión y firmar
        tx.fee()
        tx.sign()

        logger.info(
            "[BSV] Built tx %s (%d bytes, fee=%d sats)",
            tx.txid(), tx.byte_length(), tx.get_fee(),
        )
        return tx

    def _broadcast_tx(self, tx: Any) -> str:
        """Transmite la transacción vía ARC. Devuelve el txid."""
        from bsv import ARC

        broadcaster = ARC(self.arc_url)
        # Usar sync_broadcast para integración síncrona
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
        """Construye la URL del explorador WhatsOnChain."""
        if self.network == "testnet":
            return f"https://test.whatsonchain.com/tx/{txid}"
        return f"https://whatsonchain.com/tx/{txid}"


    def register(self, evidence_record: dict[str, Any]) -> dict[str, Any]:
        # Guardamos primero de manera local
        local_result = self._local.register(evidence_record)

        if not self.is_configured:
            logger.info("[BSV] No private key - registered locally only")
            return {**local_result, "status": "local_only"}

        analysis_hash = evidence_record["analysis_hash"]
        scene_id = evidence_record.get("scene_id", "unknown")

        try:
            tx = self._build_op_return_tx(analysis_hash, scene_id)
            txid = self._broadcast_tx(tx)

            # Actualizar el registro local con el txid real
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
            # Fallos esperados (sin UTXOs, transmisión rechazada)
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
        # Primero revisar el libro local (rápido)
        record = self._local.verify(analysis_hash)
        if not record:
            return None

        txid = record.get("tx_id", "")
        if txid and not txid.startswith("local_"):
            # Verificar en la cadena vía WhatsOnChain
            try:
                resp = requests.get(
                    f"{self.woc_base}/tx/{txid}", timeout=10,
                )
                if resp.status_code == 200:
                    tx_data = resp.json()
                    record["on_chain_verified"] = True
                    record["confirmations"] = tx_data.get("confirmations", 0)
                    record["explorer_url"] = self._explorer_url(txid)

                    # Intentar verificar que los datos OP_RETURN coincidan
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
# LIBRO LOCAL JSONL (siempre activo, también usado como caché para BSV)
# ═══════════════════════════════════════════════════════════════════════

class LocalLedgerAdapter(BlockchainAdapter):
    """Libro local en archivo JSONL para modo demo/offline y como caché de BSV."""

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
        """Actualiza un registro local con el txid de la cadena."""
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


def get_blockchain_adapter() -> BlockchainAdapter:
    """Devuelve BSVAdapter (siempre - internamente recurre al libro local si es necesario)."""
    return BSVAdapter()
