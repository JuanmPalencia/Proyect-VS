"""Streamlit frontend for Traffic Aerial Analysis System."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detection import VehicleDetector
from src.metrics import TrafficAnalyzer
from src.hashing.integrity import (
    build_analysis_payload,
    build_evidence_record,
    canonical_json,
    compute_hash,
    verify_integrity,
)
from src.blockchain import get_blockchain_adapter
from src.visualization.overlays import draw_detections, generate_heatmap, draw_density_grid

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Aerial Analysis",
    page_icon="🛣️",
    layout="wide",
)

st.title("🛣️ Sistema Inteligente de Análisis de Tráfico Aéreo")
st.markdown("Detección de vehículos en imágenes aéreas con evidencia verificable en blockchain BSV")


# ── Cached singletons ─────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    return VehicleDetector()


@st.cache_resource
def load_analyzer():
    return TrafficAnalyzer()


@st.cache_resource
def load_chain():
    return get_blockchain_adapter()


detector = load_detector()
analyzer = load_analyzer()
chain = load_chain()

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    dataset_id = st.selectbox("Dataset", ["uav_traffic", "roundabout", "upload"])
    is_roundabout = st.checkbox("Escena de rotonda", value=(dataset_id == "roundabout"))
    st.markdown("---")
    st.header("📋 Info del modelo")
    st.code(detector.model_version)
    st.markdown("---")
    st.header("🔗 Modo blockchain")
    st.info("Local Ledger (demo)" if not hasattr(chain, "is_configured") or not getattr(chain, "is_configured", False)
            else f"BSV {getattr(chain, 'network', 'testnet')}")

# ── Tabs ───────────────────────────────────────────────────────────────
tab_analyze, tab_verify, tab_records = st.tabs(["📸 Analizar", "✅ Verificar", "📜 Registros"])

# ═══════════════════════════════════════════════════════════════════════
# TAB: ANALYZE
# ═══════════════════════════════════════════════════════════════════════
with tab_analyze:
    uploaded_file = st.file_uploader(
        "Sube una imagen aérea (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="upload_analyze",
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = uploaded_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("No se pudo leer la imagen.")
        else:
            h, w = img_bgr.shape[:2]
            scene_id = uploaded_file.name

            # ── Detection ──────────────────────────────────────────
            with st.spinner("Detectando vehículos..."):
                detections = detector.detect(img_bgr)

            # ── Metrics ────────────────────────────────────────────
            metrics = analyzer.analyze(detections, h, w, is_roundabout=is_roundabout)

            # ── Build payload & hash ───────────────────────────────
            payload = build_analysis_payload(
                scene_id=scene_id,
                dataset_id=dataset_id,
                counts=metrics.counts,
                total_vehicles=metrics.total_vehicles,
                density_grid=metrics.density_grid,
                occupancy_pct=metrics.occupancy_pct,
                zone_occupancy=metrics.zone_occupancy,
                risk_level=metrics.risk_level,
                model_version=detector.model_version,
                is_roundabout=metrics.is_roundabout,
                roundabout_occupancy_pct=metrics.roundabout_occupancy_pct,
            )
            analysis_hash = compute_hash(payload)
            evidence = build_evidence_record(payload)

            # ── Visualization columns ──────────────────────────────
            col_img, col_heat = st.columns(2)

            with col_img:
                st.subheader("Detecciones")
                img_det = draw_detections(img_bgr, detections)
                st.image(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB),
                         caption=f"{len(detections)} vehículos detectados",
                         use_container_width=True)

            with col_heat:
                st.subheader("Mapa de calor")
                img_heat = generate_heatmap(img_bgr, detections)
                st.image(cv2.cvtColor(img_heat, cv2.COLOR_BGR2RGB),
                         caption="Densidad ponderada",
                         use_container_width=True)

            # ── Density grid ───────────────────────────────────────
            with st.expander("🔢 Mapa de densidad (grid)"):
                img_grid = draw_density_grid(img_bgr, metrics.density_grid)
                st.image(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

            # ── Metrics display ────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Métricas")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total vehículos", metrics.total_vehicles)
            m2.metric("Ocupación", f"{metrics.occupancy_pct:.1f}%")
            m3.metric("Riesgo", metrics.risk_level)
            if metrics.roundabout_occupancy_pct is not None:
                m4.metric("Ocp. rotonda", f"{metrics.roundabout_occupancy_pct:.1f}%")
            else:
                m4.metric("Zonas", f"{len(metrics.zone_occupancy)}")

            # Counts breakdown
            st.markdown("**Conteo por clase:**")
            for cls, cnt in metrics.counts.items():
                st.write(f"  - {cls}: **{cnt}**")

            # Zone occupancy
            st.markdown("**Ocupación por zona:**")
            for zone, pct in metrics.zone_occupancy.items():
                st.progress(pct / 100, text=f"{zone}: {pct:.1f}%")

            # ── Hash & Blockchain ──────────────────────────────────
            st.markdown("---")
            st.subheader("🔐 Evidencia Criptográfica")
            st.code(f"SHA-256: {analysis_hash}", language="text")

            # Register button
            if st.button("📝 Registrar en Blockchain", type="primary"):
                with st.spinner("Registrando evidencia..."):
                    tx_result = chain.register(evidence)
                st.success(f"Registrado. TX/ID: `{tx_result.get('tx_id', tx_result.get('evidence_id', 'N/A'))}`")
                st.json(tx_result)

            # Show canonical JSON
            with st.expander("📄 JSON canónico del análisis"):
                st.code(canonical_json(payload), language="json")

            with st.expander("📄 Evidence Record"):
                st.json(evidence)

# ═══════════════════════════════════════════════════════════════════════
# TAB: VERIFY
# ═══════════════════════════════════════════════════════════════════════
with tab_verify:
    st.subheader("✅ Verificar integridad de un análisis")
    st.markdown("Introduce el hash SHA-256 para buscar en el registro blockchain/ledger.")

    hash_input = st.text_input("SHA-256 Hash", placeholder="abc123...")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        if st.button("🔍 Buscar en ledger"):
            if hash_input:
                record = chain.verify(hash_input)
                if record:
                    st.success("VERIFICADO: Registro encontrado")
                    st.json(record)
                else:
                    st.warning("No se encontró ningún registro con ese hash.")
            else:
                st.info("Introduce un hash.")

    with col_v2:
        st.markdown("**Re-verificar desde JSON:**")
        json_input = st.text_area("Pega el JSON canónico del análisis")
        if st.button("🔁 Recalcular hash"):
            if json_input:
                try:
                    data = json.loads(json_input)
                    recalc = compute_hash(data)
                    st.code(f"Hash recalculado: {recalc}", language="text")
                    if hash_input and recalc == hash_input:
                        st.success("Los hashes coinciden. Integridad confirmada.")
                    elif hash_input:
                        st.error("Los hashes NO coinciden. Los datos fueron alterados.")
                except json.JSONDecodeError:
                    st.error("JSON inválido.")

# ═══════════════════════════════════════════════════════════════════════
# TAB: RECORDS
# ═══════════════════════════════════════════════════════════════════════
with tab_records:
    st.subheader("📜 Registros recientes")
    records = chain.list_records(limit=20)
    if records:
        for i, rec in enumerate(records):
            with st.expander(
                f"#{i+1} | {rec.get('scene_id', 'N/A')} | {rec.get('timestamp_utc', '')[:19]}"
            ):
                st.json(rec)
    else:
        st.info("No hay registros aún. Analiza una imagen y regístrala.")
