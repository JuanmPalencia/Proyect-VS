"""Streamlit frontend for Traffic Aerial Analysis System."""

from __future__ import annotations

import json
import sys
from datetime import datetime, date, time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detection import VehicleDetector
from src.metrics import TrafficAnalyzer
from src.hashing.integrity import (
    build_analysis_payload,
    build_evidence_record,
    canonical_json,
    compute_hash,
)
from src.blockchain import get_blockchain_adapter
from src.visualization.overlays import draw_detections, generate_heatmap, draw_density_grid, draw_collisions
from src.detection.plate_reader import PlateReader
from src.simulator import TrafficSimulator
import config

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Aerial Analysis",
    page_icon="🛣️",
    layout="wide",
)

st.title("🛣️ Sistema Inteligente de Análisis de Tráfico Aéreo")
st.markdown("Detección vehicular en imágenes UAV con evidencia verificable en blockchain BSV")


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


@st.cache_resource
def load_simulator():
    return TrafficSimulator()


@st.cache_resource
def load_plate_reader():
    return PlateReader()


detector = load_detector()
analyzer = load_analyzer()
chain = load_chain()
simulator = load_simulator()
plate_reader = load_plate_reader()

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    dataset_id = st.selectbox("Dataset", ["uav_traffic", "roundabout", "upload"])
    is_roundabout = st.checkbox("Escena de rotonda", value=(dataset_id == "roundabout"))
    st.markdown("---")
    st.header("📋 Modelo")
    st.code(detector.model_version)
    st.header("🔗 Blockchain")
    if getattr(chain, "is_configured", False):
        st.success(f"BSV {getattr(chain, 'network', 'main')} (bsv-sdk + ARC)")
        addr = getattr(chain, "address", None)
        if addr:
            st.caption(f"Address: `{addr[:8]}...{addr[-6:]}`")
    else:
        st.info("Local Ledger (demo)")

# ── Tabs ───────────────────────────────────────────────────────────────
tab_analyze, tab_simulate, tab_verify, tab_records = st.tabs(
    ["📸 Analizar", "🔮 Simulador What-If", "✅ Verificar", "📜 Registros"]
)

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: ANALYZE (image upload → detection → metrics → hash → register)
# ═══════════════════════════════════════════════════════════════════════
with tab_analyze:
    uploaded_file = st.file_uploader(
        "Sube una imagen aérea (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="upload_analyze",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("No se pudo leer la imagen.")
        else:
            h, w = img_bgr.shape[:2]
            scene_id = uploaded_file.name

            with st.spinner("Detectando vehículos..."):
                detections = detector.detect(img_bgr)

            metrics = analyzer.analyze(detections, h, w, is_roundabout=is_roundabout)

            # Extract collision data safely
            m_collisions = getattr(metrics, "collisions", None) or []
            m_collision_count = getattr(metrics, "collision_count", 0)

            # Identify vehicles involved in collisions (privacy: only on collision)
            collision_details = []
            if m_collisions and config.ENABLE_PLATE_OCR:
                involved_idxs = set()
                for col in m_collisions:
                    involved_idxs.add(col["vehicle_a_idx"])
                    involved_idxs.add(col["vehicle_b_idx"])
                for idx in involved_idxs:
                    vid = plate_reader.identify_vehicle(img_bgr, detections[idx], h, w)
                    vid["detection_idx"] = idx
                    collision_details.append(vid)

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
                collision_count=m_collision_count,
                collisions=m_collisions if m_collisions else None,
            )
            if collision_details:
                payload["collision_vehicle_details"] = collision_details
            analysis_hash = compute_hash(payload)
            evidence = build_evidence_record(payload)

            # Visualizations
            col_img, col_heat = st.columns(2)
            with col_img:
                st.subheader("Detecciones")
                img_det = draw_detections(img_bgr, detections)
                st.image(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB),
                         caption=f"{len(detections)} vehiculos detectados",
                         use_container_width=True)
            with col_heat:
                st.subheader("Mapa de calor")
                img_heat = generate_heatmap(img_bgr, detections)
                st.image(cv2.cvtColor(img_heat, cv2.COLOR_BGR2RGB),
                         caption="Densidad ponderada",
                         use_container_width=True)

            # Collision visualization
            if m_collisions:
                st.subheader("Colisiones detectadas")
                img_col = draw_collisions(img_bgr, m_collisions, detections)
                img_col = draw_detections(img_col, detections)
                n_col = m_collision_count
                st.image(cv2.cvtColor(img_col, cv2.COLOR_BGR2RGB),
                         caption=f"{n_col} posibles colisiones",
                         use_container_width=True)

            with st.expander("Mapa de densidad (grid)"):
                img_grid = draw_density_grid(img_bgr, metrics.density_grid)
                st.image(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Metrics
            st.markdown("---")
            st.subheader("Metricas")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total vehiculos", metrics.total_vehicles)
            m2.metric("Ocupacion", f"{metrics.occupancy_pct:.1f}%")
            m3.metric("Riesgo", metrics.risk_level)
            m4.metric("Colisiones", m_collision_count)
            if metrics.roundabout_occupancy_pct is not None:
                m5.metric("Ocp. rotonda", f"{metrics.roundabout_occupancy_pct:.1f}%")
            else:
                m5.metric("Zonas", f"{len(metrics.zone_occupancy)}")

            st.markdown("**Conteo por clase:**")
            for cls, cnt in metrics.counts.items():
                st.write(f"  - {cls}: **{cnt}**")

            st.markdown("**Ocupacion por zona:**")
            for zone, pct in metrics.zone_occupancy.items():
                st.progress(pct / 100, text=f"{zone}: {pct:.1f}%")

            # Collision details
            if m_collisions:
                st.markdown("---")
                st.subheader("Detalle de colisiones")
                for i, col in enumerate(m_collisions):
                    sev = col["severity"]
                    severity_color = {"HIGH": "red", "MEDIUM": "orange", "WARNING": "yellow"}.get(sev, "gray")
                    st.markdown(
                        f"**Colision #{i+1}** — "
                        f":{severity_color}[{sev}] — "
                        f"{col['vehicle_a_class']} vs {col['vehicle_b_class']} "
                        f"(IoU: {col['iou']:.3f}, dist: {col['distance']:.0f}px)"
                    )

                if collision_details:
                    st.markdown("**Vehiculos involucrados:**")
                    for vid in collision_details:
                        plate_info = "Matricula detectada (hash almacenado)" if vid.get("plate_detected") else "Matricula no legible"
                        st.write(
                            f"  - **{vid['class']}** | Color: {vid['dominant_color']} | "
                            f"Pos: {vid['position']} | {plate_info}"
                        )

            # Hash & Blockchain
            st.markdown("---")
            st.subheader("🔐 Evidencia Criptográfica")
            st.code(f"SHA-256: {analysis_hash}", language="text")

            if st.button("📝 Registrar en Blockchain", type="primary", key="reg_analyze"):
                with st.spinner("Registrando evidencia (local + broadcast ARC)..."):
                    tx_result = chain.register(evidence)

                status = tx_result.get("status", "unknown")
                if status == "on_chain":
                    st.success(f"Registrado ON-CHAIN: `{tx_result['tx_id']}`")
                    st.markdown(f"[Ver en WhatsOnChain]({tx_result['explorer_url']})")
                elif status == "local_fallback":
                    st.warning(
                        f"Guardado localmente (sin fondos on-chain). "
                        f"Fondea la address para broadcast real."
                    )
                    if tx_result.get("address"):
                        st.code(f"Address BSV: {tx_result['address']}", language="text")
                else:
                    st.info(f"Registrado localmente: `{tx_result.get('tx_id', 'N/A')}`")

                with st.expander("Detalle del registro"):
                    st.json(tx_result)

            with st.expander("📄 JSON canónico"):
                st.code(canonical_json(payload), language="json")
            with st.expander("📄 Evidence Record"):
                st.json(evidence)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════
with tab_simulate:
    st.subheader("🔮 Simulador de Escenarios de Tráfico")
    st.markdown(
        "Simula el estado del tráfico para una **fecha, hora y tipo de escena**. "
        "Ajusta parámetros manualmente o deja que el sistema estime automáticamente."
    )

    # ── Input panel ────────────────────────────────────────────────
    col_dt, col_scene, col_name = st.columns(3)
    with col_dt:
        sim_date = st.date_input("📅 Fecha", value=date.today())
        sim_time = st.time_input("🕐 Hora", value=time(8, 0))
    with col_scene:
        scene_type = st.selectbox(
            "🏙️ Tipo de escena",
            ["urban_road", "roundabout", "highway"],
            format_func=lambda x: {"urban_road": "Vía urbana", "roundabout": "Rotonda", "highway": "Autovía"}[x],
        )
    with col_name:
        sim_scene_id = st.text_input("📍 ID de escena", value="zona_centro_01")

    sim_datetime = datetime.combine(sim_date, sim_time)

    # ── Manual overrides ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Ajustes manuales** (dejar en 0 = auto-estimación)")

    ov_col1, ov_col2, ov_col3 = st.columns(3)
    with ov_col1:
        ov_total = st.number_input("Total vehículos (0=auto)", min_value=0, max_value=200, value=0, key="sim_total")
    with ov_col2:
        ov_density = st.number_input("Densidad promedio (0=auto)", min_value=0.0, max_value=20.0, value=0.0,
                                      step=0.5, key="sim_density")
    with ov_col3:
        ov_occupancy = st.number_input("Ocupación % (0=auto)", min_value=0.0, max_value=100.0, value=0.0,
                                        step=1.0, key="sim_occ")

    # Per-class override
    with st.expander("🚗 Ajuste por tipo de vehículo (opcional)"):
        st.markdown("Si introduces valores, se usarán en lugar de la estimación automática.")
        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        ov_cars = vc1.number_input("Coches", min_value=0, max_value=100, value=0, key="sim_car")
        ov_motos = vc2.number_input("Motos", min_value=0, max_value=50, value=0, key="sim_moto")
        ov_buses = vc3.number_input("Buses", min_value=0, max_value=20, value=0, key="sim_bus")
        ov_trucks = vc4.number_input("Camiones", min_value=0, max_value=30, value=0, key="sim_truck")
        ov_cycles = vc5.number_input("Bicicletas", min_value=0, max_value=30, value=0, key="sim_cycle")

        manual_counts_sum = ov_cars + ov_motos + ov_buses + ov_trucks + ov_cycles

    # ── Run simulation ─────────────────────────────────────────────
    if st.button("🚀 Simular escenario", type="primary", key="btn_simulate"):
        # Build overrides
        override_total = ov_total if ov_total > 0 else None
        override_density = ov_density if ov_density > 0 else None
        override_occupancy = ov_occupancy if ov_occupancy > 0 else None
        override_counts = None
        if manual_counts_sum > 0:
            override_counts = {}
            if ov_cars > 0:
                override_counts["car"] = ov_cars
            if ov_motos > 0:
                override_counts["motorcycle"] = ov_motos
            if ov_buses > 0:
                override_counts["bus"] = ov_buses
            if ov_trucks > 0:
                override_counts["truck"] = ov_trucks
            if ov_cycles > 0:
                override_counts["cycle"] = ov_cycles

        result = simulator.simulate(
            sim_datetime=sim_datetime,
            scene_type=scene_type,
            scene_id=sim_scene_id,
            override_total=override_total,
            override_counts=override_counts,
            override_density=override_density,
            override_occupancy=override_occupancy,
        )

        # ── Display results ────────────────────────────────────────
        state = result["traffic_state"]
        emoji = simulator.get_state_emoji(state)
        color = simulator.get_state_color(state)

        st.markdown("---")

        # Big state indicator
        st.markdown(
            f"<div style='text-align:center; padding:20px; border-radius:12px; "
            f"background-color:{color}22; border:2px solid {color};'>"
            f"<h1 style='color:{color}; margin:0;'>{emoji} {state}</h1>"
            f"<p style='font-size:1.2em; margin:5px 0 0 0;'>"
            f"{sim_datetime.strftime('%A %d/%m/%Y %H:%M')} | "
            f"{'Rotonda' if scene_type == 'roundabout' else 'Vía urbana' if scene_type == 'urban_road' else 'Autovía'}"
            f"</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Metrics row
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Vehículos", result["total_vehicles"])
        mc2.metric("Ocupación", f"{result['occupancy_pct']:.1f}%")
        mc3.metric("Riesgo", result["risk_level"])
        mc4.metric("Carga", f"{result['load_ratio']:.0%}")
        mc5.metric("Capacidad", result["capacity"])

        # Counts
        st.markdown("**Distribución estimada por tipo:**")
        counts_cols = st.columns(len(result["counts"]) or 1)
        for i, (cls, cnt) in enumerate(result["counts"].items()):
            counts_cols[i % len(counts_cols)].metric(cls.capitalize(), cnt)

        # Density grid visual
        col_grid, col_zones = st.columns(2)
        with col_grid:
            st.markdown("**Mapa de densidad simulado:**")
            grid = result["density_grid"]
            # Render as a simple colored table
            grid_html = "<table style='width:100%; border-collapse:collapse;'>"
            max_val = max(max(row) for row in grid) if grid else 1
            for row in grid:
                grid_html += "<tr>"
                for val in row:
                    intensity = val / max(max_val, 1)
                    r = int(255 * intensity)
                    g = int(255 * (1 - intensity))
                    grid_html += (
                        f"<td style='text-align:center; padding:12px; "
                        f"background-color:rgba({r},{g},0,0.6); "
                        f"color:white; font-weight:bold; border:1px solid #333;'>"
                        f"{val}</td>"
                    )
                grid_html += "</tr>"
            grid_html += "</table>"
            st.markdown(grid_html, unsafe_allow_html=True)

        with col_zones:
            st.markdown("**Ocupación por zona:**")
            for zone, pct in result["zone_occupancy"].items():
                label = {"upper": "Superior", "middle": "Central", "lower": "Inferior"}.get(zone, zone)
                st.progress(min(pct / 100, 1.0), text=f"{label}: {pct:.1f}%")

            if result["roundabout_occupancy_pct"] is not None:
                st.markdown(f"**Ocupación rotonda:** {result['roundabout_occupancy_pct']:.1f}%")

        # Context info
        with st.expander("📋 Contexto de la simulación"):
            ctx1, ctx2 = st.columns(2)
            ctx1.write(f"- Día: {'Fin de semana' if result['is_weekend'] else 'Laborable'}")
            ctx1.write(f"- Factor horario: {result['time_factor']:.2f}")
            ctx1.write(f"- Tipo escena: {scene_type}")
            ctx2.write(f"- Escena: {result['scene_id']}")
            ctx2.write(f"- Capacidad base: {result['capacity']} veh.")
            ctx2.write(f"- Ratio carga: {result['load_ratio']:.1%}")

        # ── Hash & Register simulation ─────────────────────────────
        st.markdown("---")
        st.subheader("🔐 Trazabilidad de la Simulación")

        sim_hash = compute_hash(result)
        st.code(f"SHA-256: {sim_hash}", language="text")

        sim_evidence = build_evidence_record(result)

        if st.button("📝 Registrar simulación en Blockchain", key="reg_sim"):
            with st.spinner("Registrando simulación (local + broadcast ARC)..."):
                tx_result = chain.register(sim_evidence)

            status = tx_result.get("status", "unknown")
            if status == "on_chain":
                st.success(f"Registrado ON-CHAIN: `{tx_result['tx_id']}`")
                st.markdown(f"[Ver en WhatsOnChain]({tx_result['explorer_url']})")
            elif status == "local_fallback":
                st.warning(
                    f"Guardado localmente (sin fondos on-chain). "
                    f"Fondea la address para broadcast real."
                )
                if tx_result.get("address"):
                    st.code(f"Address BSV: {tx_result['address']}", language="text")
            else:
                st.info(f"Registrado localmente: `{tx_result.get('tx_id', 'N/A')}`")

            with st.expander("Detalle del registro"):
                st.json(tx_result)

        with st.expander("📄 JSON canónico de la simulación"):
            st.code(canonical_json(result), language="json")

        with st.expander("📄 Evidence Record"):
            st.json(sim_evidence)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: VERIFY
# ═══════════════════════════════════════════════════════════════════════
with tab_verify:
    st.subheader("✅ Verificar integridad de un análisis o simulación")
    st.markdown("Introduce el hash SHA-256 para buscar en el registro blockchain/ledger.")

    hash_input = st.text_input("SHA-256 Hash", placeholder="abc123...", key="verify_hash")

    if st.button("🔍 Buscar y verificar", type="primary", key="btn_verify"):
        if hash_input:
            with st.spinner("Verificando..."):
                record = chain.verify(hash_input)
            if record:
                st.session_state["verify_result"] = record
            else:
                st.session_state["verify_result"] = "not_found"
        else:
            st.info("Introduce un hash.")

    # Mostrar resultado persistente
    if "verify_result" in st.session_state:
        result = st.session_state["verify_result"]
        if result == "not_found":
            st.warning("No se encontró ningún registro con ese hash.")
        else:
            txid = result.get("tx_id", "")
            is_onchain = not txid.startswith("local_") and txid

            if result.get("on_chain_verified"):
                st.success(
                    f"VERIFICADO ON-CHAIN "
                    f"({result.get('confirmations', 0)} confirmaciones)"
                )
                if result.get("explorer_url"):
                    st.markdown(f"[Ver en WhatsOnChain]({result['explorer_url']})")
            elif is_onchain:
                st.info(f"Registro con TX on-chain (pendiente de confirmación)")
                explorer = (
                    f"https://test.whatsonchain.com/tx/{txid}"
                    if getattr(chain, "network", "") == "testnet"
                    else f"https://whatsonchain.com/tx/{txid}"
                )
                st.markdown(f"[Ver en WhatsOnChain]({explorer})")
            else:
                st.success("Registro encontrado (solo local)")

            st.markdown("---")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown("**Datos del registro:**")
                st.write(f"- **Scene:** {result.get('scene_id', 'N/A')}")
                st.write(f"- **Modelo:** {result.get('model_version', 'N/A')}")
                st.write(f"- **Fecha:** {result.get('timestamp_utc', 'N/A')[:10]}")
                st.write(f"- **TX ID:** `{txid or 'N/A'}`")
            with col_info2:
                st.markdown("**Estado:**")
                if result.get("on_chain_verified"):
                    st.write(f"- Confirmaciones: **{result.get('confirmations', 0)}**")
                    st.write(f"- Raw TX disponible: **{result.get('raw_tx_available', False)}**")
                st.write(f"- Evidence ID: `{result.get('evidence_id', 'N/A')[:12]}...`")

            with st.expander("JSON completo del registro"):
                st.json(result)

    st.markdown("---")
    st.markdown("**Re-verificar desde JSON:**")
    json_input = st.text_area("Pega el JSON canónico del análisis o simulación")
    if st.button("🔁 Recalcular hash", key="btn_recalc"):
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
# TAB 4: RECORDS
# ═══════════════════════════════════════════════════════════════════════
with tab_records:
    st.subheader("📜 Registros recientes")
    records = chain.list_records(limit=30)
    if records:
        for i, rec in enumerate(records):
            rec_type = "🔮 SIM" if rec.get("scene_id", "").startswith("sim") or rec.get("model_version", "").startswith("simulator") else "📸 IMG"
            with st.expander(
                f"#{i+1} {rec_type} | {rec.get('scene_id', 'N/A')} | {rec.get('timestamp_utc', '')[:19]}"
            ):
                st.json(rec)
    else:
        st.info("No hay registros aún. Analiza una imagen o ejecuta una simulación.")
