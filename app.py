"""Streamlit frontend for Traffic Aerial Analysis System."""

from __future__ import annotations

import json
import sys
from datetime import datetime, date, time
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
)
from src.blockchain import get_blockchain_adapter
from src.visualization.overlays import (
    draw_detections,
    generate_heatmap,
    draw_density_grid,
    draw_collisions,
)
from src.simulator import TrafficSimulator
import config

APP_ICON = Image.open("assets/urban1.png")

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban VS",
    page_icon=APP_ICON,
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Compact header */
    .block-container { padding-top: 1.5rem; }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 8px; padding: 12px;
        border: 1px solid #e9ecef;
    }
    [data-testid="stMetric"] label { font-size: 0.8rem !important; }
    /* Evidence box */
    .evidence-box {
        background: #1a1a2e; color: #e0e0e0; border-radius: 8px;
        padding: 16px; margin: 8px 0; font-family: monospace; font-size: 0.85rem;
        border-left: 4px solid #0f3460;
    }
    .evidence-box .hash-label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; }
    .evidence-box .hash-value { color: #38bdf8; word-break: break-all; }
    /* Density banner */
    .density-banner {
        text-align: center; padding: 14px; border-radius: 10px; margin: 8px 0;
    }
    .density-banner h2 { margin: 0; font-size: 1.5rem; }
    /* Section divider */
    .section-title {
        font-size: 1.1rem; font-weight: 600; margin-top: 1rem;
        padding-bottom: 4px; border-bottom: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

st.title("Urban VS")
st.caption("Deteccion vehicular en imagenes aereas con evidencia verificable en blockchain")


# ── Helpers ────────────────────────────────────────────────────────────
_CLS_LABELS = {"car": "Coche", "motorcycle": "Moto", "truck": "Camion", "bus": "Autobus", "bicycle": "Bicicleta"}
_ZONE_LABELS = {"upper": "Superior", "middle": "Central", "lower": "Inferior"}
_SEV_LABELS = {"HIGH": "ALTO", "MEDIUM": "MEDIO", "WARNING": "AVISO"}
_SEV_COLORS = {"HIGH": "red", "MEDIUM": "orange", "WARNING": "yellow"}
_DENSITY_STYLES = {
    "FLUIDO":   ("#22c55e", "#22c55e22"),
    "MODERADO": ("#f97316", "#f9731622"),
    "DENSO":    ("#ef4444", "#ef444422"),
    "SATURADO": ("#8b5cf6", "#8b5cf622"),
}


def render_density_banner(level: str):
    border, bg = _DENSITY_STYLES.get(level, ("#94a3b8", "#94a3b822"))
    st.markdown(
        f"<div class='density-banner' style='background:{bg}; border:2px solid {border};'>"
        f"<h2>Densidad de trafico: {level}</h2></div>",
        unsafe_allow_html=True,
    )


def render_evidence_box(analysis_hash: str):
    st.markdown(
        f"<div class='evidence-box'>"
        f"<div class='hash-label'>Huella digital del analisis (SHA-256)</div>"
        f"<div class='hash-value'>{analysis_hash}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_register_button(evidence, chain_ref, button_key: str):
    if st.button("Registrar en blockchain", type="primary", key=button_key):
        with st.spinner("Registrando..."):
            tx_result = chain_ref.register(evidence)

        status = tx_result.get("status", "unknown")
        if status == "on_chain":
            st.success(f"Registrado en blockchain: `{tx_result['tx_id']}`")
            st.markdown(f"[Ver en WhatsOnChain]({tx_result['explorer_url']})")
        elif status == "local_fallback":
            st.warning("Guardado localmente (sin fondos on-chain).")
            if tx_result.get("address"):
                st.caption(f"Address BSV: `{tx_result['address']}`")
        else:
            st.info(f"Registrado localmente: `{tx_result.get('tx_id', 'N/A')}`")

        with st.expander("Detalle del registro"):
            st.json(tx_result)


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

detector = load_detector()
analyzer = load_analyzer()
chain = load_chain()
simulator = load_simulator()

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(APP_ICON, width=80)
    st.markdown("### Configuracion")
    dataset_id = st.selectbox("Dataset", ["uav_traffic", "roundabout", "upload"])
    is_roundabout = st.checkbox("Escena de rotonda", value=(dataset_id == "roundabout"))
    st.markdown("---")
    st.markdown("**Modelo**")
    st.code(detector.model_version, language="text")
    st.markdown("**Blockchain**")
    if getattr(chain, "is_configured", False):
        st.success(f"BSV {getattr(chain, 'network', 'main')}")
        addr = getattr(chain, "address", None)
        if addr:
            st.caption(f"`{addr[:8]}...{addr[-6:]}`")
    else:
        st.info("Ledger local (demo)")

# ── Tabs ───────────────────────────────────────────────────────────────
tab_analyze, tab_compare, tab_simulate, tab_verify, tab_records = st.tabs(
    ["Analizar", "Comparar capturas", "Simulador", "Verificar", "Registros"]
)

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: ANALYZE
# ═══════════════════════════════════════════════════════════════════════
with tab_analyze:
    uploaded_file = st.file_uploader(
        "Sube una imagen aerea (JPG / PNG)",
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

            with st.spinner("Detectando vehiculos..."):
                detections = detector.detect(img_bgr)

            det_f = [d for d in detections if d.confidence >= config.CONFIDENCE_THRESHOLD]
            metrics = analyzer.analyze(det_f, h, w, is_roundabout=is_roundabout)

            m_collisions = getattr(metrics, "collisions", None) or []
            m_collision_count = getattr(metrics, "collision_count", 0)

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
            analysis_hash = compute_hash(payload)
            evidence = build_evidence_record(payload)

            # ── Visualizations ──
            col_img, col_heat = st.columns(2)
            with col_img:
                st.markdown("<p class='section-title'>Detecciones</p>", unsafe_allow_html=True)
                img_det = draw_detections(img_bgr, det_f)
                st.image(
                    cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB),
                    caption=f"{len(det_f)} vehiculos detectados",
                    use_container_width=True,
                )
            with col_heat:
                st.markdown("<p class='section-title'>Mapa de calor</p>", unsafe_allow_html=True)
                img_heat = generate_heatmap(img_bgr, det_f)
                st.image(
                    cv2.cvtColor(img_heat, cv2.COLOR_BGR2RGB),
                    caption="Concentracion vehicular ponderada",
                    use_container_width=True,
                )

            if m_collisions:
                st.markdown("<p class='section-title'>Zonas de posibles incidentes</p>", unsafe_allow_html=True)
                img_col = draw_collisions(img_bgr, m_collisions, det_f)
                img_col = draw_detections(img_col, det_f)
                st.image(
                    cv2.cvtColor(img_col, cv2.COLOR_BGR2RGB),
                    caption=f"{m_collision_count} posibles incidentes detectados",
                    use_container_width=True,
                )

            with st.expander("Mapa de densidad por celdas"):
                img_grid = draw_density_grid(img_bgr, metrics.density_grid)
                st.image(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB), use_container_width=True)

            # ── Metrics ──
            st.markdown("---")
            render_density_banner(metrics.traffic_density)

            st.markdown("")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total vehiculos", metrics.total_vehicles)
            m2.metric("Cobertura vehicular", f"{metrics.occupancy_pct:.1f}%")
            m3.metric("Nivel de riesgo", metrics.risk_level)
            m4.metric("Posibles incidentes", m_collision_count)
            if metrics.roundabout_occupancy_pct is not None:
                m5.metric(
                    "Rotonda",
                    metrics.roundabout_level,
                    delta=f"{metrics.roundabout_occupancy_pct:.0f}% ocupacion",
                    delta_color="off",
                )
            else:
                m5.metric("Distribucion", f"{len(metrics.zone_occupancy)} zonas")

            col_cls, col_zone = st.columns(2)
            with col_cls:
                st.markdown("**Conteo por clase:**")
                for cls, cnt in metrics.counts.items():
                    st.write(f"- {_CLS_LABELS.get(cls, cls)}: **{cnt}**")
            with col_zone:
                st.markdown("**Distribucion por zona:**")
                for zone, pct in metrics.zone_occupancy.items():
                    st.progress(pct / 100, text=f"{_ZONE_LABELS.get(zone, zone)}: {pct:.1f}%")

            # ── Incidents ──
            if m_collisions:
                st.markdown("---")
                st.markdown("<p class='section-title'>Detalle de posibles incidentes</p>", unsafe_allow_html=True)

                sev_counts = {"HIGH": 0, "MEDIUM": 0, "WARNING": 0}
                for e in m_collisions:
                    sev = e.get("severity", "WARNING")
                    if sev in sev_counts:
                        sev_counts[sev] += 1
                st.caption(
                    f"ALTO: {sev_counts['HIGH']}  |  MEDIO: {sev_counts['MEDIUM']}  |  AVISO: {sev_counts['WARNING']}"
                )

                for i, col in enumerate(m_collisions):
                    sev = col.get("severity", "WARNING")
                    etype = col.get("type", "EVENT")
                    sev_color = _SEV_COLORS.get(sev, "gray")
                    sev_label = _SEV_LABELS.get(sev, sev)
                    iou = col.get("iou", 0.0)
                    dist_px = col.get("distance", None)
                    dist_norm = col.get("distance_norm", None)
                    dist_txt = f"{dist_px:.0f}px" if isinstance(dist_px, (int, float)) else "N/A"
                    norm_txt = f"{dist_norm:.3f}" if isinstance(dist_norm, (int, float)) else "N/A"

                    st.markdown(
                        f"**#{i+1}** — {etype} — :{sev_color}[{sev_label}] — "
                        f"{col.get('vehicle_a_class','?')} vs {col.get('vehicle_b_class','?')} "
                        f"(IoU: {iou:.3f}, dist: {dist_txt}, norm: {norm_txt})"
                    )

            # ── Evidence ──
            st.markdown("---")
            st.markdown("<p class='section-title'>Trazabilidad y evidencia</p>", unsafe_allow_html=True)
            st.caption(
                "Se genera una huella digital unica (hash) del analisis. "
                "Registrarla en blockchain garantiza que los resultados no se puedan alterar."
            )
            render_evidence_box(analysis_hash)
            render_register_button(evidence, chain, "reg_analyze")

            with st.expander("Ver datos del analisis (JSON)"):
                st.code(canonical_json(payload), language="json")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: COMPARE CAPTURES
# ═══════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("Sube dos imagenes aereas para comparar la situacion del trafico.")

    cmp_col_a, cmp_col_b = st.columns(2)
    with cmp_col_a:
        file_a = st.file_uploader("Captura A", type=["jpg", "jpeg", "png"], key="cmp_a")
    with cmp_col_b:
        file_b = st.file_uploader("Captura B", type=["jpg", "jpeg", "png"], key="cmp_b")

    cmp_roundabout = st.checkbox("Ambas son rotondas", value=False, key="cmp_roundabout")

    if file_a is not None and file_b is not None:
        img_a = cv2.imdecode(np.frombuffer(file_a.read(), np.uint8), cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(np.frombuffer(file_b.read(), np.uint8), cv2.IMREAD_COLOR)

        if img_a is None or img_b is None:
            st.error("No se pudo leer una o ambas imagenes.")
        else:
            with st.spinner("Analizando ambas capturas..."):
                det_a = detector.detect(img_a)
                det_a = [d for d in det_a if d.confidence >= config.CONFIDENCE_THRESHOLD]
                h_a, w_a = img_a.shape[:2]
                met_a = analyzer.analyze(det_a, h_a, w_a, is_roundabout=cmp_roundabout)

                det_b = detector.detect(img_b)
                det_b = [d for d in det_b if d.confidence >= config.CONFIDENCE_THRESHOLD]
                h_b, w_b = img_b.shape[:2]
                met_b = analyzer.analyze(det_b, h_b, w_b, is_roundabout=cmp_roundabout)

            # ── Side-by-side detections ──
            st.markdown("---")
            st.markdown("<p class='section-title'>Detecciones</p>", unsafe_allow_html=True)
            vis_a, vis_b = st.columns(2)
            with vis_a:
                st.caption(f"**Captura A** — {file_a.name}")
                st.image(cv2.cvtColor(draw_detections(img_a, det_a), cv2.COLOR_BGR2RGB), use_container_width=True)
            with vis_b:
                st.caption(f"**Captura B** — {file_b.name}")
                st.image(cv2.cvtColor(draw_detections(img_b, det_b), cv2.COLOR_BGR2RGB), use_container_width=True)

            # ── Side-by-side heatmaps ──
            st.markdown("<p class='section-title'>Mapas de calor</p>", unsafe_allow_html=True)
            heat_a, heat_b = st.columns(2)
            with heat_a:
                st.image(cv2.cvtColor(generate_heatmap(img_a, det_a), cv2.COLOR_BGR2RGB), use_container_width=True)
            with heat_b:
                st.image(cv2.cvtColor(generate_heatmap(img_b, det_b), cv2.COLOR_BGR2RGB), use_container_width=True)

            # ── Density banners side by side ──
            den_a, den_b = st.columns(2)
            with den_a:
                render_density_banner(met_a.traffic_density)
            with den_b:
                render_density_banner(met_b.traffic_density)

            # ── Metrics comparison table ──
            st.markdown("---")
            st.markdown("<p class='section-title'>Comparativa de metricas</p>", unsafe_allow_html=True)

            def _delta_str(va, vb):
                diff = vb - va
                sign = "+" if diff > 0 else ""
                return f"{sign}{diff}" if isinstance(diff, int) else f"{sign}{diff:.1f}"

            rows = [
                ("Total vehiculos", str(met_a.total_vehicles), str(met_b.total_vehicles),
                 _delta_str(met_a.total_vehicles, met_b.total_vehicles)),
                ("Densidad de trafico", met_a.traffic_density, met_b.traffic_density, "—"),
                ("Cobertura vehicular", f"{met_a.occupancy_pct:.1f}%", f"{met_b.occupancy_pct:.1f}%",
                 f"{_delta_str(met_a.occupancy_pct, met_b.occupancy_pct)}%"),
                ("Nivel de riesgo", met_a.risk_level, met_b.risk_level, "—"),
                ("Posibles incidentes", str(met_a.collision_count), str(met_b.collision_count),
                 _delta_str(met_a.collision_count, met_b.collision_count)),
            ]

            if cmp_roundabout and met_a.roundabout_level and met_b.roundabout_level:
                rows.append((
                    "Rotonda",
                    f"{met_a.roundabout_level} ({met_a.roundabout_occupancy_pct:.0f}%)",
                    f"{met_b.roundabout_level} ({met_b.roundabout_occupancy_pct:.0f}%)",
                    f"{_delta_str(met_a.roundabout_occupancy_pct, met_b.roundabout_occupancy_pct)}%",
                ))

            all_classes = sorted(set(list(met_a.counts.keys()) + list(met_b.counts.keys())))
            for cls in all_classes:
                ca = met_a.counts.get(cls, 0)
                cb = met_b.counts.get(cls, 0)
                rows.append((_CLS_LABELS.get(cls, cls), str(ca), str(cb), _delta_str(ca, cb)))

            table_md = "| Metrica | Captura A | Captura B | Diferencia |\n"
            table_md += "|:--------|:---------:|:---------:|:----------:|\n"
            for name, va, vb, delta in rows:
                table_md += f"| {name} | {va} | {vb} | {delta} |\n"
            st.markdown(table_md)

            # ── Zone comparison ──
            st.markdown("<p class='section-title'>Distribucion por zona</p>", unsafe_allow_html=True)
            zone_a_col, zone_b_col = st.columns(2)
            with zone_a_col:
                st.caption("**Captura A**")
                for zone, pct in met_a.zone_occupancy.items():
                    st.progress(pct / 100, text=f"{_ZONE_LABELS.get(zone, zone)}: {pct:.1f}%")
            with zone_b_col:
                st.caption("**Captura B**")
                for zone, pct in met_b.zone_occupancy.items():
                    st.progress(pct / 100, text=f"{_ZONE_LABELS.get(zone, zone)}: {pct:.1f}%")

            # ── Evidence for comparison ──
            st.markdown("---")
            st.markdown("<p class='section-title'>Trazabilidad de la comparacion</p>", unsafe_allow_html=True)
            st.caption("Cada captura genera su propia huella digital. Se pueden registrar ambas de forma independiente.")

            # Build payloads for each
            payload_a = build_analysis_payload(
                scene_id=file_a.name, dataset_id=dataset_id,
                counts=met_a.counts, total_vehicles=met_a.total_vehicles,
                density_grid=met_a.density_grid, occupancy_pct=met_a.occupancy_pct,
                zone_occupancy=met_a.zone_occupancy, risk_level=met_a.risk_level,
                model_version=detector.model_version, is_roundabout=met_a.is_roundabout,
                roundabout_occupancy_pct=met_a.roundabout_occupancy_pct,
                collision_count=met_a.collision_count,
                collisions=met_a.collisions if met_a.collisions else None,
            )
            payload_b = build_analysis_payload(
                scene_id=file_b.name, dataset_id=dataset_id,
                counts=met_b.counts, total_vehicles=met_b.total_vehicles,
                density_grid=met_b.density_grid, occupancy_pct=met_b.occupancy_pct,
                zone_occupancy=met_b.zone_occupancy, risk_level=met_b.risk_level,
                model_version=detector.model_version, is_roundabout=met_b.is_roundabout,
                roundabout_occupancy_pct=met_b.roundabout_occupancy_pct,
                collision_count=met_b.collision_count,
                collisions=met_b.collisions if met_b.collisions else None,
            )
            hash_a = compute_hash(payload_a)
            hash_b = compute_hash(payload_b)
            ev_a = build_evidence_record(payload_a)
            ev_b = build_evidence_record(payload_b)

            ev_col_a, ev_col_b = st.columns(2)
            with ev_col_a:
                st.markdown("**Captura A**")
                render_evidence_box(hash_a)
                render_register_button(ev_a, chain, "reg_cmp_a")
            with ev_col_b:
                st.markdown("**Captura B**")
                render_evidence_box(hash_b)
                render_register_button(ev_b, chain, "reg_cmp_b")

    elif file_a is not None or file_b is not None:
        st.info("Sube ambas imagenes para iniciar la comparacion.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════
with tab_simulate:
    st.markdown("Simula el estado del trafico para una fecha, hora y tipo de escena.")

    col_dt, col_scene, col_meta, col_id = st.columns([1, 1, 1, 1])
    with col_dt:
        sim_date = st.date_input("Fecha", value=date.today(), key="sim_date")
        sim_time = st.time_input("Hora", value=time(8, 0), key="sim_time")
    with col_scene:
        scene_type = st.selectbox(
            "Tipo de escena",
            ["urban_road", "roundabout", "highway"],
            format_func=lambda x: {"urban_road": "Via urbana", "roundabout": "Rotonda", "highway": "Autovia"}[x],
            key="sim_scene_type",
        )
    with col_meta:
        weather = st.selectbox(
            "Meteorologia",
            ["soleado", "nublado", "lluvia", "niebla"],
            format_func=lambda x: x.capitalize(),
            key="sim_weather",
        )
        event_level = st.selectbox(
            "Evento especial",
            ["none", "low", "medium", "high"],
            format_func=lambda x: {"none": "Ninguno", "low": "Bajo", "medium": "Medio", "high": "Alto"}[x],
            key="sim_event",
        )
    with col_id:
        sim_scene_id = st.text_input("ID de escena", value="zona_centro_01", key="sim_scene_id")
        st.caption("Ej: zona_centro_01, rotonda_norte")

    sim_datetime = datetime.combine(sim_date, sim_time)

    with st.expander("Ajustes manuales (opcional)"):
        ov_col1, ov_col2, ov_col3 = st.columns(3)
        with ov_col1:
            ov_total = st.number_input("Total vehiculos (0=auto)", min_value=0, max_value=200, value=0, key="sim_total")
        with ov_col2:
            ov_density = st.number_input("Densidad (0=auto)", min_value=0.0, max_value=20.0, value=0.0, step=0.5, key="sim_density")
        with ov_col3:
            ov_occupancy = st.number_input("Cobertura % (0=auto)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sim_occ")

        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        ov_cars = vc1.number_input("Coches", min_value=0, max_value=100, value=0, key="sim_car")
        ov_motos = vc2.number_input("Motos", min_value=0, max_value=50, value=0, key="sim_moto")
        ov_buses = vc3.number_input("Buses", min_value=0, max_value=20, value=0, key="sim_bus")
        ov_trucks = vc4.number_input("Camiones", min_value=0, max_value=30, value=0, key="sim_truck")
        ov_cycles = vc5.number_input("Bicicletas", min_value=0, max_value=30, value=0, key="sim_cycle")
        manual_counts_sum = ov_cars + ov_motos + ov_buses + ov_trucks + ov_cycles

    if st.button("Simular escenario", type="primary", key="btn_simulate"):
        override_total = ov_total if ov_total > 0 else None
        override_density = ov_density if ov_density > 0 else None
        override_occupancy = ov_occupancy if ov_occupancy > 0 else None

        override_counts = None
        if manual_counts_sum > 0:
            override_counts = {}
            if ov_cars > 0: override_counts["car"] = ov_cars
            if ov_motos > 0: override_counts["motorcycle"] = ov_motos
            if ov_buses > 0: override_counts["bus"] = ov_buses
            if ov_trucks > 0: override_counts["truck"] = ov_trucks
            if ov_cycles > 0: override_counts["cycle"] = ov_cycles

        result = simulator.simulate(
            sim_datetime=sim_datetime, scene_type=scene_type,
            scene_id=sim_scene_id, weather=weather, event_level=event_level,
            override_total=override_total, override_counts=override_counts,
            override_density=override_density, override_occupancy=override_occupancy,
        )

        state = result["traffic_state"]
        emoji = simulator.get_state_emoji(state)
        color = simulator.get_state_color(state)

        st.markdown("---")
        st.markdown(
            f"<div style='text-align:center; padding:20px; border-radius:12px; "
            f"background-color:{color}22; border:2px solid {color};'>"
            f"<h1 style='color:{color}; margin:0;'>{emoji} {state}</h1>"
            f"<p style='font-size:1.1em; margin:4px 0 0 0;'>"
            f"{sim_datetime.strftime('%A %d/%m/%Y %H:%M')} | "
            f"{'Rotonda' if scene_type == 'roundabout' else 'Via urbana' if scene_type == 'urban_road' else 'Autovia'}"
            f"</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Vehiculos", result["total_vehicles"])
        mc2.metric("Cobertura", f"{result['occupancy_pct']:.1f}%")
        mc3.metric("Riesgo", result["risk_level"])
        mc4.metric("Carga", f"{result['load_ratio']:.0%}")
        mc5.metric("Capacidad", result["capacity"])

        st.markdown("**Distribucion por tipo:**")
        counts_cols = st.columns(len(result["counts"]) or 1)
        for i, (cls, cnt) in enumerate(result["counts"].items()):
            counts_cols[i % len(counts_cols)].metric(_CLS_LABELS.get(cls, cls.capitalize()), cnt)

        col_grid, col_zones = st.columns(2)
        with col_grid:
            st.markdown("**Mapa de densidad simulado:**")
            grid = result["density_grid"]
            grid_html = "<table style='width:100%; border-collapse:collapse;'>"
            max_val = max(max(row) for row in grid) if grid else 1
            for row in grid:
                grid_html += "<tr>"
                for val in row:
                    intensity = val / max(max_val, 1)
                    r = int(255 * intensity)
                    g = int(255 * (1 - intensity))
                    grid_html += (
                        f"<td style='text-align:center; padding:10px; "
                        f"background-color:rgba({r},{g},0,0.6); "
                        f"color:white; font-weight:bold; border:1px solid #444; border-radius:4px;'>"
                        f"{val}</td>"
                    )
                grid_html += "</tr>"
            grid_html += "</table>"
            st.markdown(grid_html, unsafe_allow_html=True)

        with col_zones:
            st.markdown("**Distribucion por zona:**")
            for zone, pct in result["zone_occupancy"].items():
                label = _ZONE_LABELS.get(zone, zone)
                st.progress(min(pct / 100, 1.0), text=f"{label}: {pct:.1f}%")

            if result["roundabout_occupancy_pct"] is not None:
                st.markdown(f"**Rotonda:** {result['roundabout_occupancy_pct']:.1f}%")

        # ── Evidence ──
        st.markdown("---")
        st.markdown("<p class='section-title'>Trazabilidad de la simulacion</p>", unsafe_allow_html=True)
        st.caption("La simulacion tambien genera una huella digital para garantizar su integridad.")
        sim_hash = compute_hash(result)
        render_evidence_box(sim_hash)
        sim_evidence = build_evidence_record(result)
        render_register_button(sim_evidence, chain, "reg_sim")

        with st.expander("Ver datos de la simulacion (JSON)"):
            st.code(canonical_json(result), language="json")

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: VERIFY
# ═══════════════════════════════════════════════════════════════════════
with tab_verify:
    st.markdown("Comprueba si un analisis o simulacion fue registrado y no ha sido alterado.")
    st.markdown("")

    hash_input = st.text_input(
        "Huella digital (hash SHA-256)",
        placeholder="Pega aqui el hash del analisis que quieres verificar...",
        key="verify_hash",
    )

    col_verify_btn, col_verify_spacer = st.columns([1, 3])
    with col_verify_btn:
        verify_clicked = st.button("Verificar", type="primary", key="btn_verify")

    if verify_clicked:
        if hash_input:
            with st.spinner("Buscando en el registro..."):
                record = chain.verify(hash_input)
            if record:
                st.session_state["verify_result"] = record
            else:
                st.session_state["verify_result"] = "not_found"
        else:
            st.warning("Introduce una huella digital para verificar.")

    if "verify_result" in st.session_state:
        result = st.session_state["verify_result"]
        if result == "not_found":
            st.error("No se encontro ningun registro con esa huella digital.")
        else:
            txid = result.get("tx_id", "")
            is_onchain = not txid.startswith("local_") and txid

            if result.get("on_chain_verified"):
                st.success(
                    f"Verificado en blockchain ({result.get('confirmations', 0)} confirmaciones)"
                )
                if result.get("explorer_url"):
                    st.markdown(f"[Ver transaccion en WhatsOnChain]({result['explorer_url']})")
            elif is_onchain:
                st.info("Registro encontrado en blockchain (pendiente de confirmacion)")
                explorer = (
                    f"https://test.whatsonchain.com/tx/{txid}"
                    if getattr(chain, "network", "") == "testnet"
                    else f"https://whatsonchain.com/tx/{txid}"
                )
                st.markdown(f"[Ver transaccion en WhatsOnChain]({explorer})")
            else:
                st.success("Registro encontrado en el ledger local")

            st.markdown("")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown("**Datos del registro**")
                st.write(f"- **Escena:** {result.get('scene_id', 'N/A')}")
                st.write(f"- **Modelo:** {result.get('model_version', 'N/A')}")
                st.write(f"- **Fecha:** {result.get('timestamp_utc', 'N/A')[:19]}")
            with col_info2:
                st.markdown("**Identificadores**")
                st.write(f"- **TX:** `{txid or 'N/A'}`")
                eid = result.get('evidence_id', 'N/A')
                st.write(f"- **Evidence ID:** `{eid[:16]}...`" if len(eid) > 16 else f"- **Evidence ID:** `{eid}`")
                if result.get("on_chain_verified"):
                    st.write(f"- **Confirmaciones:** {result.get('confirmations', 0)}")

            with st.expander("Ver registro completo (JSON)"):
                st.json(result)

    st.markdown("---")
    st.markdown("<p class='section-title'>Recalcular huella digital desde datos</p>", unsafe_allow_html=True)
    st.caption("Pega el JSON de un analisis para verificar que la huella digital coincide.")
    json_input = st.text_area("JSON del analisis", key="verify_json", height=120)
    if st.button("Recalcular", key="btn_recalc"):
        if json_input:
            try:
                data = json.loads(json_input)
                recalc = compute_hash(data)
                render_evidence_box(recalc)
                if hash_input and recalc == hash_input:
                    st.success("Las huellas coinciden. Los datos no fueron alterados.")
                elif hash_input:
                    st.error("Las huellas NO coinciden. Los datos fueron modificados.")
            except json.JSONDecodeError:
                st.error("El JSON introducido no es valido.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: RECORDS
# ═══════════════════════════════════════════════════════════════════════
with tab_records:
    st.markdown("Ultimos registros guardados en el ledger.")
    st.markdown("")

    records = chain.list_records(limit=30)
    if records:
        for i, rec in enumerate(records):
            is_sim = (
                rec.get("model_version", "").startswith("simulator")
                or rec.get("dataset_id", "") == "simulation"
                or rec.get("type", "") == "simulation"
            )
            rec_type = "Simulacion" if is_sim else "Analisis"
            rec_icon = "SIM" if is_sim else "IMG"
            scene = rec.get("scene_id", "N/A")
            ts = rec.get("timestamp_utc", "")[:19]

            with st.expander(f"**{rec_icon}** | {rec_type} | {scene} | {ts}"):
                col_rec1, col_rec2 = st.columns([2, 1])
                with col_rec1:
                    st.write(f"**Escena:** {scene}")
                    st.write(f"**Modelo:** {rec.get('model_version', 'N/A')}")
                    st.write(f"**Fecha:** {ts}")
                with col_rec2:
                    h = rec.get("analysis_hash", "N/A")
                    st.write(f"**Hash:** `{h[:24]}...`" if len(h) > 24 else f"**Hash:** `{h}`")
                    txid = rec.get("tx_id", "N/A")
                    st.write(f"**TX:** `{txid[:16]}...`" if len(txid) > 16 else f"**TX:** `{txid}`")
                with st.expander("JSON completo"):
                    st.json(rec)
    else:
        st.info("No hay registros aun. Analiza una imagen o ejecuta una simulacion para generar el primero.")
