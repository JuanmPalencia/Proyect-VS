"""Streamlit frontend for Traffic Aerial Analysis System (Urban VS)."""

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
st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; }

    /* Sidebar header */
    .sidebar-title {
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
        padding: 0;
        color: #f8fafc;
    }
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Metric readability in dark theme */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        padding: 14px !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        backdrop-filter: blur(2px);
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    /* Evidence box */
    .evidence-box {
        background: #0b1220;
        color: #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.9rem;
        border: 1px solid rgba(56,189,248,0.25);
        box-shadow: 0 0 0 1px rgba(15,23,42,0.6) inset;
    }
    .evidence-box .hash-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: 6px;
    }
    .evidence-box .hash-value { color: #38bdf8; word-break: break-all; }

    /* Density banner */
    .density-banner {
        text-align: center;
        padding: 14px;
        border-radius: 12px;
        margin: 8px 0;
    }
    .density-banner h2 { margin: 0; font-size: 1.5rem; }

    /* Section divider */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1rem;
        padding-bottom: 6px;
        border-bottom: 2px solid rgba(148,163,184,0.25);
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        border: 1px solid rgba(148,163,184,0.25);
        background: rgba(255,255,255,0.06);
        color: #e2e8f0;
    }

    /* Simulator hero card */
    .hero {
        border-radius: 16px;
        padding: 18px 18px;
        border: 1px solid rgba(148,163,184,0.22);
        background: rgba(255,255,255,0.04);
        box-shadow: 0 0 0 1px rgba(15,23,42,0.65) inset;
        margin: 10px 0 14px 0;
    }
    .hero h1{
        margin:0;
        font-size: 1.6rem;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .hero p{
        margin:6px 0 0 0;
        color: #94a3b8;
        font-size: .95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Labels / styles ────────────────────────────────────────────────────
_CLS_LABELS = {"car": "Coche", "motorcycle": "Moto", "truck": "Camion", "bus": "Autobus", "bicycle": "Bicicleta"}
_ZONE_LABELS = {"upper": "Superior", "middle": "Central", "lower": "Inferior"}
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
    # Sin expanders aquí para evitar anidamientos accidentales
    if st.button("Registrar en blockchain", type="primary", key=button_key):
        with st.spinner("Registrando..."):
            tx_result = chain_ref.register(evidence)
        st.session_state[f"tx_result_{button_key}"] = tx_result

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

    tx_key = f"tx_result_{button_key}"
    if tx_key in st.session_state:
        if st.checkbox("Ver detalle del registro", key=f"show_tx_detail_{button_key}"):
            st.json(st.session_state[tx_key])


def detect_roundabout(img_bgr: np.ndarray) -> tuple[bool, float]:
    """
    Heurística rápida para sugerir rotonda (solo ayuda UI).
    """
    try:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        min_dim = min(h, w)
        min_r = max(12, int(min_dim * 0.06))
        max_r = int(min_dim * 0.45)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim * 0.25,
            param1=120,
            param2=35,
            minRadius=min_r,
            maxRadius=max_r,
        )
        hough_hit = circles is not None and len(circles[0]) > 0

        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_circ = 0.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < (h * w) * 0.01:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4 * np.pi * (area / (peri * peri))
            best_circ = max(best_circ, float(circularity))

        score = 0.0
        if hough_hit:
            score += 0.65
        score += max(0.0, min(0.35, (best_circ - 0.45) * 0.7))
        score = max(0.0, min(1.0, score))
        return (score >= 0.60), score
    except Exception:
        return False, 0.0


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

# ── Sidebar (simple) ──────────────────────────────────────────────────
with st.sidebar:
    c1, c2 = st.columns([1, 2.4], vertical_alignment="center")
    with c1:
        st.image(APP_ICON, width=70)
    with c2:
        st.markdown("<p class='sidebar-title'>Urban VS</p>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-subtitle'>Analítica de tráfico aéreo</div>", unsafe_allow_html=True)
    st.markdown("---")

st.caption("Deteccion vehicular en imagenes aereas con evidencia verificable en blockchain")

# ── Tabs arriba ───────────────────────────────────────────────────────
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

            auto_is_rb, auto_score = detect_roundabout(img_bgr)
            force_roundabout = st.toggle(
                "Forzar: la escena es una rotonda",
                value=auto_is_rb,
                help="Actívalo si el detector no acierta. Por defecto toma una sugerencia automática.",
            )
            is_roundabout = bool(force_roundabout)
            dataset_id = "roundabout" if is_roundabout else "uav_traffic"

            st.markdown(
                f"<span class='badge'>Sugerencia automática: {'Rotonda' if auto_is_rb else 'Vía urbana'} (score {auto_score:.2f})</span>",
                unsafe_allow_html=True,
            )

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

            # Oculto: imagen de incidentes
            if m_collisions:
                with st.expander(f"Zonas de posibles incidentes ({m_collision_count})", expanded=False):
                    img_col = draw_collisions(img_bgr, m_collisions, det_f)
                    img_col = draw_detections(img_col, det_f)
                    st.image(
                        cv2.cvtColor(img_col, cv2.COLOR_BGR2RGB),
                        caption=f"{m_collision_count} posibles incidentes detectados",
                        use_container_width=True,
                    )

            with st.expander("Mapa de densidad por celdas", expanded=False):
                img_grid = draw_density_grid(img_bgr, metrics.density_grid)
                st.image(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB), use_container_width=True)

            # ── Metrics ──
            st.markdown("---")
            render_density_banner(metrics.traffic_density)

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

            # ✅ Detalle de incidentes críticos (SIN matrícula)
            if m_collisions:
                st.markdown("---")
                st.markdown("<p class='section-title'>Detalle de incidentes criticos</p>", unsafe_allow_html=True)

                sev_counts = {"HIGH": 0, "MEDIUM": 0, "WARNING": 0}
                for e in m_collisions:
                    sev = (e.get("severity") or "WARNING").upper()
                    if sev in sev_counts:
                        sev_counts[sev] += 1

                st.caption(
                    f"Resumen: HIGH={sev_counts['HIGH']} | MEDIUM={sev_counts['MEDIUM']} | WARNING={sev_counts['WARNING']}"
                )

                for i, ev in enumerate(m_collisions, start=1):
                    etype = ev.get("type", "EVENT")
                    sev = (ev.get("severity") or "WARNING").upper()
                    a_cls = ev.get("vehicle_a_class", "?")
                    b_cls = ev.get("vehicle_b_class", "?")
                    iou = float(ev.get("iou", 0.0) or 0.0)
                    dist_px = ev.get("distance", None)
                    dist_norm = ev.get("distance_norm", None)

                    dist_txt = f"{dist_px:.0f}px" if isinstance(dist_px, (int, float)) else "N/A"
                    norm_txt = f"{dist_norm:.3f}" if isinstance(dist_norm, (int, float)) else "N/A"

                    st.markdown(
                        f"**Evento #{i}** — **{etype}** — **{sev}** — "
                        f"{a_cls} vs {b_cls} "
                        f"(IoU: {iou:.3f}, dist: {dist_txt}, dist_norm: {norm_txt})"
                    )

                st.markdown("")
                st.markdown("**Vehiculos involucrados:**")

                involved = []
                for ev in m_collisions:
                    if isinstance(ev.get("vehicles"), list):
                        involved.extend([v for v in ev["vehicles"] if isinstance(v, dict)])
                    for k in ("vehicle_a", "vehicle_b"):
                        if isinstance(ev.get(k), dict):
                            involved.append(ev[k])
                    if ev.get("vehicle_a_class"):
                        involved.append({"class": ev.get("vehicle_a_class")})
                    if ev.get("vehicle_b_class"):
                        involved.append({"class": ev.get("vehicle_b_class")})

                seen = set()
                cleaned = []
                for v in involved:
                    cls = v.get("class") or v.get("cls") or v.get("label") or "?"
                    color = v.get("color") or v.get("vehicle_color") or "N/A"
                    pos = v.get("pos") or v.get("position") or v.get("zone") or "N/A"
                    key = (cls, color, pos)
                    if key not in seen:
                        seen.add(key)
                        cleaned.append((cls, color, pos))

                if cleaned:
                    for cls, color, pos in cleaned:
                        st.write(f"- **{cls}** | Color: {color} | Pos: {pos}")
                else:
                    st.write("- (No hay detalles adicionales de vehículos en el payload de incidentes)")

            # ── Evidence ──
            st.markdown("---")
            st.markdown("<p class='section-title'>Trazabilidad y evidencia</p>", unsafe_allow_html=True)
            render_evidence_box(analysis_hash)
            render_register_button(evidence, chain, "reg_analyze")

            with st.expander("Ver datos del analisis (JSON)", expanded=False):
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

    if file_a is not None and file_b is not None:
        img_a = cv2.imdecode(np.frombuffer(file_a.read(), np.uint8), cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(np.frombuffer(file_b.read(), np.uint8), cv2.IMREAD_COLOR)

        if img_a is None or img_b is None:
            st.error("No se pudo leer una o ambas imagenes.")
        else:
            auto_rb_a, sc_a = detect_roundabout(img_a)
            auto_rb_b, sc_b = detect_roundabout(img_b)

            cta, ctb = st.columns(2)
            with cta:
                force_a = st.toggle("Escena rotonda (A)", value=auto_rb_a)
                st.markdown(
                    f"<span class='badge'>Sugerencia A: {'Rotonda' if auto_rb_a else 'Vía urbana'} ({sc_a:.2f})</span>",
                    unsafe_allow_html=True,
                )
            with ctb:
                force_b = st.toggle("Escena rotonda (B)", value=auto_rb_b)
                st.markdown(
                    f"<span class='badge'>Sugerencia B: {'Rotonda' if auto_rb_b else 'Vía urbana'} ({sc_b:.2f})</span>",
                    unsafe_allow_html=True,
                )

            is_rb_a = bool(force_a)
            is_rb_b = bool(force_b)

            with st.spinner("Analizando ambas capturas..."):
                det_a = detector.detect(img_a)
                det_a = [d for d in det_a if d.confidence >= config.CONFIDENCE_THRESHOLD]
                h_a, w_a = img_a.shape[:2]
                met_a = analyzer.analyze(det_a, h_a, w_a, is_roundabout=is_rb_a)

                det_b = detector.detect(img_b)
                det_b = [d for d in det_b if d.confidence >= config.CONFIDENCE_THRESHOLD]
                h_b, w_b = img_b.shape[:2]
                met_b = analyzer.analyze(det_b, h_b, w_b, is_roundabout=is_rb_b)

            st.markdown("---")
            st.markdown("<p class='section-title'>Detecciones</p>", unsafe_allow_html=True)
            vis_a, vis_b = st.columns(2)
            with vis_a:
                st.caption(f"**Captura A** — {file_a.name}")
                st.image(cv2.cvtColor(draw_detections(img_a, det_a), cv2.COLOR_BGR2RGB), use_container_width=True)
            with vis_b:
                st.caption(f"**Captura B** — {file_b.name}")
                st.image(cv2.cvtColor(draw_detections(img_b, det_b), cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("<p class='section-title'>Mapas de calor</p>", unsafe_allow_html=True)
            heat_a, heat_b = st.columns(2)
            with heat_a:
                st.image(cv2.cvtColor(generate_heatmap(img_a, det_a), cv2.COLOR_BGR2RGB), use_container_width=True)
            with heat_b:
                st.image(cv2.cvtColor(generate_heatmap(img_b, det_b), cv2.COLOR_BGR2RGB), use_container_width=True)

            den_a, den_b = st.columns(2)
            with den_a:
                render_density_banner(met_a.traffic_density)
            with den_b:
                render_density_banner(met_b.traffic_density)

            st.markdown("---")
            st.markdown("<p class='section-title'>Trazabilidad de la comparacion</p>", unsafe_allow_html=True)

            dataset_id_a = "roundabout" if is_rb_a else "uav_traffic"
            dataset_id_b = "roundabout" if is_rb_b else "uav_traffic"

            payload_a = build_analysis_payload(
                scene_id=file_a.name, dataset_id=dataset_id_a,
                counts=met_a.counts, total_vehicles=met_a.total_vehicles,
                density_grid=met_a.density_grid, occupancy_pct=met_a.occupancy_pct,
                zone_occupancy=met_a.zone_occupancy, risk_level=met_a.risk_level,
                model_version=detector.model_version, is_roundabout=met_a.is_roundabout,
                roundabout_occupancy_pct=met_a.roundabout_occupancy_pct,
                collision_count=met_a.collision_count,
                collisions=met_a.collisions if met_a.collisions else None,
            )
            payload_b = build_analysis_payload(
                scene_id=file_b.name, dataset_id=dataset_id_b,
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
# TAB 3: SIMULATOR (bonito + sin mostrar ID de escena)
# ═══════════════════════════════════════════════════════════════════════
with tab_simulate:
    st.markdown(
        "<div class='hero'>"
        "<h1>Simulador de escenarios</h1>"
        "<p>Simula el estado del tráfico para una fecha, hora y tipo de escena.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_dt, col_scene = st.columns([1, 1])
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

    col_w, col_e = st.columns([1, 1])
    with col_w:
        weather = st.selectbox(
            "Meteorologia",
            ["soleado", "nublado", "lluvia", "niebla"],
            format_func=lambda x: x.capitalize(),
            key="sim_weather",
        )
    with col_e:
        event_level = st.selectbox(
            "Evento especial",
            ["none", "low", "medium", "high"],
            format_func=lambda x: {"none": "Ninguno", "low": "Bajo", "medium": "Medio", "high": "Alto"}[x],
            key="sim_event",
        )

    sim_datetime = datetime.combine(sim_date, sim_time)

    # ID de escena: NO visible. Se usa internamente.
    sim_scene_id = "zona_centro_01"
    with st.expander("Opciones avanzadas", expanded=False):
        sim_scene_id = st.text_input("ID de escena (avanzado)", value=sim_scene_id, key="sim_scene_id_adv")
        st.caption("Ej: zona_centro_01, rotonda_norte")

    if st.button("Simular escenario", type="primary", key="btn_simulate"):
        result = simulator.simulate(
            sim_datetime=sim_datetime,
            scene_type=scene_type,
            scene_id=sim_scene_id,
            weather=weather,
            event_level=event_level,
            override_total=None,
            override_counts=None,
            override_density=None,
            override_occupancy=None,
        )

        state = result["traffic_state"]
        emoji = simulator.get_state_emoji(state)
        color = simulator.get_state_color(state)

        st.markdown(
            f"<div class='hero' style='border-color:{color}55;'>"
            f"<h1>{emoji} {state}</h1>"
            f"<p>{sim_datetime.strftime('%A %d/%m/%Y %H:%M')} · "
            f"{'Rotonda' if scene_type=='roundabout' else 'Vía urbana' if scene_type=='urban_road' else 'Autovía'} · "
            f"{weather.capitalize()} · "
            f"{ {'none':'Sin evento','low':'Evento bajo','medium':'Evento medio','high':'Evento alto'}[event_level] }"
            f"</p></div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Vehiculos", result["total_vehicles"])
        m2.metric("Cobertura", f"{result['occupancy_pct']:.1f}%")
        m3.metric("Riesgo", result["risk_level"])
        m4.metric("Carga", f"{result['load_ratio']:.0%}")
        m5.metric("Capacidad", result["capacity"])

        st.markdown("---")
        st.markdown("<p class='section-title'>Trazabilidad de la simulacion</p>", unsafe_allow_html=True)
        sim_hash = compute_hash(result)
        render_evidence_box(sim_hash)
        sim_evidence = build_evidence_record(result)
        render_register_button(sim_evidence, chain, "reg_sim")

        with st.expander("Ver datos de la simulacion (JSON)", expanded=False):
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

    if st.button("Verificar", type="primary", key="btn_verify"):
        if hash_input:
            with st.spinner("Buscando en el registro..."):
                record = chain.verify(hash_input)
            st.session_state["verify_result"] = record if record else "not_found"
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
                st.success(f"Verificado en blockchain ({result.get('confirmations', 0)} confirmaciones)")
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

            with st.expander("Ver registro completo (JSON)", expanded=False):
                st.json(result)

    st.markdown("---")
    st.markdown("<p class='section-title'>Recalcular huella digital desde datos</p>", unsafe_allow_html=True)
    json_input = st.text_area("JSON del analisis", key="verify_json", height=120)
    if st.button("Recalcular", key="btn_recalc"):
        if json_input:
            try:
                data = json.loads(json_input)
                recalc = compute_hash(data)
                render_evidence_box(recalc)
            except json.JSONDecodeError:
                st.error("El JSON introducido no es valido.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: RECORDS (sin expanders anidados)
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
                    hsh = rec.get("analysis_hash", "N/A")
                    st.write(f"**Hash:** `{hsh[:24]}...`" if len(hsh) > 24 else f"**Hash:** `{hsh}`")
                    txid = rec.get("tx_id", "N/A")
                    st.write(f"**TX:** `{txid[:16]}...`" if len(txid) > 16 else f"**TX:** `{txid}`")

                show_json = st.checkbox("Ver JSON completo", key=f"records_json_{i}")
                if show_json:
                    st.json(rec)
    else:
        st.info("No hay registros aun. Analiza una imagen o ejecuta una simulacion para generar el primero.")
