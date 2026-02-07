"""What-if traffic simulator.

Generates estimated traffic metrics based on:
- Time-of-day / day-of-week patterns
- Scene type (road segment vs roundabout)
- User-adjustable parameters (vehicle counts, density overrides)
- Historical baseline from previous real analyses (if available)
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

# ── Time-of-day traffic multipliers (0-23h) ──────────────────────────
# Based on typical urban traffic patterns
_HOURLY_FACTOR = {
    0: 0.10, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.08, 5: 0.15,
    6: 0.40, 7: 0.75, 8: 0.95, 9: 0.80, 10: 0.60, 11: 0.65,
    12: 0.70, 13: 0.75, 14: 0.70, 15: 0.65, 16: 0.75, 17: 0.90,
    18: 1.00, 19: 0.85, 20: 0.60, 21: 0.40, 22: 0.25, 23: 0.15,
}

# Weekend reduction factor
_WEEKEND_FACTOR = 0.6

# Vehicle type distribution templates (proportions that sum to ~1.0)
_VEHICLE_MIX = {
    "urban_road": {"car": 0.70, "motorcycle": 0.15, "bus": 0.08, "truck": 0.05, "cycle": 0.02},
    "roundabout": {"car": 0.75, "motorcycle": 0.10, "bus": 0.05, "truck": 0.08, "cycle": 0.02},
    "highway":    {"car": 0.60, "motorcycle": 0.10, "bus": 0.05, "truck": 0.23, "cycle": 0.02},
}

# Base capacity (max vehicles per scene type at peak)
_BASE_CAPACITY = {
    "urban_road": 45,
    "roundabout": 30,
    "highway": 60,
}

# Traffic state thresholds (% of capacity)
_STATE_THRESHOLDS = {
    "FLUIDO":         (0.0,  0.40),
    "MODERADO":       (0.40, 0.65),
    "CONGESTIONADO":  (0.65, 0.85),
    "RIESGO ALTO":    (0.85, 1.50),
}


class TrafficSimulator:
    """What-if traffic scenario simulator."""

    def __init__(self, historical_path: Path | None = None):
        self._historical = self._load_historical(historical_path)

    @staticmethod
    def _load_historical(path: Path | None) -> list[dict]:
        """Load previous analysis results to use as baseline."""
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def simulate(
        self,
        sim_datetime: datetime,
        scene_type: str = "urban_road",
        scene_id: str = "simulated_scene",
        # User overrides (None = auto-calculate)
        override_total: int | None = None,
        override_counts: dict[str, int] | None = None,
        override_density: float | None = None,
        override_occupancy: float | None = None,
    ) -> dict[str, Any]:
        """Run a what-if simulation.

        Returns a full analysis-like payload with estimated metrics.
        """
        hour = sim_datetime.hour
        weekday = sim_datetime.weekday()  # 0=Monday, 6=Sunday
        is_weekend = weekday >= 5

        # ── Calculate traffic intensity factor ─────────────────────
        time_factor = _HOURLY_FACTOR.get(hour, 0.5)
        if is_weekend:
            time_factor *= _WEEKEND_FACTOR

        capacity = _BASE_CAPACITY.get(scene_type, 40)

        # ── Total vehicles ─────────────────────────────────────────
        if override_total is not None:
            total_vehicles = override_total
        else:
            # Add slight randomness for realism
            noise = random.uniform(0.85, 1.15)
            total_vehicles = max(0, int(capacity * time_factor * noise))

        # ── Vehicle distribution ───────────────────────────────────
        if override_counts is not None:
            counts = override_counts
            total_vehicles = sum(counts.values())
        else:
            mix = _VEHICLE_MIX.get(scene_type, _VEHICLE_MIX["urban_road"])
            counts = {}
            remaining = total_vehicles
            sorted_classes = sorted(mix.items(), key=lambda x: x[1], reverse=True)
            for i, (cls, prop) in enumerate(sorted_classes):
                if i == len(sorted_classes) - 1:
                    counts[cls] = remaining
                else:
                    n = int(total_vehicles * prop)
                    counts[cls] = n
                    remaining -= n
            # Remove zeros
            counts = {k: v for k, v in counts.items() if v > 0}

        # ── Load ratio (how full is the road) ──────────────────────
        load_ratio = total_vehicles / max(capacity, 1)

        # ── Density ────────────────────────────────────────────────
        if override_density is not None:
            avg_density = override_density
        else:
            avg_density = round(load_ratio * config.GRID_SIZE, 2)

        # ── Build density grid (simulated) ─────────────────────────
        grid_size = config.GRID_SIZE
        density_grid = self._simulate_density_grid(
            total_vehicles, grid_size, scene_type
        )

        # ── Occupancy ──────────────────────────────────────────────
        if override_occupancy is not None:
            occupancy_pct = override_occupancy
        else:
            # Rough estimate: each vehicle ~0.3-0.8% of image area
            avg_vehicle_area_pct = 0.5
            occupancy_pct = round(min(total_vehicles * avg_vehicle_area_pct, 95.0), 2)

        # ── Traffic state ──────────────────────────────────────────
        traffic_state = self._classify_state(load_ratio)

        # ── Risk level ─────────────────────────────────────────────
        has_heavy = any(counts.get(c, 0) > 0 for c in ("bus", "truck"))
        max_cell = max(max(row) for row in density_grid) if density_grid else 0
        if max_cell >= config.RISK_DENSITY_THRESHOLD and has_heavy:
            risk_level = "HIGH"
        elif max_cell >= config.RISK_DENSITY_THRESHOLD or has_heavy:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # ── Zone occupancy ─────────────────────────────────────────
        zone_occupancy = self._simulate_zone_occupancy(scene_type, time_factor)

        # ── Roundabout ─────────────────────────────────────────────
        is_roundabout = scene_type == "roundabout"
        roundabout_occ = round(load_ratio * 80, 2) if is_roundabout else None

        # ── Build result ───────────────────────────────────────────
        return {
            "type": "simulation",
            "scene_id": scene_id,
            "scene_type": scene_type,
            "simulated_datetime": sim_datetime.isoformat(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "is_weekend": is_weekend,
            "time_factor": round(time_factor, 3),
            "counts": dict(sorted(counts.items())),
            "total_vehicles": total_vehicles,
            "density_grid": density_grid,
            "avg_density": avg_density,
            "occupancy_pct": occupancy_pct,
            "zone_occupancy": {k: round(v, 2) for k, v in zone_occupancy.items()},
            "traffic_state": traffic_state,
            "risk_level": risk_level,
            "load_ratio": round(load_ratio, 3),
            "capacity": capacity,
            "is_roundabout": is_roundabout,
            "roundabout_occupancy_pct": roundabout_occ,
            "model_version": "simulator_v1.0",
            "dataset_id": "simulation",
        }

    @staticmethod
    def _classify_state(load_ratio: float) -> str:
        for state, (lo, hi) in _STATE_THRESHOLDS.items():
            if lo <= load_ratio < hi:
                return state
        return "RIESGO ALTO" if load_ratio >= 0.85 else "FLUIDO"

    @staticmethod
    def _simulate_density_grid(total: int, grid_size: int, scene_type: str) -> list[list[int]]:
        """Distribute vehicles across a grid with realistic clustering."""
        grid = [[0] * grid_size for _ in range(grid_size)]
        if total == 0:
            return grid

        # Center-biased for roundabouts, spread for roads
        for _ in range(total):
            if scene_type == "roundabout":
                # Bias toward center
                gx = int(random.gauss(grid_size / 2, grid_size / 4))
                gy = int(random.gauss(grid_size / 2, grid_size / 4))
            else:
                # Bias toward horizontal middle (road lanes)
                gx = random.randint(0, grid_size - 1)
                gy = int(random.gauss(grid_size / 2, grid_size / 3))

            gx = max(0, min(grid_size - 1, gx))
            gy = max(0, min(grid_size - 1, gy))
            grid[gy][gx] += 1

        return grid

    @staticmethod
    def _simulate_zone_occupancy(scene_type: str, time_factor: float) -> dict[str, float]:
        """Simulated zone distribution."""
        if scene_type == "roundabout":
            upper = 25 + random.uniform(-5, 5)
            middle = 50 + random.uniform(-10, 10)
            lower = 25 + random.uniform(-5, 5)
        else:
            # Rush hour: more in middle zone (main road)
            upper = 20 + random.uniform(-5, 5)
            middle = 45 + time_factor * 15 + random.uniform(-5, 5)
            lower = 35 - time_factor * 10 + random.uniform(-5, 5)

        total = upper + middle + lower
        return {
            "upper": upper / total * 100,
            "middle": middle / total * 100,
            "lower": lower / total * 100,
        }

    @staticmethod
    def get_state_color(state: str) -> str:
        """Return a color hex for the traffic state."""
        return {
            "FLUIDO": "#2ecc71",
            "MODERADO": "#f39c12",
            "CONGESTIONADO": "#e74c3c",
            "RIESGO ALTO": "#8e44ad",
        }.get(state, "#95a5a6")

    @staticmethod
    def get_state_emoji(state: str) -> str:
        return {
            "FLUIDO": "🟢",
            "MODERADO": "🟡",
            "CONGESTIONADO": "🔴",
            "RIESGO ALTO": "🟣",
        }.get(state, "⚪")
