"""Biomechanics educational content using Driveline Open Biomechanics data.

Uses the POI (Point of Interest) metrics CSV from:
https://github.com/drivelineresearch/openbiomechanics

411 fastball pitches from ~100 pitchers (mostly college level).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

log = logging.getLogger(__name__)

_BIO_DIR = DATA_DIR / "biomechanics"
_POI_PATH = _BIO_DIR / "poi_metrics.csv"
_META_PATH = _BIO_DIR / "metadata.csv"

# In-memory caches
_poi_cache: pd.DataFrame | None = None
_merged_cache: pd.DataFrame | None = None


def load_poi() -> pd.DataFrame:
    """Load the POI metrics DataFrame."""
    global _poi_cache
    if _poi_cache is not None:
        return _poi_cache
    _poi_cache = pd.read_csv(_POI_PATH)
    log.info("Loaded %d POI rows from %s", len(_poi_cache), _POI_PATH)
    return _poi_cache


def load_merged() -> pd.DataFrame:
    """Load POI + metadata merged DataFrame."""
    global _merged_cache
    if _merged_cache is not None:
        return _merged_cache
    poi = load_poi()
    if _META_PATH.exists():
        meta = pd.read_csv(_META_PATH)
        _merged_cache = poi.merge(
            meta[["session_pitch", "session_mass_kg", "session_height_m",
                  "age_yrs", "playing_level"]],
            on="session_pitch", how="left", suffixes=("", "_meta"),
        )
    else:
        _merged_cache = poi
    return _merged_cache


# ── Topic definitions ─────────────────────────────────────────────────

# Each topic defines:
#   id: unique string for dedup
#   title: chart title
#   x_col, y_col: columns for scatter/distribution
#   x_label, y_label: axis labels
#   chart_type: "scatter", "distribution", "comparison"
#   prompt_context: context fed to Claude for the explanation
#   education: brief educational framing for the AI prompt

TOPICS = [
    {
        "id": "velo_vs_elbow_varus",
        "title": "Elbow Varus Moment vs Pitch Velocity",
        "x_col": "pitch_speed_mph",
        "y_col": "elbow_varus_moment",
        "x_label": "Pitch Speed (mph)",
        "y_label": "Elbow Varus Moment (Nm)",
        "chart_type": "scatter",
        "prompt_context": "elbow varus moment (UCL stress indicator)",
        "education": "Elbow varus moment is the torque on the inner elbow "
                     "during a pitch — it's the primary stress on the UCL "
                     "(the ligament repaired in Tommy John surgery).",
    },
    {
        "id": "hip_shoulder_separation",
        "title": "Hip-Shoulder Separation Distribution",
        "x_col": "max_rotation_hip_shoulder_separation",
        "y_col": None,
        "x_label": "Peak Hip-Shoulder Separation (deg)",
        "y_label": "Count",
        "chart_type": "distribution",
        "prompt_context": "peak hip-shoulder separation angle",
        "education": "Hip-shoulder separation is the angular difference "
                     "between pelvis and torso rotation at foot plant. "
                     "More separation = more stored elastic energy = more velo potential.",
    },
    {
        "id": "velo_vs_shoulder_er",
        "title": "Shoulder Layback vs Pitch Velocity",
        "x_col": "pitch_speed_mph",
        "y_col": "max_shoulder_external_rotation",
        "x_label": "Pitch Speed (mph)",
        "y_label": "Max Shoulder External Rotation (deg)",
        "chart_type": "scatter",
        "prompt_context": "shoulder external rotation (layback)",
        "education": "Shoulder external rotation (layback) is how far the "
                     "arm lays back before accelerating forward. More layback "
                     "means a longer acceleration path, but also more shoulder stress.",
    },
    {
        "id": "stride_length_vs_velo",
        "title": "Stride Length vs Pitch Velocity",
        "x_col": "stride_length",
        "y_col": "pitch_speed_mph",
        "x_label": "Stride Length (% Body Height)",
        "y_label": "Pitch Speed (mph)",
        "chart_type": "scatter",
        "prompt_context": "stride length as a percentage of body height",
        "education": "Stride length (measured as % of body height) affects "
                     "how much momentum transfers into the pitch. Typical "
                     "range is 75-100% of body height.",
    },
    {
        "id": "arm_slot_distribution",
        "title": "Arm Slot Distribution",
        "x_col": "arm_slot",
        "y_col": None,
        "x_label": "Arm Slot Angle (deg)",
        "y_label": "Count",
        "chart_type": "distribution",
        "prompt_context": "arm slot (forearm projection angle at release)",
        "education": "Arm slot is the angle of the forearm at ball release. "
                     "Higher angles = more over-the-top, lower = more sidearm. "
                     "Most pitchers fall between 30-55 degrees.",
    },
    {
        "id": "torso_rotation_velo",
        "title": "Torso Rotational Velocity vs Pitch Speed",
        "x_col": "max_torso_rotational_velo",
        "y_col": "pitch_speed_mph",
        "x_label": "Peak Torso Rotational Velocity (deg/s)",
        "y_label": "Pitch Speed (mph)",
        "chart_type": "scatter",
        "prompt_context": "peak torso rotational velocity",
        "education": "Torso rotation velocity is how fast the trunk rotates "
                     "towards home plate. It's a key link in the kinetic chain — "
                     "the trunk generates a massive amount of the ball's energy.",
    },
    {
        "id": "elbow_varus_distribution",
        "title": "Elbow Varus Moment Distribution",
        "x_col": "elbow_varus_moment",
        "y_col": None,
        "x_label": "Elbow Varus Moment (Nm)",
        "y_label": "Count",
        "chart_type": "distribution",
        "prompt_context": "elbow varus moment (UCL stress)",
        "education": "The UCL can typically handle ~34 Nm before failure. "
                     "During a pitch, dynamic forces from muscles and other "
                     "tissues share the load — but higher varus moments still "
                     "mean more cumulative UCL stress.",
    },
    {
        "id": "pelvis_vs_torso_timing",
        "title": "Pelvis-Torso Rotational Timing",
        "x_col": "timing_peak_torso_to_peak_pelvis_rot_velo",
        "y_col": "pitch_speed_mph",
        "x_label": "Time: Peak Pelvis → Peak Torso Rotation (sec)",
        "y_label": "Pitch Speed (mph)",
        "chart_type": "scatter",
        "prompt_context": "timing between peak pelvis and peak torso "
                          "rotational velocity (kinetic chain sequencing)",
        "education": "Efficient pitchers have a separation in timing — "
                     "the pelvis rotates first, then the torso follows. "
                     "This sequential 'whip' effect is the kinetic chain in action.",
    },
    {
        "id": "shoulder_energy_transfer",
        "title": "Shoulder Energy Transfer vs Pitch Speed",
        "x_col": "shoulder_transfer_fp_br",
        "y_col": "pitch_speed_mph",
        "x_label": "Shoulder Energy Transfer (J)",
        "y_label": "Pitch Speed (mph)",
        "chart_type": "scatter",
        "prompt_context": "energy transfer across the throwing shoulder "
                          "between foot plant and ball release",
        "education": "Energy transfer at the shoulder measures how much "
                     "energy flows from the trunk through the shoulder to "
                     "the arm. Higher transfer = more efficient kinetic chain.",
    },
    {
        "id": "lead_leg_block",
        "title": "Lead Knee Extension vs Pitch Speed",
        "x_col": "lead_knee_extension_from_fp_to_br",
        "y_col": "pitch_speed_mph",
        "x_label": "Lead Knee Extension: Foot Plant → Ball Release (deg)",
        "y_label": "Pitch Speed (mph)",
        "chart_type": "scatter",
        "prompt_context": "lead knee extension from foot plant to ball release "
                          "(the 'lead leg block')",
        "education": "The lead leg 'block' is when a pitcher braces and "
                     "extends their front leg after landing. This creates a "
                     "firm post that helps transfer energy up the chain — "
                     "like a pole vaulter planting the pole.",
    },
    {
        "id": "velo_vs_shoulder_ir_moment",
        "title": "Shoulder Internal Rotation Moment vs Velocity",
        "x_col": "pitch_speed_mph",
        "y_col": "shoulder_internal_rotation_moment",
        "x_label": "Pitch Speed (mph)",
        "y_label": "Shoulder IR Moment (Nm)",
        "chart_type": "scatter",
        "prompt_context": "shoulder internal rotation moment",
        "education": "Shoulder internal rotation moment is the rotational "
                     "torque at the shoulder as the arm accelerates forward. "
                     "It's a key indicator of shoulder stress and rotator "
                     "cuff loading.",
    },
    {
        "id": "ground_reaction_forces",
        "title": "Rear vs Lead Leg Peak Ground Reaction Force",
        "x_col": "rear_grf_mag_max",
        "y_col": "lead_grf_mag_max",
        "x_label": "Rear Leg Peak GRF (N)",
        "y_label": "Lead Leg Peak GRF (N)",
        "chart_type": "scatter",
        "prompt_context": "rear leg vs lead leg peak ground reaction force magnitude",
        "education": "Ground reaction forces are how hard the pitcher pushes "
                     "into the ground. The rear leg drives forward (push-off), "
                     "the lead leg brakes (landing). Both are critical for "
                     "velocity — pitching starts from the ground up.",
    },
    {
        "id": "hip_shoulder_sep_vs_velo",
        "title": "Hip-Shoulder Separation vs Pitch Speed",
        "x_col": "max_rotation_hip_shoulder_separation",
        "y_col": "pitch_speed_mph",
        "x_label": "Peak Hip-Shoulder Separation (deg)",
        "y_label": "Pitch Speed (mph)",
        "chart_type": "scatter",
        "prompt_context": "hip-shoulder separation vs pitch velocity",
        "education": "Hip-shoulder separation is one of the most talked-about "
                     "metrics in pitching development. The idea: rotate the "
                     "hips early while the torso stays closed, creating elastic "
                     "energy that catapults the arm forward.",
    },
    {
        "id": "cog_velo_vs_pitch_speed",
        "title": "Center of Gravity Velocity vs Pitch Speed",
        "x_col": "max_cog_velo_x",
        "y_col": "pitch_speed_mph",
        "x_label": "Peak CoG Velocity Towards Home (m/s)",
        "y_label": "Pitch Speed (mph)",
        "chart_type": "scatter",
        "prompt_context": "peak center of gravity velocity towards home plate",
        "education": "Center of gravity velocity measures how fast the "
                     "pitcher's body moves towards home plate during the "
                     "delivery. More linear momentum = more energy available "
                     "to transfer into the ball.",
    },
]


def pick_topic(recent_ids: list[str] | None = None) -> dict:
    """Pick a random topic, avoiding recently used ones."""
    available = TOPICS.copy()
    if recent_ids:
        available = [t for t in available if t["id"] not in recent_ids]
    if not available:
        available = TOPICS.copy()
    return random.choice(available)


def compute_topic_stats(topic: dict, df: pd.DataFrame) -> dict:
    """Compute summary statistics for a topic's columns."""
    stats = {}
    x_col = topic["x_col"]
    y_col = topic.get("y_col")

    x_vals = df[x_col].dropna()
    stats["x_n"] = len(x_vals)
    stats["x_mean"] = float(x_vals.mean())
    stats["x_std"] = float(x_vals.std())
    stats["x_median"] = float(x_vals.median())
    stats["x_min"] = float(x_vals.min())
    stats["x_max"] = float(x_vals.max())
    stats["x_p10"] = float(x_vals.quantile(0.10))
    stats["x_p90"] = float(x_vals.quantile(0.90))

    if y_col and y_col in df.columns:
        y_vals = df[y_col].dropna()
        stats["y_n"] = len(y_vals)
        stats["y_mean"] = float(y_vals.mean())
        stats["y_std"] = float(y_vals.std())
        stats["y_median"] = float(y_vals.median())

        # Correlation
        valid = df[[x_col, y_col]].dropna()
        if len(valid) > 10:
            stats["correlation"] = float(valid[x_col].corr(valid[y_col]))

    stats["n_pitchers"] = int(df["session"].nunique())
    stats["n_pitches"] = len(df)

    return stats
