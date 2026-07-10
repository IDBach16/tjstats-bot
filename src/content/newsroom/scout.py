"""Advance scout: widen the newsroom board with extra Baseball Savant leaderboards.

feeds.py covers the core four kinds; this scout digs deeper into Savant so the
assignment editor always has a fresh, varied pool to pick from (and daily posting
never runs dry on repeats). Every lead is real and carries its exact numbers —
same contract as feeds.Lead, so it flows through researcher/writer/copydesk/social
with no special handling beyond a fact-sheet block in researcher.py.

Added story kinds:
  hard_hitter   — hitter doing the most damage on contact (barrel rate / exit velo)
  flamethrower  — pitcher with the hardest average four-seam fastball
  pitcher_luck  — pitcher whose ERA and xERA disagree the most (a luck story in
                  either direction: ERA-flattered & due to regress, or ERA-cursed
                  & due for a breakout)
"""

from __future__ import annotations

import logging

from .feeds import (Lead, _csv, _flip_name, _ordered_candidates, _r,
                    PER_KIND, _SAVANT)
from ...config import MLB_SEASON

log = logging.getLogger(__name__)

MIN_BATTED_BALLS = 150    # statcast custom "attempts" (batted-ball events)
MIN_BATTERS_FACED = 300   # pitcher expected-stats sample size


def _int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


# ── lead builders ──────────────────────────────────────────────────────
def _hard_hitter_leads(reds: set[int]) -> list[Lead]:
    """Biggest thumpers from the custom Statcast batter board (by barrel rate)."""
    df = _csv(f"{_SAVANT}/statcast?type=batter&year={MLB_SEASON}"
              f"&min={MIN_BATTED_BALLS}&csv=true")
    if df.empty:
        return []
    df = df.dropna(subset=["brl_percent", "avg_hit_speed"])
    if df.empty:
        return []
    league_ev = _r(df["avg_hit_speed"].mean(), 1)
    df = _ordered_candidates(df.sort_values("brl_percent", ascending=False),
                             "player_id", reds)
    leads = []
    for i, (_, row) in enumerate(df.head(PER_KIND).iterrows()):
        pid = int(row["player_id"])
        leads.append(Lead(
            kind="hard_hitter", subject=_flip_name(row["last_name, first_name"]),
            player_id=pid, is_pitcher=False,
            angle="Does more damage on contact than almost anyone in baseball.",
            rank=i + 1, total=len(df), is_red=pid in reds,
            facts={
                "avg_hit_speed": _r(row["avg_hit_speed"], 1),
                "max_hit_speed": _r(row.get("max_hit_speed"), 1),
                "brl_percent": _r(row["brl_percent"], 1),
                "barrels": _int(row.get("barrels")),
                "ev95percent": _r(row.get("ev95percent"), 1),
                "max_distance": _int(row.get("max_distance")),
                "attempts": _int(row.get("attempts")),
                "league_ev": league_ev,
            },
        ))
    return leads


def _flamethrower_leads(reds: set[int]) -> list[Lead]:
    """Hardest average four-seam fastballs from the pitch-arsenals board."""
    df = _csv(f"{_SAVANT}/pitch-arsenals?type=avg_speed&year={MLB_SEASON}"
              f"&min=100&csv=true")
    if df.empty:
        return []
    df = df.dropna(subset=["ff_avg_speed"])
    if df.empty:
        return []
    league_ff = _r(df["ff_avg_speed"].mean(), 1)
    df = _ordered_candidates(df.sort_values("ff_avg_speed", ascending=False),
                             "pitcher", reds)
    leads = []
    for i, (_, row) in enumerate(df.head(PER_KIND).iterrows()):
        pid = int(row["pitcher"])
        leads.append(Lead(
            kind="flamethrower", subject=_flip_name(row["last_name, first_name"]),
            player_id=pid, is_pitcher=True,
            angle="Throws one of the hardest fastballs on the planet.",
            rank=i + 1, total=len(df), is_red=pid in reds,
            facts={
                "ff_avg_speed": _r(row["ff_avg_speed"], 1),
                "si_avg_speed": _r(row.get("si_avg_speed"), 1),
                "league_ff": league_ff,
            },
        ))
    return leads


def _pitcher_luck_leads(reds: set[int]) -> list[Lead]:
    """Pitchers whose ERA and xERA disagree most (luck story, either direction)."""
    df = _csv(f"{_SAVANT}/expected_statistics?type=pitcher&year={MLB_SEASON}"
              f"&min=150&csv=true")
    if df.empty:
        return []
    df = df.dropna(subset=["era", "xera", "pa"])
    df = df[df["pa"] >= MIN_BATTERS_FACED].copy()
    if df.empty:
        return []
    # gap < 0  -> ERA below xERA -> results better than earned (lucky)
    # gap > 0  -> ERA above xERA -> results worse than earned (unlucky)
    df["gap"] = df["era"] - df["xera"]
    df["absgap"] = df["gap"].abs()
    df = _ordered_candidates(df.sort_values("absgap", ascending=False),
                             "player_id", reds)
    leads = []
    for i, (_, row) in enumerate(df.head(PER_KIND).iterrows()):
        pid = int(row["player_id"])
        lucky = bool(row["gap"] < 0)
        angle = ("His ERA looks great, but xERA says he's been lucky — regression "
                 "is coming." if lucky else
                 "His ERA looks ugly, but xERA says he's pitched far better — a "
                 "turnaround is overdue.")
        leads.append(Lead(
            kind="pitcher_luck", subject=_flip_name(row["last_name, first_name"]),
            player_id=pid, is_pitcher=True, angle=angle,
            rank=i + 1, total=len(df), is_red=pid in reds,
            facts={
                "era": _r(row["era"], 2), "xera": _r(row["xera"], 2),
                "gap": _r(row["gap"], 2),
                "woba": _r(row["woba"], 3), "est_woba": _r(row["est_woba"], 3),
                "ba": _r(row["ba"], 3), "est_ba": _r(row["est_ba"], 3),
                "pa": int(row["pa"]), "lucky": lucky,
            },
        ))
    return leads


def build_scout_leads(reds: set[int]) -> dict[str, list[Lead]]:
    """All scout kinds -> {kind: [Lead, ...]}. Each feed fails independently."""
    out: dict[str, list[Lead]] = {}
    for key, fn in (("hard_hitter", _hard_hitter_leads),
                    ("flamethrower", _flamethrower_leads),
                    ("pitcher_luck", _pitcher_luck_leads)):
        try:
            out[key] = fn(reds)
        except Exception:
            log.warning("scout %s feed failed", key, exc_info=True)
            out[key] = []
    return out
