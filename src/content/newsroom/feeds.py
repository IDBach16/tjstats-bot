"""Wire service: pull candidate story leads from Baseball Savant leaderboards.

Each lead is a verified, real story angle with the exact numbers attached (so the
columnist never has to invent anything). Coverage is all-MLB and purely merit-based
— the best story on the board wins, regardless of team.

Story kinds:
  overperformer  — hitter whose results are way ahead of their batted-ball data
  underperformer — hitter crushing the ball with nothing to show for it (unlucky)
  nasty_pitch    — pitcher throwing the filthiest single pitch in the game
  bat_speed      — hitter with the fastest bat in baseball
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import pandas as pd
import requests

from ...config import MLB_SEASON, MLB_API_BASE

log = logging.getLogger(__name__)

REDS_TEAM_ID = 113
PER_KIND = 6            # how many candidates to keep per kind (for video fallback)
TOPN_REDS_WINDOW = 20   # a Red inside the top-N of a board gets the nod
MIN_PA = 200            # Savant's URL min= does NOT filter PA — we filter in code
MIN_BAT_SWINGS = 100    # min competitive swings for a bat-speed story

_SAVANT = "https://baseballsavant.mlb.com/leaderboard"


@dataclass
class Lead:
    kind: str
    subject: str          # "First Last"
    player_id: int
    is_pitcher: bool
    angle: str            # short editorial angle for the assignment
    facts: dict = field(default_factory=dict)
    rank: int = 0
    total: int = 0
    is_red: bool = False


# ── helpers ────────────────────────────────────────────────────────────
def _csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _flip_name(name: str) -> str:
    """Savant 'Last, First' -> 'First Last'."""
    name = str(name)
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
        return f"{first} {last}"
    return name


def reds_player_ids() -> set[int]:
    """MLBAM ids of every Red this season (matches Savant player_id)."""
    try:
        url = (f"{MLB_API_BASE}/teams/{REDS_TEAM_ID}/roster"
               f"?season={MLB_SEASON}&rosterType=fullSeason")
        data = requests.get(url, timeout=15).json()
        return {p["person"]["id"] for p in data.get("roster", [])}
    except Exception:
        log.warning("Reds roster fetch failed", exc_info=True)
        return set()


def _ordered_candidates(df: pd.DataFrame, id_col: str, reds: set[int]) -> pd.DataFrame:
    """Reds-lean ordering: if a Red is in the top window, float them to the front."""
    df = df.reset_index(drop=True)
    if reds:
        window = df.head(TOPN_REDS_WINDOW)
        red_hits = window[window[id_col].isin(reds)]
        if not red_hits.empty:
            rest = df.drop(red_hits.index)
            df = pd.concat([red_hits, rest]).reset_index(drop=True)
    return df


def _r(x, n=3):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


# ── lead builders ──────────────────────────────────────────────────────
def _hitter_expected_leads(reds: set[int]) -> tuple[list[Lead], list[Lead]]:
    """Overperformers and underperformers from batter expected_statistics."""
    df = _csv(f"{_SAVANT}/expected_statistics?type=batter&year={MLB_SEASON}"
              f"&position=&team=&min=150&csv=true")
    if df.empty:
        return [], []
    df = df.dropna(subset=["woba", "est_woba", "pa"])
    df = df[df["pa"] >= MIN_PA].copy()
    if df.empty:
        return [], []
    # Compute the gap ourselves so the sign is unambiguous:
    #   gap < 0  -> woba beats xwOBA  -> OVERperforming (lucky)
    #   gap > 0  -> xwOBA beats woba  -> UNDERperforming (unlucky)
    df["gap"] = df["est_woba"] - df["woba"]
    league_woba = _r(df["woba"].mean())

    def build(sub_df: pd.DataFrame, kind: str, angle: str) -> list[Lead]:
        sub_df = _ordered_candidates(sub_df, "player_id", reds)
        leads = []
        for i, (_, row) in enumerate(sub_df.head(PER_KIND).iterrows()):
            pid = int(row["player_id"])
            leads.append(Lead(
                kind=kind, subject=_flip_name(row["last_name, first_name"]),
                player_id=pid, is_pitcher=False, angle=angle,
                rank=i + 1, total=len(sub_df), is_red=pid in reds,
                facts={
                    "woba": _r(row["woba"]), "est_woba": _r(row["est_woba"]),
                    "gap": _r(row["gap"]), "pa": int(row["pa"]),
                    "ba": _r(row["ba"], 3), "est_ba": _r(row["est_ba"], 3),
                    "slg": _r(row["slg"], 3), "est_slg": _r(row["est_slg"], 3),
                    "league_woba": league_woba,
                },
            ))
        return leads

    over = df.sort_values("gap", ascending=True)    # most negative gap = biggest overperformer
    under = df.sort_values("gap", ascending=False)  # most positive gap = biggest underperformer
    return (
        build(over, "overperformer",
              "Results are miles ahead of the batted-ball data — regression is coming."),
        build(under, "underperformer",
              "Smoking the ball with nothing to show for it — a breakout waiting to happen."),
    )


def _nasty_pitch_leads(reds: set[int]) -> list[Lead]:
    """Filthiest single pitches from pitch-arsenal-stats (by whiff%)."""
    df = _csv(f"{_SAVANT}/pitch-arsenal-stats?type=pitcher&pitchType=&year={MLB_SEASON}"
              f"&position=&team=&min=100&csv=true")
    if df.empty:
        return []
    df = df.dropna(subset=["whiff_percent", "pitches"])
    df = df[df["pitches"] >= 150]
    league_whiff = _r(df["whiff_percent"].mean(), 1)
    df = _ordered_candidates(df.sort_values("whiff_percent", ascending=False),
                             "player_id", reds)
    leads = []
    for i, (_, row) in enumerate(df.head(PER_KIND).iterrows()):
        pid = int(row["player_id"])
        leads.append(Lead(
            kind="nasty_pitch", subject=_flip_name(row["last_name, first_name"]),
            player_id=pid, is_pitcher=True,
            angle=f"Throws the nastiest {row['pitch_name']} in baseball.",
            rank=i + 1, total=len(df), is_red=pid in reds,
            facts={
                "pitch_name": str(row["pitch_name"]),
                "whiff_percent": _r(row["whiff_percent"], 1),
                "k_percent": _r(row["k_percent"], 1),
                "put_away": _r(row["put_away"], 1),
                "run_value_per_100": _r(row["run_value_per_100"], 1),
                "pitches": int(row["pitches"]),
                "pitch_usage": _r(row["pitch_usage"], 1),
                "opp_ba": _r(row["ba"], 3),
                "team": str(row.get("team_name_alt", "")),
                "league_whiff": league_whiff,
            },
        ))
    return leads


def _bat_speed_leads(reds: set[int]) -> list[Lead]:
    """Fastest bats from bat-tracking (by avg_bat_speed)."""
    df = _csv(f"{_SAVANT}/bat-tracking?attackZone=&batSide=&year={MLB_SEASON}"
              f"&min=100&csv=true")
    if df.empty:
        return []
    df = df.dropna(subset=["avg_bat_speed"])
    if "swings_competitive" in df.columns:
        df = df[df["swings_competitive"] >= MIN_BAT_SWINGS]
    if df.empty:
        return []
    league_bs = _r(df["avg_bat_speed"].mean(), 1)
    df = _ordered_candidates(df.sort_values("avg_bat_speed", ascending=False),
                             "id", reds)
    leads = []
    for i, (_, row) in enumerate(df.head(PER_KIND).iterrows()):
        pid = int(row["id"])
        leads.append(Lead(
            kind="bat_speed", subject=_flip_name(row["name"]),
            player_id=pid, is_pitcher=False,
            angle="Swings the fastest bat in the sport.",
            rank=i + 1, total=len(df), is_red=pid in reds,
            facts={
                "avg_bat_speed": _r(row["avg_bat_speed"], 1),
                "blast_per_swing": _r(row.get("blast_per_swing"), 1),
                "squared_up_per_swing": _r(row.get("squared_up_per_swing"), 1),
                "hard_swing_rate": _r(row.get("hard_swing_rate"), 1),
                "swing_length": _r(row.get("swing_length"), 1),
                "league_bat_speed": league_bs,
            },
        ))
    return leads


def build_leads() -> dict[str, list[Lead]]:
    """Return up to PER_KIND ranked candidate leads per story kind."""
    # All-MLB, no team lean. Swap in reds_player_ids() here to re-enable a Reds tilt.
    reds: set[int] = set()
    out: dict[str, list[Lead]] = {
        "overperformer": [], "underperformer": [],
        "nasty_pitch": [], "bat_speed": [], "article": [],
    }
    try:
        over, under = _hitter_expected_leads(reds)
        out["overperformer"], out["underperformer"] = over, under
    except Exception:
        log.warning("expected_statistics feed failed", exc_info=True)
    try:
        out["nasty_pitch"] = _nasty_pitch_leads(reds)
    except Exception:
        log.warning("arsenal feed failed", exc_info=True)
    try:
        out["bat_speed"] = _bat_speed_leads(reds)
    except Exception:
        log.warning("bat-tracking feed failed", exc_info=True)
    try:
        # local import: articles imports Lead from this module (avoid a cycle)
        from . import articles
        out["article"] = articles.build_article_leads()
    except Exception:
        log.warning("article feed failed", exc_info=True)
    n = sum(len(v) for v in out.values())
    log.info("newsroom feeds: %d candidate leads across %d kinds", n,
             sum(1 for v in out.values() if v))
    return out
