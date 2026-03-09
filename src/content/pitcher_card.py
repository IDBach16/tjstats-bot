"""Content generator: premium pitcher card image (PLV-style)."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ._helpers import fmt_stat, safe_stat, build_stat_block, get_name
from .. import pitch_profiler
from .._player_pick import pick_player
from ..analysis import generate_analysis
from ..charts import plot_pitcher_card
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


class PitcherCardGenerator(ContentGenerator):
    name = "pitcher_card"

    async def generate(self) -> PostContent:
        player_info = pick_player()
        name = player_info["name"]
        team = player_info.get("team")
        player_id = player_info.get("id")

        season_df = pitch_profiler.get_season_pitchers()
        if season_df.empty:
            log.warning("No season pitcher data available")
            return PostContent(text="")

        pitches_df = pitch_profiler.get_season_pitches()

        # Render the card
        image_path = plot_pitcher_card(name, season_df, pitches_df, team=team,
                                       player_id=player_id)
        if not image_path:
            log.warning("Pitcher card rendering failed for %s", name)
            return PostContent(text="")

        # Build tweet text with a compact stat summary
        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in season_df.columns:
                name_col = c
                break

        summary_parts: list[str] = []
        season_stats: dict[str, str] = {}
        if name_col:
            matches = season_df[season_df[name_col] == name]
            if not matches.empty:
                p = matches.iloc[0]
                for col, label, fmt, pct in [
                    ("era", "ERA", ".2f", False),
                    ("fip", "FIP", ".2f", False),
                    ("strike_out_percentage", "K%", None, True),
                    ("walk_percentage", "BB%", None, True),
                    ("whiff_rate", "Whiff%", None, True),
                    ("stuff_plus", "Stuff+", ".0f", False),
                ]:
                    if col in p.index:
                        try:
                            val = float(p[col])
                            if pct:
                                display = f"{val * 100:.1f}%"
                            elif fmt:
                                display = format(val, fmt)
                            else:
                                display = str(val)
                            season_stats[label] = display
                        except (TypeError, ValueError):
                            pass

                era = safe_stat(p, "era")
                k_pct = safe_stat(p, "strike_out_percentage", pct=True)
                whiff = safe_stat(p, "whiff_rate", pct=True)
                stuff = safe_stat(p, "stuff_plus")
                if era is not None:
                    summary_parts.append(f"ERA: {era:.2f}")
                if k_pct is not None:
                    summary_parts.append(f"K%: {k_pct:.1f}")
                if whiff is not None:
                    summary_parts.append(f"Whiff%: {whiff:.1f}")
                if stuff is not None:
                    summary_parts.append(f"Stuff+: {stuff:.0f}")

        # ── Build per-pitch stats for AI analysis ──
        pitch_stats: list[dict] = []
        pitch_name_col = None
        for c in ("pitch_name", "pitch_type"):
            if c in pitches_df.columns:
                pitch_name_col = c
                break

        if pitch_name_col and name_col and name_col in pitches_df.columns:
            pitcher_pitches = pitches_df[pitches_df[name_col] == name]
            if not pitcher_pitches.empty:
                grouped = pitcher_pitches.groupby(pitch_name_col)
                total_thrown = pitcher_pitches["percentage_thrown"].sum() if "percentage_thrown" in pitcher_pitches.columns else 1
                for ptype, grp in grouped:
                    row = grp.iloc[0]
                    ps: dict = {"pitch_name": str(ptype)}
                    if "percentage_thrown" in grp.columns:
                        usage = grp["percentage_thrown"].sum() / total_thrown * 100
                        ps["usage"] = round(usage, 1)
                    for attr, key in [
                        ("velocity", "velocity"),
                        ("whiff_rate", "whiff_rate"),
                        ("stuff_plus", "stuff_plus"),
                        ("run_value_per_100_pitches", "run_value"),
                    ]:
                        if attr in row.index:
                            try:
                                v = float(row[attr])
                                if key == "whiff_rate" and v < 1:
                                    v *= 100
                                ps[key] = v
                            except (TypeError, ValueError):
                                pass
                    pitch_stats.append(ps)

        # ── Generate AI analysis ──
        analysis_text = None
        if season_stats:
            analysis_text = generate_analysis(name, season_stats, pitch_stats)

        # ── Fetch video clip ──
        video_path = None
        if player_id:
            try:
                video_path = get_pitcher_clip(player_id, name)
                if video_path:
                    log.info("Got video clip for %s: %s", name, video_path)
            except Exception:
                log.warning("Video clip fetch failed for %s", name, exc_info=True)

        stat_line = " | ".join(summary_parts) if summary_parts else ""
        stat_section = f"\n\n{stat_line}" if stat_line else ""

        text = (
            f"{name}'s {MLB_SEASON} Pitcher Card"
            f"{stat_section}"
            f"\n\n@TJStats {DEFAULT_HASHTAGS}"
        )

        # Build reply with analysis
        reply_content = None
        if analysis_text:
            reply_content = PostContent(
                text=analysis_text,
                tags=["analysis"],
            )

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Pitcher card for {name} showing season stats and arsenal breakdown",
            tags=["pitcher_card", name],
            reply=reply_content,
        )
