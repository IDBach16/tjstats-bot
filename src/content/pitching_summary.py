"""Content generator: TJStats-style pitching summary dashboard."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..charts import plot_pitching_summary
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


# A pitcher needs enough work for the card's rate stats to mean anything. 250 is
# the same floor the scout board uses for stuff_plus.
MIN_PITCHES = 250


def _candidate_pitchers(season_df) -> list[dict]:
    """Qualified pitchers this generator has NOT featured, most work first.

    Ordered by pitches thrown so the biggest sample goes first -- a card on a
    pitcher with 260 pitches is thinner than one on a pitcher with 2,600, and if
    the first choice fails to render we would rather fall to the next-biggest
    than to a random arm.
    """
    from ..scheduler import recent_generator_tags

    if season_df is None or season_df.empty:
        return []
    cols = set(season_df.columns)
    name_col = "pitcher_name" if "pitcher_name" in cols else (
        "player_name" if "player_name" in cols else None)
    if not name_col or "pitcher_id" not in cols:
        log.warning("Season board is missing name/id columns; falling back to the watchlist")
        return []

    df = season_df
    if "pitches_thrown" in cols:
        df = df[df["pitches_thrown"] >= MIN_PITCHES]
        df = df.sort_values("pitches_thrown", ascending=False)

    posted = recent_generator_tags("pitching_summary", index=1, lookback=200)
    out: list[dict] = []
    for _, r in df.iterrows():
        nm = str(r[name_col]).strip()
        if not nm or nm in posted:
            continue
        out.append({"name": nm, "id": int(r["pitcher_id"]),
                    "team": (str(r["team"]).strip() if "team" in cols and r.get("team") else None)})
    if not out:
        # Everyone qualified has been featured. Say so rather than silently
        # repeating -- it means the floor or the lookback wants revisiting.
        log.info("Every qualified pitcher has been featured; falling back to the watchlist")
    return out


class PitchingSummaryGenerator(ContentGenerator):
    name = "pitching_summary"

    async def generate(self) -> PostContent:
        # Use current season only
        image_path = None
        season_df = pitch_profiler.get_season_pitchers(MLB_SEASON)
        if season_df.empty:
            log.warning("No season data for %d", MLB_SEASON)
            return PostContent(text="")
        pitches_df = pitch_profiler.get_season_pitches(MLB_SEASON)

        # Candidates come from the QUALIFIED SEASON FIELD, not data/players.json.
        # That watchlist holds 20 names; posting daily works through it in 20 days
        # and then logs "All players posted recently, resetting pool" and starts
        # repeating. The season board has ~198 pitchers over 250 pitches, which is
        # a real pool -- and it stays current without anyone maintaining a list.
        # pick_player() remains the fallback for when the API is short.
        candidates = _candidate_pitchers(season_df)

        for attempt in range(3):
            player_info = candidates[attempt] if attempt < len(candidates) else pick_player()
            name = player_info["name"]
            team = player_info.get("team")
            player_id = player_info.get("id")

            image_path = plot_pitching_summary(
                name, season_df, pitches_df,
                team=team, player_id=player_id,
            )
            if image_path:
                break
            log.warning("Pitching summary failed for %s (season=%d, attempt %d)",
                        name, MLB_SEASON, attempt + 1)

        if not image_path:
            log.warning("All pitching summary attempts failed")
            return PostContent(text="")

        # Build tweet text with key stats
        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in season_df.columns:
                name_col = c
                break

        summary_parts: list[str] = []
        if name_col:
            matches = season_df[season_df[name_col] == name]
            if not matches.empty:
                p = matches.iloc[0]
                for col, label, fmt in [
                    ("era", "ERA", ".2f"),
                    ("fip", "FIP", ".2f"),
                    ("strike_out_percentage", "K%", None),
                    ("whiff_rate", "Whiff%", None),
                ]:
                    if col in p.index:
                        try:
                            val = float(p[col])
                            if fmt:
                                summary_parts.append(f"{label}: {format(val, fmt)}")
                            else:
                                summary_parts.append(f"{label}: {val * 100:.1f}%")
                        except (TypeError, ValueError):
                            pass

        # AI analysis
        analysis_text = analyze_pitcher(name, season_df, pitches_df)

        # Fetch video clip
        video_path = None
        if player_id:
            try:
                video_path = get_pitcher_clip(player_id, name)
                if video_path:
                    log.info("Got video clip for %s: %s", name, video_path)
            except Exception:
                log.warning("Video clip fetch failed for %s", name, exc_info=True)

        stat_line = " | ".join(summary_parts) if summary_parts else ""

        # Lead with the take, not the title
        if analysis_text:
            text = (
                f"{analysis_text}"
                f"\n\n{name}'s {MLB_SEASON} Pitching Summary"
                f"\n\n@TJStats {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"{name}'s {MLB_SEASON} Pitching Summary"
                f"\n\n@TJStats {DEFAULT_HASHTAGS}"
            )

        # Stats go in the reply (the graphic already shows them)
        reply_content = None
        if stat_line:
            reply_content = PostContent(
                text=f"{name} | {stat_line}",
                tags=["stats"],
            )

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Season pitching summary dashboard for {name}",
            tags=["pitching_summary", name],
            reply=reply_content,
        )
