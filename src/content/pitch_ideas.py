"""Rotating pitching-analysis posts (the 17:00 UTC 'evening' slot).

Four ideas, each built on one of Lance Brozdowski's principles; the finding is
computed in pitching_ideas, the model only explains it through the shared lens:

  ThrowItMoreGenerator      his best pitch is one he barely throws
  EraVsSkillGenerator       ERA a full run from xERA, either direction
  BestOffspeedGenerator     biggest drop gap off the fastball (CH/FS)
  BatMissersGenerator       whiff rate furthest above its pitch-type norm

Every post: the pitcher's season Pitching Summary card, a two-sentence take,
and as the video reply a strikeout clip of THE pitch the post is about
(get_pitch_clip), falling back to any strikeout clip of his. Pitchers already
featured by the same idea are skipped, so the pools (40-560 candidates) last.
"""

from __future__ import annotations

import logging
import os

from .base import ContentGenerator, PostContent
from .. import savant_pitching
from .. import pitching_ideas
from ..analysis import _get_client
from ..charts import plot_pitching_summary
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..pitching_lens import analyst_request, reply_text
from ..video_clips import get_pitch_clip, get_pitcher_clip

log = logging.getLogger(__name__)


def _take(prompt: str) -> str | None:
    """Two-sentence take through the shared lens; one rewrite if it runs long."""
    client = _get_client()
    if not client:
        return None
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = reply_text(client.messages.create(**analyst_request(), messages=msgs))
        if len(text) > 270:
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"That is {len(text)} characters. Rewrite the same "
                                                 "take in under 240 characters. Output only the sentences."}]
            text = reply_text(client.messages.create(**analyst_request(), messages=msgs)) or text
        if len(text) > 270:
            text = text[:267] + "..."
        return text or None
    except Exception:
        log.warning("Idea take generation failed", exc_info=True)
        return None


_RULES = """Rules:
- Two sentences, UNDER 240 characters total, one idea.
- Explain WHY through the framework in your instructions; cite only the numbers above.
- Do NOT use the word "elite". No hashtags, emojis, @ mentions, dashes or hyphens.
- Do NOT start with the pitcher's name (the post already has it).
- Not in this data: arm slot, release, extension, location and age. Make no claim about them.
- Output ONLY the sentences."""


class _IdeaGenerator(ContentGenerator):
    name = "pitch_idea"
    title = ""            # e.g. "Throw It More"
    angle = ""            # what the finding means, for the prompt

    def candidates(self, season_df, pitches_df) -> list[dict]:
        raise NotImplementedError

    def leaderboard(self, cands: list[dict]) -> str | None:
        """Optional extra reply listing the next few names."""
        return None

    async def generate(self) -> PostContent:
        from ..scheduler import recent_generator_tags

        season_df = savant_pitching.get_season_pitchers(MLB_SEASON)
        pitches_df = savant_pitching.get_season_pitches(MLB_SEASON)
        if season_df.empty or pitches_df.empty:
            log.warning("%s: no Savant data", self.name)
            return PostContent(text="")

        cands = self.candidates(season_df, pitches_df)
        # Manual re-run of a named pitcher (workflow input `player`).
        override = os.environ.get("PITCHER_OVERRIDE", "").strip().lower()
        if override:
            cands = [c for c in cands if c["pitcher_name"].lower() == override]
        else:
            posted = recent_generator_tags(self.name, index=1)
            cands = [c for c in cands if c["pitcher_name"] not in posted]
        if not cands:
            log.info("%s: no unposted candidate", self.name)
            return PostContent(text="")

        for pick in cands[:3]:
            name, pid, pt = pick["pitcher_name"], pick["pitcher_id"], pick["pitch_type"]
            image = plot_pitching_summary(name, season_df, pitches_df, player_id=pid)
            if image:
                break
        else:
            log.warning("%s: card failed for the top candidates", self.name)
            return PostContent(text="")

        facts = "\n".join(f"  {k}: {v}" for k, v in pick["facts"].items())
        take = _take(f"""You are a sharp baseball analyst writing for Twitter/X about {name}'s {MLB_SEASON} season.

The finding ({self.title}): {self.angle}

The numbers:
{facts}

{_RULES}""")

        video = None
        try:
            video = get_pitch_clip(pid, name, pt) if pt else get_pitcher_clip(pid, name)
        except Exception:
            log.warning("%s: clip fetch failed for %s", self.name, name, exc_info=True)

        head = f"{self.title}: {name}"
        text = (f"{take}\n\n{head}" if take else head) + f"\n\n@TJStats {DEFAULT_HASHTAGS}"
        # main.py posts .reply as the video reply's text (one tweet, so <=280).
        board = self.leaderboard(cands)
        if board and len(board) > 280:
            board = board[:277] + "..."
        return PostContent(
            text=text, image_path=image, video_path=video,
            alt_text=f"{MLB_SEASON} season pitching summary dashboard for {name}",
            tags=[self.name, name, pt],
            reply=PostContent(text=board, tags=["leaderboard"]) if board else None,
        )


class ThrowItMoreGenerator(_IdeaGenerator):
    name = "throw_it_more"
    title = "Throw It More"
    angle = ("one of his pitches beats his most-used pitch on both whiff rate and xwOBA, yet he "
             "barely throws it. Usage is the cheapest fix a pitcher has.")

    def candidates(self, season_df, pitches_df):
        return pitching_ideas.throw_it_more(season_df, pitches_df)


class EraVsSkillGenerator(_IdeaGenerator):
    name = "era_vs_skill"
    title = "ERA vs Skill"
    angle = ("his ERA sits a full run or more away from his expected ERA (xERA), a gap that "
             "demands an explanation: luck, defense, or a profile that really is different. "
             "Say which way it is likely to move and why, using his strikeout, walk and contact numbers.")

    def candidates(self, season_df, pitches_df):
        return pitching_ideas.era_vs_skill(savant_pitching.get_expected_stats(MLB_SEASON), season_df)


class BestOffspeedGenerator(_IdeaGenerator):
    name = "best_offspeed"
    title = "Best Offspeed Shapes"
    angle = ("this changeup or splitter drops more relative to his fastball than almost any in "
             "baseball. Drop separation is the trait that makes an offspeed pitch miss bats.")

    def candidates(self, season_df, pitches_df):
        return pitching_ideas.best_offspeed(season_df, pitches_df)

    def leaderboard(self, cands):
        rows = [f"{i}. {c['pitcher_name']} {c['facts']['pitch']}: {c['facts']['drop gap']} drop gap, "
                f"{c['facts']['whiff']} whiff" for i, c in enumerate(cands[:3], 1)]
        return f"Biggest offspeed drop gaps off the fastball, {MLB_SEASON}:\n" + "\n".join(rows)


class BatMissersGenerator(_IdeaGenerator):
    name = "bat_missers"
    title = "Bat-Missing Outliers"
    angle = ("this pitch's whiff rate is one of the furthest above the norm for its pitch type in "
             "baseball. Explain what about it (velocity, shape, usage) lets it miss so many bats.")

    def candidates(self, season_df, pitches_df):
        return pitching_ideas.bat_missers(season_df, pitches_df)

    def leaderboard(self, cands):
        rows = [f"{i}. {c['pitcher_name']} {c['facts']['pitch']}: {c['facts']['whiff']} whiff "
                f"({c['facts']['whiff vs pitch-type norm']} vs the norm)" for i, c in enumerate(cands[:3], 1)]
        return f"Whiff rates furthest above their pitch-type norm, {MLB_SEASON}:\n" + "\n".join(rows)
