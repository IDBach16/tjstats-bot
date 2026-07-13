"""Orchestrator: run the newsroom pipeline and return a ready-to-post thread.

feeds -> editor (rank) -> pick a story WITH game film -> researcher -> writer
       -> copy desk (fact-check, one retry) -> graphics (hero card) -> social.

Rules baked in:
  * Every thread leads with a real game video clip. If a subject has no clip, we
    skip to the next candidate (never post video-less).
  * The columnist only ever sees verified numbers; the copy desk re-checks the
    draft and can send it back once before we give up on that story.
  * Coverage is all-MLB and merit-based; the assignment editor ranks candidates.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta

from ..base import ContentGenerator, PostContent
from ...config import DATA_DIR
from ...video_clips import get_pitcher_clip, get_hitter_clip
from . import feeds, researcher, social, personas, editor, graphics

log = logging.getLogger(__name__)

# Player-analysis story kinds only (the Moniak/overperformer style). "article"
# (react to a FanGraphs/Savant piece) is intentionally DISABLED — Ian prefers
# original single-player threads over article link-backs. The article pipeline
# still lives in articles.py; re-add "article" here to turn it back on.
KIND_ROTATION = ["nasty_pitch", "overperformer", "bat_speed", "underperformer",
                 "hard_contact", "pitcher_stuff", "pitcher_luck"]
MAX_CANDIDATES = 6  # bound how many clip lookups / write attempts per run


class NewsroomGenerator(ContentGenerator):
    name = "newsroom"

    # Subclasses set force_kind to dedicate the slot to one story kind. None =
    # rank the whole board.
    force_kind: str | None = None
    # If a forced kind finds nothing publishable, fall back to the full board so
    # the slot never wastes. Subclasses can turn this off to stay single-kind
    # (post nothing rather than something off-topic).
    fallback_to_board: bool = True

    async def generate(self) -> PostContent:
        try:
            leads_by_kind = feeds.build_leads()
        except Exception:
            log.warning("newsroom: feeds failed", exc_info=True)
            return PostContent(text="", tags=["newsroom", "error"])

        seed = date.today().timetuple().tm_yday
        # An env override (for testing one kind) wins; otherwise a dedicated
        # slot's force_kind biases the pick.
        env_kind = os.environ.get("NEWSROOM_KIND")
        only = env_kind or self.force_kind
        if only not in KIND_ROTATION:
            only = None

        post = self._run(self._candidates(leads_by_kind, only), seed)
        if post is not None:
            return post

        # Dedicated-slot fallback: a forced kind found nothing publishable, so
        # fall back to the full board rather than post nothing. (An explicit env
        # override is a test of one kind — don't fall back in that case.)
        if only and not env_kind and self.fallback_to_board:
            log.info("newsroom: no publishable '%s' story — falling back to full board", only)
            post = self._run(self._candidates(leads_by_kind, None), seed)
            if post is not None:
                return post

        log.warning("newsroom: no publishable story today")
        return PostContent(text="", tags=["newsroom", "no_story"])

    def _run(self, candidates: list, seed: int) -> PostContent | None:
        """Rank candidates and publish the first one we can fully build."""
        if not candidates:
            return None
        ranked = editor.rank(candidates)
        for lead in ranked[:MAX_CANDIDATES]:
            if self._recently_covered(lead.subject):
                log.info("newsroom: %s covered recently — next", lead.subject)
                continue
            # Every newsroom thread MUST lead with real game film — no usable
            # video clip, no post. Try the next candidate; publish nothing if
            # none has one. Absolute rule, no per-kind exemption.
            clip = self._get_clip(lead)
            if not (clip and clip.exists()):
                log.info("newsroom: no usable video for %s (%s) — next",
                         lead.subject, lead.kind)
                continue

            fact_sheet = researcher.build_fact_sheet(lead)
            persona = personas.pick_persona(seed)
            article = self._write_and_check(fact_sheet, persona)
            if not article:
                continue

            chart = graphics.render_stat_card(fact_sheet, lead)
            log.info("newsroom: publishing '%s' — %s (%s%s), by %s",
                     article.get("headline", "")[:60], lead.subject, lead.kind,
                     ", RED" if lead.is_red else "", persona["name"])
            return social.build_post(article, lead, clip, chart)
        return None

    # ── helpers ────────────────────────────────────────────────────────
    def _candidates(self, leads_by_kind: dict, only: str | None = None) -> list:
        """Candidate leads. ``only`` (a kind) restricts to that bucket; None
        returns the whole board (the caller resolves env/force_kind first)."""
        if only in KIND_ROTATION:
            return list(leads_by_kind.get(only, []))
        out: list = []
        for k in KIND_ROTATION:
            out.extend(leads_by_kind.get(k, []))
        return out

    def _recently_covered(self, subject: str, days: int = 14) -> bool:
        """True if the newsroom already featured this subject in the last `days`."""
        try:
            path = DATA_DIR / "post_history.json"
            if not path.exists():
                return False
            posts = json.loads(path.read_text()).get("posts", [])
            cutoff = datetime.utcnow() - timedelta(days=days)
            for entry in posts:
                if not str(entry.get("generator") or "").startswith("newsroom"):
                    continue
                try:
                    when = datetime.fromisoformat(entry["date"])
                except (KeyError, ValueError):
                    continue
                if when >= cutoff and subject in (entry.get("tags") or []):
                    return True
        except Exception:
            log.debug("recently-covered check failed", exc_info=True)
        return False

    def _get_clip(self, lead):
        try:
            if lead.is_pitcher:
                return get_pitcher_clip(lead.player_id, lead.subject)
            return get_hitter_clip(lead.player_id, lead.subject)
        except Exception:
            log.warning("newsroom: clip fetch failed for %s", lead.subject, exc_info=True)
            return None

    def _write_and_check(self, fact_sheet: dict, persona: dict):
        """Write the thread, then fact-check it. One revision, then give up."""
        from . import writer, copydesk
        try:
            article = writer.write_thread(fact_sheet, persona)
            if not article:
                return None
            verdict = copydesk.review(article, fact_sheet)
            if not verdict["ok"]:
                log.info("newsroom: copydesk sent back %s — revising", fact_sheet["subject"])
                article = writer.write_thread(fact_sheet, persona,
                                              revision_notes=verdict["issues"])
                if not article or not copydesk.review(article, fact_sheet)["ok"]:
                    log.warning("newsroom: %s failed fact-check twice — skipping",
                                fact_sheet["subject"])
                    return None
            return article
        except Exception:
            log.warning("newsroom: write/check failed for %s",
                        fact_sheet.get("subject"), exc_info=True)
            return None


class NewsroomArticleGenerator(NewsroomGenerator):
    """Dedicated daily article-reaction slot: react to a fresh FanGraphs /
    Baseball Savant piece. If there's nothing worth reacting to that day, post
    nothing (no data-thread filler) — the daily data newsroom covers that."""
    name = "newsroom_article"
    force_kind = "article"
    fallback_to_board = False
