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

KIND_ROTATION = ["nasty_pitch", "overperformer", "bat_speed", "underperformer",
                 "article"]
MAX_CANDIDATES = 6  # bound how many clip lookups / write attempts per run


class NewsroomGenerator(ContentGenerator):
    name = "newsroom"

    # Subclasses set force_kind to dedicate the slot to one story kind (with a
    # graceful fallback to the full board). None = rank the whole board.
    force_kind: str | None = None

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
        if only and not env_kind:
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
            # Article reactions lead with a source card, not game film, so they
            # skip the clip requirement; every other kind must have a clip.
            clip = None if lead.kind == "article" else self._get_clip(lead)
            if lead.kind != "article" and not clip:
                log.info("newsroom: no clip for %s (%s) — next", lead.subject, lead.kind)
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
                if entry.get("generator") != "newsroom":
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
    """Dedicated article-reaction slot (a few days a week): prefer a fresh
    FanGraphs / Baseball Savant article, but fall back to the full data board
    if there's nothing worth reacting to that day, so the slot never wastes."""
    name = "newsroom_article"
    force_kind = "article"
