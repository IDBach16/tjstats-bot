"""Orchestrator: run the newsroom pipeline and return a ready-to-post thread.

feeds -> pick a story WITH available game film -> researcher -> writer -> social.

Rules baked in:
  * Every thread leads with a real game video clip. If a subject has no clip, we
    skip to the next candidate (never post video-less).
  * Reds lean comes from feeds (a Red near the top of a board is preferred).
  * Story kind rotates by day so the account doesn't repeat itself.

Phase 2 will slot an LLM assignment editor + copy-desk fact-checker into this flow.
"""

from __future__ import annotations

import logging
import os
from datetime import date

from ..base import ContentGenerator, PostContent
from ...video_clips import get_pitcher_clip, get_hitter_clip
from . import feeds, researcher, social, personas

log = logging.getLogger(__name__)

KIND_ROTATION = ["nasty_pitch", "overperformer", "bat_speed", "underperformer"]
MAX_CANDIDATES = 6  # bound how many clip lookups we attempt per run


class NewsroomGenerator(ContentGenerator):
    name = "newsroom"

    async def generate(self) -> PostContent:
        try:
            leads_by_kind = feeds.build_leads()
        except Exception:
            log.warning("newsroom: feeds failed", exc_info=True)
            return PostContent(text="", tags=["newsroom", "error"])

        candidates = self._order_candidates(leads_by_kind)
        if not candidates:
            log.warning("newsroom: no candidate leads")
            return PostContent(text="", tags=["newsroom", "no_leads"])

        for lead in candidates[:MAX_CANDIDATES]:
            clip = self._get_clip(lead)
            if not clip:
                log.info("newsroom: no clip for %s (%s) — trying next",
                         lead.subject, lead.kind)
                continue

            fact_sheet = researcher.build_fact_sheet(lead)
            article = self._write(fact_sheet)
            if not article:
                continue

            log.info("newsroom: publishing '%s' on %s (%s%s)",
                     article.get("headline", "")[:60], lead.subject, lead.kind,
                     ", RED" if lead.is_red else "")
            return social.build_post(article, lead, clip)

        log.warning("newsroom: no publishable story with a clip today")
        return PostContent(text="", tags=["newsroom", "no_clip"])

    # ── helpers ────────────────────────────────────────────────────────
    def _order_candidates(self, leads_by_kind: dict) -> list:
        """Rotate the featured kind by day; env NEWSROOM_KIND forces one."""
        override = os.environ.get("NEWSROOM_KIND")
        if override in KIND_ROTATION:
            order = [override] + [k for k in KIND_ROTATION if k != override]
        else:
            rot = date.today().timetuple().tm_yday % len(KIND_ROTATION)
            order = KIND_ROTATION[rot:] + KIND_ROTATION[:rot]

        ordered: list = []
        for k in order:
            ordered.extend(leads_by_kind.get(k, []))
        return ordered

    def _get_clip(self, lead):
        try:
            if lead.is_pitcher:
                return get_pitcher_clip(lead.player_id, lead.subject)
            return get_hitter_clip(lead.player_id, lead.subject)
        except Exception:
            log.warning("newsroom: clip fetch failed for %s", lead.subject, exc_info=True)
            return None

    def _write(self, fact_sheet: dict):
        from . import writer
        try:
            return writer.write_thread(fact_sheet, personas.default_persona())
        except Exception:
            log.warning("newsroom: writer failed for %s",
                        fact_sheet.get("subject"), exc_info=True)
            return None
