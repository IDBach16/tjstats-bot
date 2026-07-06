"""Assignment editor: rank today's candidate leads for the best thread.

Replaces the Phase-1 day-rotation with a real editorial judgment call, while
still returning a *ranked list* so the orchestrator can fall through to the next
story if the top pick has no game film.
"""

from __future__ import annotations

import json
import logging
import os
import re

from .personas import HELPER_MODEL
from .feeds import Lead

log = logging.getLogger(__name__)


def _key_stat(lead: Lead) -> str:
    f = lead.facts
    if lead.kind in ("overperformer", "underperformer"):
        return (f"wOBA {f.get('woba')} vs xwOBA {f.get('est_woba')} "
                f"(gap {f.get('gap')}), {f.get('pa')} PA")
    if lead.kind == "nasty_pitch":
        return (f"{f.get('pitch_name')} — {f.get('whiff_percent')}% whiff "
                f"(lg {f.get('league_whiff')}%)")
    if lead.kind == "bat_speed":
        return f"{f.get('avg_bat_speed')} mph bat speed (lg {f.get('league_bat_speed')})"
    return ""


def rank(candidates: list[Lead]) -> list[Lead]:
    """Return candidates re-ordered best-first. Falls back to input order on error."""
    if len(candidates) <= 1:
        return candidates

    digest = "\n".join(
        f"[{i}] {l.kind} — {l.subject}{' (REDS)' if l.is_red else ''}: "
        f"{_key_stat(l)} | ranks #{l.rank}/{l.total}"
        for i, l in enumerate(candidates)
    )
    prompt = f"""You are the assignment editor for BachTalk, a Barstool-style baseball account.
Here are today's candidate stories, each backed by real Baseball Savant data:

{digest}

Rank them from MOST to least compelling for a punchy thread today. Favor:
- extreme, surprising, or league-leading numbers
- a clear "nobody's talking about this" narrative
- Reds angles (small bonus — don't force it)
- variety (don't stack five of the same kind at the top)

Return ONLY JSON: {{"ranked": [the [index] numbers, best first]}}"""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=HELPER_MODEL, max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        order = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))["ranked"]
        ranked = [candidates[i] for i in order
                  if isinstance(i, int) and 0 <= i < len(candidates)]
        # Append anything the editor left out so video-gating still has fallbacks.
        seen = {id(x) for x in ranked}
        ranked += [c for c in candidates if id(c) not in seen]
        if ranked:
            log.info("editor: top pick %s (%s%s)", ranked[0].subject, ranked[0].kind,
                     ", RED" if ranked[0].is_red else "")
            return ranked
    except Exception:
        log.warning("editor ranking failed; using feed order", exc_info=True)
    return candidates
