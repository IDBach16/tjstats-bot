"""Copy desk: independently fact-check the drafted thread against the fact sheet.

The writer is already grounded (it only sees verified numbers), so this is a
second line of defense: catch a fabricated or contradicted STAT before it posts.
Rhetorical numbers ("top 5", "half"), jokes, and predictions are fine.
"""

from __future__ import annotations

import json
import logging
import os
import re

from .personas import HELPER_MODEL

log = logging.getLogger(__name__)


def review(article: dict, fact_sheet: dict) -> dict:
    """Return {'ok': bool, 'issues': [str]}. Fails OPEN (writer is grounded)."""
    thread = "\n".join(article.get("tweets", []))
    prompt = f"""You are the copy-desk fact-checker for a baseball account. Compare the DRAFT
THREAD to the FACT SHEET. Flag ONLY real problems:
- a statistic that is fabricated or contradicts the fact sheet
- a wrong player name, team, or a factual claim the sheet doesn't support
IGNORE rhetorical numbers ("top 5", "half"), opinions, jokes, and predictions.

FACT SHEET:
{fact_sheet['sheet']}

DRAFT THREAD:
{thread}

Return ONLY JSON: {{"ok": true/false, "issues": ["short problem description", ...]}}
ok=true means every hard stat in the draft is supported by the fact sheet."""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=HELPER_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        data = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        ok = bool(data.get("ok"))
        issues = [str(x) for x in data.get("issues", []) if str(x).strip()]
        if not ok:
            log.info("copydesk flagged: %s", " | ".join(issues) or "unspecified")
        return {"ok": ok, "issues": issues}
    except Exception:
        log.warning("copydesk failed; passing draft through", exc_info=True)
        return {"ok": True, "issues": []}
