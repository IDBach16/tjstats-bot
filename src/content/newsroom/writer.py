"""Columnist: Claude writes the Barstool-voice thread from the fact sheet only."""

from __future__ import annotations

import json
import logging
import os
import re

from .personas import STYLE_GUIDE, WRITER_MODEL, HELPER_MODEL

log = logging.getLogger(__name__)

MAX_TWEET = 260  # leave headroom under X's 280


def _complete(system: str, prompt: str, max_tokens: int = 3000) -> str | None:
    """Call Claude with the writer model; fall back to the cheap model on error.

    max_tokens must be generous: WRITER_MODEL (Sonnet 5) emits a *thinking* block
    before the text block, so a tight budget (e.g. 900) gets consumed by thinking
    and truncates before any text is emitted — the join below then finds no text
    block and every thread silently falls back to the cheap model.
    """
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic not installed")
        return None
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set — cannot write article")
        return None
    client = anthropic.Anthropic(api_key=key)
    for model in (WRITER_MODEL, HELPER_MODEL):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            # Join text blocks; models with extended thinking put a ThinkingBlock first.
            text = "".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"
            ).strip()
            if text:
                return text
            log.warning("writer model %s returned no text block", model)
        except Exception as e:
            log.warning("writer model %s failed: %s", model, e)
    return None


def _parse_json(text: str) -> dict | None:
    """Pull the JSON object out of the model response, tolerant of code fences."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # last-ditch: strip trailing commas
        cleaned = re.sub(r",\s*([\]}])", r"\1", m.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("could not parse writer JSON")
            return None


def write_thread(fact_sheet: dict, persona: dict,
                 revision_notes: list[str] | None = None) -> dict | None:
    """Return {'headline': str, 'tweets': [str, ...]} or None.

    If revision_notes is given (from the copy desk), the writer must fix those
    problems and re-submit.
    """
    fix_block = ""
    if revision_notes:
        fix_block = ("\n\nA FACT-CHECKER REJECTED YOUR LAST DRAFT. Fix these and use ONLY "
                     "fact-sheet numbers:\n- " + "\n- ".join(revision_notes) + "\n")

    art = fact_sheet.get("article")
    if fact_sheet.get("kind") == "article" and art:
        prompt = f"""You are {persona['name']} for BachTalk. {persona['blurb']}

React to this freshly published article with an X (Twitter) THREAD in your voice.{fix_block}

ARTICLE (summarize and react — do NOT reproduce its wording):
Outlet: {art.get('outlet')}
Author: {art.get('author') or 'staff'}
Headline: {art.get('title')}
Summary: {art.get('summary')}

FORMAT — return ONLY valid JSON, no markdown:
{{"headline": "<a punchy blog-style headline>",
  "tweets": ["<hook>", "<body>", "<body>", "<kicker>"]}}

RULES
- tweets[0] is the HOOK: lead with the single most interesting finding, in your voice.
- Then 2-4 tweets that unpack the key insight in plain English and add YOUR take.
- CREDIT the source by name at least once (e.g. "Per {art.get('outlet')}'s {art.get('author') or 'staff'}…"). We add the link separately, so don't paste a URL.
- Use ONLY facts and numbers found in the summary above — invent nothing. If you're unsure of a specific number, speak generally.
- Final tweet is a KICKER: a bold, original take (not a stat).
- Each tweet must be UNDER {MAX_TWEET} characters. No hashtags (we add those).
  Emojis okay but sparingly. Do NOT number the tweets. Do NOT mention an attached video/image.
"""
    else:
        prompt = f"""You are {persona['name']} for BachTalk. {persona['blurb']}

Write an X (Twitter) THREAD in your voice about this story.{fix_block}

FACT SHEET (use ONLY these numbers — invent nothing):
{fact_sheet['sheet']}

FORMAT — return ONLY valid JSON, no markdown:
{{"headline": "<a punchy blog-style headline>",
  "tweets": ["<hook>", "<body>", "<body>", "<kicker>"]}}

RULES
- tweets[0] is the HOOK: make someone stop scrolling. Name the player and the wildest number.
- Then 2-4 tweets that build the case in plain English, with attitude.
- Final tweet is a KICKER: a bold prediction or mic-drop line.
- Each tweet must be UNDER {MAX_TWEET} characters. No hashtags (we add those).
  Emojis okay but sparingly. Do NOT number the tweets. Do NOT mention an attached video.
"""
    raw = _complete(STYLE_GUIDE, prompt)
    data = _parse_json(raw or "")
    if not data or not isinstance(data.get("tweets"), list) or not data["tweets"]:
        log.warning("writer produced no usable thread for %s", fact_sheet.get("subject"))
        return None
    # keep only non-empty strings
    data["tweets"] = [str(t).strip() for t in data["tweets"] if str(t).strip()]
    data["headline"] = str(data.get("headline", "")).strip()
    return data if data["tweets"] else None
