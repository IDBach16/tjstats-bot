"""Social editor: package the written thread into a PostContent for the poster.

The lead tweet carries the game video clip (per the 'every post has film' rule);
the body tweets follow as a reply chain, capped with the BachTalk sign-off.
"""

from __future__ import annotations

from pathlib import Path

from ..base import PostContent
from .feeds import Lead
from .personas import SIGN_OFF
from .writer import MAX_TWEET

HARD_LIMIT = 278  # absolute X ceiling with a little safety


def _fit(text: str) -> str:
    """Guarantee a tweet fits X's limit; trim on a word boundary if needed."""
    text = text.strip()
    if len(text) <= HARD_LIMIT:
        return text
    cut = text[:HARD_LIMIT - 1]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip() + "…"


def build_post(article: dict, lead: Lead, video_path: Path) -> PostContent:
    tweets = [_fit(t) for t in article["tweets"]]
    headline = article.get("headline", "") or lead.subject

    replies: list[PostContent] = []
    for t in tweets[1:]:
        replies.append(PostContent(text=t, tags=["newsroom", lead.kind]))
    replies.append(PostContent(text=SIGN_OFF, tags=["newsroom", "signoff"]))

    return PostContent(
        text=tweets[0],
        video_path=video_path,          # clip rides on the lead tweet
        alt_text=headline[:1000],
        tags=["newsroom", lead.kind, lead.subject],
        replies=replies,
    )
