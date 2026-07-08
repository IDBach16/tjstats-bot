"""Social editor: package the written thread into a PostContent for the poster.

The lead tweet carries the game video clip (per the 'every post has film' rule);
the body tweets follow as a reply chain, capped with the BachTalk sign-off.
"""

from __future__ import annotations

from pathlib import Path

from ..base import PostContent
from .feeds import Lead
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


def build_post(article: dict, lead: Lead, video_path: Path | None = None,
               chart_path: Path | None = None) -> PostContent:
    tweets = [_fit(t) for t in article["tweets"]]
    headline = article.get("headline", "") or lead.subject
    is_article = lead.kind == "article"

    replies: list[PostContent] = []
    body = tweets[1:]
    for i, t in enumerate(body):
        reply = PostContent(text=t, tags=["newsroom", lead.kind])
        # Video-led threads carry the hero card on the first reply; article
        # threads have no video, so their source card rides the lead tweet.
        if i == 0 and chart_path is not None and not is_article:
            reply.image_path = chart_path
            reply.alt_text = f"{lead.subject} stat card"
        replies.append(reply)
    # if a video-led thread had no body tweets, still show the card
    if chart_path is not None and not body and not is_article:
        replies.append(PostContent(text="The receipts 📊", image_path=chart_path,
                                   alt_text=f"{lead.subject} stat card",
                                   tags=["newsroom", lead.kind]))

    # Article threads: append an attribution + link-back reply.
    if is_article:
        url = lead.facts.get("url")
        outlet = lead.facts.get("outlet", "the original")
        if url:
            replies.append(PostContent(
                text=f"📄 Full piece via {outlet}: {url}",
                tags=["newsroom", "article", "source_link"]))

    post = PostContent(
        text=tweets[0],
        alt_text=headline[:1000],
        tags=["newsroom", lead.kind, lead.subject],
        replies=replies,
    )
    if is_article:
        post.image_path = chart_path        # source card on the lead tweet
        if chart_path is not None:
            post.alt_text = f"{lead.subject} — via {lead.facts.get('outlet', '')}"
    else:
        post.video_path = video_path         # clip rides on the lead tweet
    return post
