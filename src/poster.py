"""X/Twitter posting via tweepy (v1.1 media upload + v2 tweet creation)."""

from __future__ import annotations

import logging
from pathlib import Path

import tweepy

from .config import (
    X_API_KEY,
    X_API_SECRET,
    X_ACCESS_TOKEN,
    X_ACCESS_TOKEN_SECRET,
    X_BEARER_TOKEN,
)

log = logging.getLogger(__name__)


def _get_v1_api() -> tweepy.API:
    """OAuth 1.0a for media upload (v1.1 endpoint)."""
    auth = tweepy.OAuth1UserHandler(
        X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
    )
    return tweepy.API(auth)


def _get_v2_client() -> tweepy.Client:
    """OAuth 2.0 Bearer + user context for tweet creation (v2 endpoint)."""
    return tweepy.Client(
        bearer_token=X_BEARER_TOKEN,
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_TOKEN_SECRET,
    )


def post_text(text: str) -> str:
    """Post a text-only tweet. Returns the tweet ID."""
    client = _get_v2_client()
    resp = client.create_tweet(text=text)
    tweet_id = resp.data["id"]
    log.info("Posted text tweet %s", tweet_id)
    return tweet_id


def post_with_image(text: str, image_path: Path | str, alt_text: str = "") -> str:
    """Upload image via v1.1, then post tweet via v2. Returns tweet ID."""
    v1 = _get_v1_api()
    media = v1.media_upload(filename=str(image_path))
    if alt_text:
        v1.create_media_metadata(media.media_id, alt_text=alt_text)
    log.info("Uploaded media %s (%s)", media.media_id, image_path)

    client = _get_v2_client()
    resp = client.create_tweet(text=text, media_ids=[media.media_id])
    tweet_id = resp.data["id"]
    log.info("Posted image tweet %s", tweet_id)
    return tweet_id


def post_reply(
    text: str,
    in_reply_to: str,
    image_path: Path | str | None = None,
    alt_text: str = "",
) -> str:
    """Post a reply tweet. Returns the reply tweet ID."""
    media_ids = None
    if image_path:
        v1 = _get_v1_api()
        media = v1.media_upload(filename=str(image_path))
        if alt_text:
            v1.create_media_metadata(media.media_id, alt_text=alt_text)
        media_ids = [media.media_id]

    client = _get_v2_client()
    kwargs: dict = {"text": text, "in_reply_to_tweet_id": in_reply_to}
    if media_ids:
        kwargs["media_ids"] = media_ids
    resp = client.create_tweet(**kwargs)
    tweet_id = resp.data["id"]
    log.info("Posted reply tweet %s (in reply to %s)", tweet_id, in_reply_to)
    return tweet_id


def post_video_reply(in_reply_to: str, video_path: Path | str, text: str = "") -> str:
    """Upload video via v1.1 chunked upload and reply to a tweet. Returns reply tweet ID."""
    v1 = _get_v1_api()
    media = v1.chunked_upload(
        filename=str(video_path),
        media_category="tweet_video",
    )
    log.info("Uploaded video %s (%s)", media.media_id, video_path)

    client = _get_v2_client()
    resp = client.create_tweet(
        text=text,
        media_ids=[media.media_id],
        in_reply_to_tweet_id=in_reply_to,
    )
    tweet_id = resp.data["id"]
    log.info("Posted video reply %s (in reply to %s)", tweet_id, in_reply_to)
    return tweet_id


def post_with_video(text: str, video_path: Path | str) -> str:
    """Upload video via v1.1 chunked upload, then post tweet via v2. Returns tweet ID."""
    v1 = _get_v1_api()
    media = v1.chunked_upload(
        filename=str(video_path),
        media_category="tweet_video",
    )
    log.info("Uploaded video %s (%s)", media.media_id, video_path)

    client = _get_v2_client()
    resp = client.create_tweet(text=text, media_ids=[media.media_id])
    tweet_id = resp.data["id"]
    log.info("Posted video tweet %s", tweet_id)
    return tweet_id
