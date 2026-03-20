"""Entry point / orchestrator for the TJStats Bot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from . import poster
from .scheduler import get_generators_for_today, get_daily_generators, record_post, GENERATORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

DRY_RUN = False


async def run_post(slot: str, generator_name: str | None = None) -> None:
    """Generate and post content for a given slot.

    Slots: 'screenshot' (morning), 'text' (afternoon), 'evening',
           'biomechanics' (4th slot, some days), 'daily', or 'all'.
    If generator_name is provided, run that specific generator instead.
    """
    if generator_name:
        cls = GENERATORS.get(generator_name)
        if not cls:
            log.error("Unknown generator: %s (available: %s)",
                      generator_name, ", ".join(GENERATORS))
            return
        await _generate_and_post(cls())
        return

    gens = get_generators_for_today()

    if slot == "screenshot":
        await _generate_and_post(gens[0])
    elif slot == "text":
        await _generate_and_post(gens[1])
    elif slot == "evening":
        await _generate_and_post(gens[2])
    elif slot == "biomechanics":
        if len(gens) > 3:
            await _generate_and_post(gens[3])
        else:
            log.info("No biomechanics slot scheduled for today")
    elif slot == "daily":
        for gen in get_daily_generators():
            await _generate_and_post(gen)
    else:
        # "all" — run rotation + daily generators
        for gen in gens:
            await _generate_and_post(gen)
        for gen in get_daily_generators():
            await _generate_and_post(gen)


async def _generate_and_post(generator) -> None:
    log.info("Running generator: %s", generator.name)
    content = await generator.generate()

    if not content.text:
        log.warning("Generator %s produced empty content — skipping", generator.name)
        return

    log.info("Posting: %s", content.text[:80])

    has_image = content.image_path and content.image_path.exists()
    has_video = content.video_path and content.video_path.exists()

    if DRY_RUN:
        media_parts = []
        if has_image:
            media_parts.append(f"image {content.image_path}")
        if has_video:
            media_parts.append(f"video {content.video_path}")
        media_info = f" with {' + '.join(media_parts)}" if media_parts else ""
        print(f"[DRY RUN] Would post{media_info}:\n{content.text}\n")
        if has_image and has_video:
            print("[DRY RUN] Would reply with video clip\n")
        if content.reply:
            print(f"[DRY RUN] Would reply (after 5 min):\n{content.reply.text}\n")
        tweet_id = "dry-run-0"
    elif has_image:
        tweet_id = poster.post_with_image(
            content.text, content.image_path, content.alt_text
        )
        # If we also have a video, reply to the main tweet with it
        # Combine with analysis text if available (spark notes)
        if has_video:
            try:
                reply_text = ""
                if content.reply and content.reply.text:
                    reply_text = content.reply.text
                    content.reply = None  # consumed — don't post again below
                vid_id = poster.post_video_reply(
                    tweet_id, content.video_path, text=reply_text
                )
                log.info("Posted video reply — tweet %s", vid_id)
            except Exception:
                log.warning("Video reply failed", exc_info=True)
    elif has_video:
        tweet_id = poster.post_with_video(content.text, content.video_path)
    else:
        tweet_id = poster.post_text(content.text)

    record_post(generator.name, tweet_id, content.tags)
    log.info("Done — tweet %s", tweet_id)

    # Handle reply (delayed for reveals, immediate for analysis)
    if content.reply and not DRY_RUN:
        is_reveal = "analysis" not in (content.reply.tags or [])
        if is_reveal:
            log.info("Waiting 5 minutes before posting reply...")
            await asyncio.sleep(300)
        reply = content.reply
        try:
            reply_id = poster.post_reply(
                reply.text,
                in_reply_to=tweet_id,
                image_path=reply.image_path,
                alt_text=reply.alt_text,
            )
            log.info("Posted reply — tweet %s", reply_id)
        except Exception:
            log.warning("Reply failed, posting as standalone tweet", exc_info=True)
            reply_id = poster.post_text(reply.text)
            log.info("Posted reveal as standalone tweet %s", reply_id)

    # Handle replies list (thread posting — e.g. Reds summary)
    if content.replies:
        reply_to_id = tweet_id
        for i, reply in enumerate(content.replies):
            if DRY_RUN:
                media_parts = []
                if reply.image_path and reply.image_path.exists():
                    media_parts.append(f"image {reply.image_path}")
                if reply.video_path and reply.video_path.exists():
                    media_parts.append(f"video {reply.video_path}")
                media_info = f" with {' + '.join(media_parts)}" if media_parts else ""
                print(f"[DRY RUN] Would reply #{i+1}{media_info}:\n{reply.text}\n")
                continue

            try:
                has_reply_image = reply.image_path and reply.image_path.exists()
                has_reply_video = reply.video_path and reply.video_path.exists()

                if has_reply_video:
                    # Video reply
                    rid = poster.post_video_reply(
                        in_reply_to=reply_to_id,
                        video_path=reply.video_path,
                        text=reply.text,
                    )
                elif has_reply_image:
                    # Image reply
                    rid = poster.post_reply(
                        text=reply.text,
                        in_reply_to=reply_to_id,
                        image_path=reply.image_path,
                        alt_text=reply.alt_text,
                    )
                else:
                    # Text-only reply
                    rid = poster.post_reply(
                        text=reply.text,
                        in_reply_to=reply_to_id,
                    )
                log.info("Posted thread reply #%d — tweet %s", i + 1, rid)
                reply_to_id = rid  # chain replies
            except Exception:
                log.warning("Thread reply #%d failed", i + 1, exc_info=True)


def main() -> None:
    global DRY_RUN

    parser = argparse.ArgumentParser(description="TJStats Baseball X Bot")
    parser.add_argument(
        "--slot",
        choices=["screenshot", "text", "evening", "biomechanics", "daily", "all"],
        default="all",
        help="Which content slot to run (default: all)",
    )
    parser.add_argument(
        "--generator",
        choices=list(GENERATORS.keys()),
        default=None,
        help="Run a specific generator by name (overrides --slot)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate content but don't post to X",
    )
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    asyncio.run(run_post(args.slot, args.generator))


if __name__ == "__main__":
    main()
