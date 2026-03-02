"""Entry point / orchestrator for the TJStats Bot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from . import poster
from .scheduler import get_generators_for_today, record_post

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

DRY_RUN = False


async def run_post(slot: str) -> None:
    """Generate and post content for a given slot ('screenshot' or 'text')."""
    ss_gen, txt_gen = get_generators_for_today()

    if slot == "screenshot":
        await _generate_and_post(ss_gen)
    elif slot == "text":
        await _generate_and_post(txt_gen)
    else:
        for gen in (ss_gen, txt_gen):
            await _generate_and_post(gen)


async def _generate_and_post(generator) -> None:
    log.info("Running generator: %s", generator.name)
    content = await generator.generate()

    if not content.text:
        log.warning("Generator %s produced empty content — skipping", generator.name)
        return

    log.info("Posting: %s", content.text[:80])

    if DRY_RUN:
        img_info = f" with image {content.image_path}" if content.image_path else ""
        print(f"[DRY RUN] Would post{img_info}:\n{content.text}\n")
        tweet_id = "dry-run-0"
    elif content.image_path and content.image_path.exists():
        tweet_id = poster.post_with_image(
            content.text, content.image_path, content.alt_text
        )
    else:
        tweet_id = poster.post_text(content.text)

    record_post(generator.name, tweet_id, content.tags)
    log.info("Done — tweet %s", tweet_id)


def main() -> None:
    global DRY_RUN

    parser = argparse.ArgumentParser(description="TJStats Baseball X Bot")
    parser.add_argument(
        "--slot",
        choices=["screenshot", "text", "both"],
        default="both",
        help="Which content slot to run (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate content but don't post to X",
    )
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    asyncio.run(run_post(args.slot))


if __name__ == "__main__":
    main()
