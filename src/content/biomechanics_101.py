"""Content generator: Biomechanics 101 educational posts.

Uses Driveline Open Biomechanics data to create scatter/distribution
charts with AI-generated plain-English explanations.
"""

from __future__ import annotations

import logging
import os

import anthropic

from .base import ContentGenerator, PostContent
from ..biomechanics import load_merged, pick_topic, compute_topic_stats, TOPICS
from ..charts import plot_biomechanics
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=key)
    return _client


def _generate_explanation(topic: dict, stats: dict) -> str | None:
    """Generate a plain-English explanation of the biomechanics chart."""
    client = _get_client()
    if not client:
        return None

    # Build stats context
    stats_lines = []
    stats_lines.append(f"Dataset: {stats.get('n_pitches', '?')} fastballs "
                       f"from {stats.get('n_pitchers', '?')} pitchers "
                       f"(mostly college level)")

    if topic["chart_type"] == "scatter":
        stats_lines.append(
            f"X-axis ({topic['x_label']}): "
            f"mean={stats.get('x_mean', '?'):.1f}, "
            f"range=[{stats.get('x_min', '?'):.1f}, {stats.get('x_max', '?'):.1f}]"
        )
        if "y_mean" in stats:
            stats_lines.append(
                f"Y-axis ({topic['y_label']}): "
                f"mean={stats['y_mean']:.1f}"
            )
        if "correlation" in stats:
            stats_lines.append(f"Correlation (r): {stats['correlation']:.2f}")
    else:
        stats_lines.append(
            f"{topic['x_label']}: "
            f"mean={stats.get('x_mean', '?'):.1f}, "
            f"median={stats.get('x_median', '?'):.1f}, "
            f"10th %ile={stats.get('x_p10', '?'):.1f}, "
            f"90th %ile={stats.get('x_p90', '?'):.1f}"
        )

    stats_text = "\n".join(stats_lines)

    prompt = f"""You are a baseball biomechanics educator writing for Twitter/X. Given this chart data, write a concise educational explanation.

Chart: {topic['title']}
Type: {topic['chart_type']}
Topic context: {topic['prompt_context']}
Background: {topic['education']}

Data Summary:
{stats_text}

Rules:
- Write 2-3 sentences that explain what the chart shows and why it matters
- Make it accessible — explain the biomechanics concept for someone who may not know what it is
- Reference specific numbers from the data to make it concrete
- Sound like a knowledgeable coach/trainer, not a textbook
- Keep it UNDER 260 characters total (strict Twitter limit)
- Do NOT use hashtags, emojis, or @ mentions
- Do NOT use dashes or hyphens (use commas, periods, or other punctuation instead)
- Start with a hook or key takeaway, not "This chart shows..."
"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        if len(text) > 270:
            text = text[:267] + "..."
        log.info("Generated biomech explanation (%d chars): %s",
                 len(text), text[:80])
        return text
    except Exception:
        log.warning("Biomech explanation generation failed", exc_info=True)
        return None


class BiomechanicsGenerator(ContentGenerator):
    name = "biomechanics_101"

    async def generate(self) -> PostContent:
        df = load_merged()
        if df.empty:
            log.warning("No biomechanics data available")
            return PostContent(text="")

        # Pick a topic (avoid recent ones via post history)
        from ..scheduler import was_recently_posted
        recent = [t["id"] for t in TOPICS
                  if was_recently_posted(f"biomech_{t['id']}", lookback=10)]
        topic = pick_topic(recent_ids=recent)

        stats = compute_topic_stats(topic, df)

        # Generate chart
        image_path = plot_biomechanics(topic, df, stats)
        if not image_path:
            log.warning("Biomech chart failed for %s", topic["id"])
            return PostContent(text="")

        # Generate AI explanation
        explanation = _generate_explanation(topic, stats)

        # Main tweet text
        text = (
            f"Biomechanics 101: {topic['title']}"
            f"\n\n@TJStats {DEFAULT_HASHTAGS} #Biomechanics"
        )

        # Explanation as reply
        reply_content = None
        if explanation:
            reply_content = PostContent(
                text=explanation,
                tags=["biomech_explanation"],
            )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Biomechanics chart: {topic['title']}",
            tags=["biomechanics_101", f"biomech_{topic['id']}"],
            reply=reply_content,
        )
