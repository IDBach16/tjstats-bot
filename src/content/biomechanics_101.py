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
- Do NOT use the word "elite" — use more creative language
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


def _generate_deep_dive(topic: dict, stats: dict) -> str | None:
    """Generate a deeper follow-up reply with training implications and context."""
    client = _get_client()
    if not client:
        return None

    stats_lines = []
    stats_lines.append(f"Dataset: {stats.get('n_pitches', '?')} fastballs "
                       f"from {stats.get('n_pitchers', '?')} pitchers "
                       f"(mostly college level)")

    if topic["chart_type"] == "scatter":
        stats_lines.append(
            f"X ({topic['x_label']}): mean={stats.get('x_mean', '?'):.1f}, "
            f"range=[{stats.get('x_min', '?'):.1f}, {stats.get('x_max', '?'):.1f}]"
        )
        if "y_mean" in stats:
            stats_lines.append(f"Y ({topic['y_label']}): mean={stats['y_mean']:.1f}")
        if "correlation" in stats:
            stats_lines.append(f"Correlation (r): {stats['correlation']:.2f}")
    else:
        stats_lines.append(
            f"{topic['x_label']}: mean={stats.get('x_mean', '?'):.1f}, "
            f"median={stats.get('x_median', '?'):.1f}, "
            f"10th %ile={stats.get('x_p10', '?'):.1f}, "
            f"90th %ile={stats.get('x_p90', '?'):.1f}"
        )

    stats_text = "\n".join(stats_lines)

    prompt = f"""You are a baseball biomechanics expert writing a deep-dive thread reply on Twitter/X. This is a follow-up to a chart post about {topic['prompt_context']}.

Chart: {topic['title']}
Background: {topic['education']}

Data Summary:
{stats_text}

Write a 2-tweet thread (each tweet UNDER 275 chars, separated by ---).

Tweet 1: Explain WHY this matters for player development. What should a pitcher or pitching coach take away from this data? Reference specific numbers (percentiles, averages). Think like a pitching coordinator explaining this to their staff.

Tweet 2: Give 1-2 actionable training cues or drills that target this mechanic. Be specific — name real exercises, constraints, or movement patterns that coaches actually use (e.g. rocker drills, pivot pickoffs, hip lead wall drills, PlyoCare, connection ball). End with a practical takeaway.

Rules:
- Sound like a top-tier pitching development coach, not a professor
- Do NOT use the word "elite" — find more creative descriptors
- Reference specific data points from the stats
- Do NOT use hashtags, emojis, or @ mentions
- Do NOT use dashes or hyphens (use commas, periods, or other punctuation instead)
- Each tweet must be UNDER 275 characters
- Separate the two tweets with ---
"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        parts = [p.strip() for p in text.split("---") if p.strip()]
        # Cap each part
        capped = []
        for p in parts[:2]:
            if len(p) > 280:
                p = p[:277] + "..."
            capped.append(p)
        log.info("Generated biomech deep dive (%d parts)", len(capped))
        return capped
    except Exception:
        log.warning("Biomech deep dive generation failed", exc_info=True)
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

        # Generate deep dive thread
        deep_dive = _generate_deep_dive(topic, stats)

        # Main tweet text
        text = (
            f"Biomechanics 101: {topic['title']}"
            f"\n\n@TJStats {DEFAULT_HASHTAGS} #Biomechanics"
        )

        # Build reply chain: explanation → deep dive part 1 → deep dive part 2
        reply_chain = None
        if explanation:
            # Start with the explanation reply
            reply_chain = PostContent(
                text=explanation,
                tags=["biomech_explanation"],
            )
            # Append deep dive parts as nested replies
            if deep_dive:
                current = reply_chain
                for i, part in enumerate(deep_dive):
                    next_reply = PostContent(
                        text=part,
                        tags=[f"biomech_deep_dive_{i+1}"],
                    )
                    current.reply = next_reply
                    current = next_reply
        elif deep_dive:
            # No explanation but have deep dive — lead with first part
            reply_chain = PostContent(
                text=deep_dive[0],
                tags=["biomech_deep_dive_1"],
            )
            if len(deep_dive) > 1:
                reply_chain.reply = PostContent(
                    text=deep_dive[1],
                    tags=["biomech_deep_dive_2"],
                )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Biomechanics chart: {topic['title']}",
            tags=["biomechanics_101", f"biomech_{topic['id']}"],
            reply=reply_chain,
        )
