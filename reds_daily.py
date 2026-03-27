"""Post Reds pitcher game summary from Pitch Profiler 2026 data."""

import os
import sys
import re
import logging

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, r"C:\Users\IDBac\tjstats-bot")

from dotenv import load_dotenv
load_dotenv(r"C:\Users\IDBac\tjstats-bot\.env")

from src.charts import plot_pitching_summary
from src.pitch_profiler import get_season_pitchers, get_season_pitches
from src.config import CLIPS_DIR
import anthropic
import tweepy

TEAM = "CIN"


def generate_outing_summary(name, stats):
    """Generate 2-sentence summary of the pitcher's outing."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None

    client = anthropic.Anthropic(api_key=key)
    prompt = f"""Write exactly 2 sentences summarizing {name}'s pitching outing.

Stats: {stats}

Rules:
- Exactly 2 sentences, no more
- Be direct and factual
- Reference specific stats from the outing
- No hashtags, emojis, @ mentions, dashes, or hyphens
- Do NOT include character counts
- Output ONLY the 2 sentences"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        text = re.sub(r'\s*\(\d+ characters?\)\s*$', '', text)
        return text
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None


def main(pitcher_name=None):
    print("Pulling 2026 Pitch Profiler data...")
    season_df = get_season_pitchers(2026)
    pitches_df = get_season_pitches(2026)
    print(f"  {len(season_df)} pitchers, {len(pitches_df)} pitch rows")

    # If no pitcher specified, find the most recent Reds starter
    if not pitcher_name:
        # Try to find Reds pitchers (no team column directly, check by known names or all)
        print("No pitcher specified, looking for Reds pitchers...")
        # For now require a name
        print("Usage: python reds_daily.py 'Andrew Abbott'")
        return

    # Find the pitcher
    matches = season_df[season_df["pitcher_name"] == pitcher_name]
    if matches.empty:
        print(f"'{pitcher_name}' not found in 2026 data")
        return

    p = matches.iloc[0]
    player_id = int(p.get("pitcher_id", 0)) if p.get("pitcher_id") else None

    # Build stats string for summary
    ip = p.get("innings_pitched", "?")
    er = round(float(p.get("era", 0)) * float(ip) / 9, 0) if ip != "?" else "?"
    stats_str = (
        f"{ip} IP, {int(er) if er != '?' else '?'} ER, "
        f"{int(p.get('strike_outs', 0))} K, {int(p.get('walks', 0))} BB, "
        f"{int(p.get('hits', 0))} H, "
        f"{p.get('era', '?')} ERA, {p.get('fip', '?')} FIP, "
        f"{float(p.get('whiff_rate', 0))*100:.1f}% whiff rate, "
        f"{float(p.get('ground_ball_percentage', 0))*100:.1f}% GB rate"
    )
    print(f"Stats: {stats_str}")

    # Generate card
    print("Generating pitching summary card...")
    image_path = plot_pitching_summary(
        pitcher_name, season_df, pitches_df,
        team=TEAM, player_id=player_id, level="MLB",
    )
    if not image_path:
        print("Card generation failed!")
        return
    print(f"Card: {image_path}")

    # Generate 2-sentence summary
    print("Generating outing summary...")
    summary = generate_outing_summary(pitcher_name, stats_str)
    if summary:
        print(f"Summary: {summary}")

    # Post
    auth = tweepy.OAuth1UserHandler(
        os.environ["X_API_KEY"], os.environ["X_API_SECRET"],
        os.environ["X_ACCESS_TOKEN"], os.environ["X_ACCESS_TOKEN_SECRET"],
    )
    v1 = tweepy.API(auth)
    client = tweepy.Client(
        bearer_token=os.environ["X_BEARER_TOKEN"],
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"],
    )

    tweet_text = (
        f"{summary}\n\n" if summary else ""
    ) + (
        f"{pitcher_name}'s 2026 Pitching Summary\n\n"
        f"@TJStats #MLB #Statcast #Reds"
    )

    media = v1.media_upload(filename=str(image_path))
    v1.create_media_metadata(media.media_id, alt_text=f"Pitching summary for {pitcher_name}")
    resp = client.create_tweet(text=tweet_text, media_ids=[media.media_id])
    tweet_id = resp.data["id"]

    print(f"\nPosted: https://x.com/BachTalk1/status/{tweet_id}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Andrew Abbott"
    main(name)
