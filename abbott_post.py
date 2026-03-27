"""Post Andrew Abbott Opening Day pitching summary thread to X."""

import os
import sys
import logging
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, r"C:\Users\IDBac\tjstats-bot")

from dotenv import load_dotenv
load_dotenv(r"C:\Users\IDBac\tjstats-bot\.env")

from src.charts import plot_pitching_summary
from src.video_clips import get_pitcher_clip, _download_mp4
from src.pitch_profiler import get_season_pitchers, get_season_pitches
from src.config import MLB_SEASON, CLIPS_DIR
import anthropic
import tweepy

ABBOTT_ID = 671096
NAME = "Andrew Abbott"
TEAM = "CIN"

# Savant playIds for strikeout pitches
PLAY_IDS = {
    "FF": "1550dfcd-4078-3e67-a36a-69c5ac19c237",
    "CH": "9a0437f3-81f4-37f2-88f3-80d74e6a38f2",
    "CU": "fca17e66-49c5-39d8-9d6d-c8a6a240e4b7",
    "ST": "4f25ffb2-0897-32d9-8e20-84448a322143",
    "FC": "f6afc3db-340b-31a4-ae66-96dd84dad3c3",
}


def download_savant_clip(play_id, pitch_type):
    """Download a Savant sporty-video clip as mp4."""
    import re
    out_path = CLIPS_DIR / f"abbott_{pitch_type}.mp4"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  {pitch_type}: cached ({out_path.stat().st_size // 1024}KB)")
        return out_path

    # Fetch the sporty-videos HTML page and extract mp4 URL
    url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    try:
        resp = requests.get(url, timeout=15)
        html = resp.text
        mp4_urls = re.findall(r'https?://sporty-clips\.mlb\.com/[^\s"<>]+\.mp4', html)
        if not mp4_urls:
            print(f"  {pitch_type}: no mp4 URL found in page")
            return None

        mp4_url = mp4_urls[0]
        if _download_mp4(mp4_url, out_path):
            print(f"  {pitch_type}: downloaded ({out_path.stat().st_size // 1024}KB)")
            return out_path
    except Exception as e:
        print(f"  {pitch_type}: error: {e}")

    return None


def generate_analysis():
    """Generate Abbott-specific analysis without character count in output."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None

    client = anthropic.Anthropic(api_key=key)

    prompt = """Write exactly 2 sentences analyzing Andrew Abbott (Reds, LHP) as the 2026 Opening Day starter.

Stats: 10-7, 2.87 ERA, 3.66 FIP, 1.15 WHIP, 149 K, 166.1 IP, 21.8% K rate, 6.3% BB rate, 24% whiff rate.
Stuff+: 100. 93rd percentile hard-hit suppression. 3rd percentile ground ball rate.
Sweeper has 34% whiff rate. Changeup 25% whiff.

Rules:
- Exactly 2 sentences, nothing more
- Keep it under 240 characters total
- Be sharp and insightful
- Do NOT use "elite"
- Reference 2-3 specific stats
- No hashtags, emojis, @ mentions, dashes, or hyphens
- Do NOT start with his name
- Output ONLY the 2 sentences, absolutely nothing else"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        # Remove any trailing character count like "(259 characters)"
        import re
        text = re.sub(r'\s*\(\d+ characters?\)\s*$', '', text)
        text = re.sub(r'\s*\d+ characters?\s*$', '', text)
        return text
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None


def main():
    print("Pulling Pitch Profiler data...")
    season_df = get_season_pitchers(2025)
    pitches_df = get_season_pitches(2025)

    print("Generating pitching summary card...")
    image_path = plot_pitching_summary(
        NAME, season_df, pitches_df,
        team=TEAM, player_id=ABBOTT_ID, level="MLB",
    )
    if not image_path:
        print("Card generation failed!")
        return
    print(f"Card: {image_path}")

    print("Generating analysis...")
    analysis = generate_analysis()
    if analysis:
        print(f"Analysis: {analysis}")

    print("Downloading Savant video clips...")
    pitch_clips = {}
    for pt, play_id in PLAY_IDS.items():
        clip = download_savant_clip(play_id, pt)
        if clip:
            pitch_clips[pt] = clip

    print("Fetching Film Room clip...")
    film_clip = get_pitcher_clip(ABBOTT_ID, NAME)
    if not film_clip:
        existing = CLIPS_DIR / "andrew_abbott_2025.mp4"
        if existing.exists():
            film_clip = existing

    # ── Post thread ──
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

    def post_with_video(text, reply_to, video_path=None):
        """Post tweet, optionally with video."""
        media_ids = None
        if video_path and video_path.exists():
            try:
                vid = v1.chunked_upload(filename=str(video_path), media_category="tweet_video")
                media_ids = [vid.media_id]
            except Exception as e:
                print(f"  Video upload failed: {e}")
        kwargs = {"text": text, "in_reply_to_tweet_id": reply_to}
        if media_ids:
            kwargs["media_ids"] = media_ids
        resp = client.create_tweet(**kwargs)
        return resp.data["id"]

    # Tweet 1: Card + analysis
    tweet1 = (
        f"{analysis}\n\n" if analysis else ""
    ) + (
        f"{NAME}'s {MLB_SEASON} Pitching Summary\n"
        f"2026 Opening Day Starter\n\n"
        f"@TJStats #MLB #Statcast #Reds #OpeningDay"
    )
    media = v1.media_upload(filename=str(image_path))
    v1.create_media_metadata(media.media_id, alt_text=f"Pitching summary for {NAME}")
    resp = client.create_tweet(text=tweet1, media_ids=[media.media_id])
    t1 = resp.data["id"]
    print(f"Tweet 1 (card): {t1}")

    # Tweet 2: Fastball + video
    t2 = post_with_video(
        "4-Seam Fastball | 47% usage | 92.8 mph\n\n"
        "Not a blow-it-by-you pitch at 26th percentile velo. "
        "Leads all qualified pitchers with 31% opposite-field contact rate. "
        "Moved outer-half usage from 48% to 53%. "
        "Hitters simply cannot pull him.",
        t1, pitch_clips.get("FF")
    )
    print(f"Tweet 2 (FF): {t2}")

    # Tweet 3: Changeup + video
    t3 = post_with_video(
        "Changeup | 20% usage | 84.8 mph\n\n"
        "8 mph velo gap from the fastball with identical arm speed. "
        "15\" of arm-side run. 25% whiff rate. "
        "6\" horizontal separation tunnels perfectly off the heater. "
        "Primary weapon against righties.",
        t2, pitch_clips.get("CH")
    )
    print(f"Tweet 3 (CH): {t3}")

    # Tweet 4: Curveball + video
    t4 = post_with_video(
        "Curveball | 15% usage | 81 mph\n\n"
        "Classic 12-6 shape. 45\" of vertical drop. "
        "2800 RPM spin (80th percentile). "
        "Buries it below the zone for chase swings. "
        "Same speed as the sweeper but breaks straight down.",
        t3, pitch_clips.get("CU")
    )
    print(f"Tweet 4 (CU): {t4}")

    # Tweet 5: Sweeper + video
    t5 = post_with_video(
        "Sweeper | 14% usage | 82.8 mph\n\n"
        "The put-away pitch. 34% whiff rate, highest on the staff. "
        "11\" of horizontal sweep. Same speed as the curve but breaks sideways. "
        "Hitters can't distinguish them out of the hand.",
        t4, pitch_clips.get("ST")
    )
    print(f"Tweet 5 (ST): {t5}")

    # Tweet 6: Cutter + video
    t6 = post_with_video(
        "Cutter | 4% usage | 88.6 mph\n\n"
        "Sparingly used but bridges the gap between fastball and breaking stuff. "
        "Runs in on righties' hands. 22.5% whiff rate.",
        t5, pitch_clips.get("FC")
    )
    print(f"Tweet 6 (FC): {t6}")

    # Tweet 7: Film Room clip
    last = t6
    if film_clip and film_clip.exists():
        t7 = post_with_video("Abbott in action", t6, film_clip)
        last = t7
        print(f"Tweet 7 (film room): {t7}")

    # Tweet 8: Summary
    resp = client.create_tweet(
        text=(
            "The concerns:\n\n"
            "26th percentile fastball velo\n"
            "3rd percentile ground ball rate\n"
            "FIP (3.66) and xFIP (4.31) suggest some regression\n"
            "Career second-half struggles\n\n"
            "But 93rd percentile hard-hit suppression is real. "
            "Abbott doesn't overpower you. He outsmarts you."
        ),
        in_reply_to_tweet_id=last
    )
    print(f"Tweet 8 (summary): {resp.data['id']}")

    print(f"\nThread: https://x.com/BachTalk1/status/{t1}")


if __name__ == "__main__":
    main()
