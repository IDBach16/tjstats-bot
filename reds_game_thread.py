"""Post Reds game pitching thread — all pitchers from a game with cards + video."""

import os
import sys
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, r"C:\Users\IDBac\tjstats-bot")

from dotenv import load_dotenv
load_dotenv(r"C:\Users\IDBac\tjstats-bot\.env")

from pybaseball import statcast
from src.charts import plot_pitching_summary
from src.pitch_profiler import get_season_pitchers, get_season_pitches
from src.video_clips import _download_mp4
from src.config import CLIPS_DIR
import requests
import anthropic
import tweepy
import pandas as pd

TEAM = "CIN"
TEAM_FULL = "Cincinnati Reds"


def find_reds_game(game_date):
    """Find the Reds game on a given date and return all Reds pitcher data."""
    print(f"Pulling Statcast for {game_date}...")
    df = statcast(game_date, game_date)
    if df is None or df.empty:
        print("No Statcast data for this date")
        return None

    # Find Reds game
    reds_games = df[(df["home_team"] == "CIN") | (df["away_team"] == "CIN")]
    if reds_games.empty:
        print("No Reds game found")
        return None

    game_pk = int(reds_games["game_pk"].iloc[0])
    game = df[df["game_pk"] == game_pk]
    home = game["home_team"].iloc[0]
    away = game["away_team"].iloc[0]
    opponent = away if home == "CIN" else home

    # Reds pitch when opponents bat (Top if CIN is home, Bot if away)
    if home == "CIN":
        reds_pitches = game[game["inning_topbot"] == "Top"]
    else:
        reds_pitches = game[game["inning_topbot"] == "Bot"]

    # Get pitcher stats
    pitchers = []
    for pid, grp in reds_pitches.groupby("pitcher"):
        name = str(grp["player_name"].iloc[0])
        if "," in name:
            parts = name.split(", ")
            name = f"{parts[1]} {parts[0]}"

        ks = len(grp[grp["events"] == "strikeout"])
        bbs = len(grp[grp["events"] == "walk"])
        hits = len(grp[grp["events"].isin(["single", "double", "triple", "home_run"])])
        ers = len(grp[grp["events"] == "home_run"])  # rough proxy

        pitchers.append({
            "name": name, "id": int(pid),
            "pitches": len(grp), "ks": ks, "bbs": bbs, "hits": hits,
        })

    pitchers.sort(key=lambda x: -x["pitches"])

    # Get video playIds from MLB play-by-play
    print("Getting video playIds...")
    pbp = requests.get(
        f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay", timeout=15
    ).json()

    for p in pitchers:
        play_id = None
        for play in pbp.get("allPlays", []):
            if play.get("matchup", {}).get("pitcher", {}).get("id") != p["id"]:
                continue
            event = (play.get("result", {}).get("event") or "").lower()
            if "strikeout" in event:
                for pe in reversed(play.get("playEvents", [])):
                    if pe.get("playId"):
                        play_id = pe["playId"]
                        break
                if play_id:
                    break

        p["play_id"] = play_id
        p["video_path"] = None

        if play_id:
            clip_path = CLIPS_DIR / f"reds_{game_date}_{p['name'].replace(' ', '_').lower()}.mp4"
            if clip_path.exists() and clip_path.stat().st_size > 1000:
                p["video_path"] = clip_path
            else:
                try:
                    surl = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
                    resp = requests.get(surl, timeout=10)
                    mp4s = re.findall(r'https?://sporty-clips\.mlb\.com/[^\s"<>]+\.mp4', resp.text)
                    if mp4s and _download_mp4(mp4s[0], clip_path):
                        p["video_path"] = clip_path
                except Exception as e:
                    print(f"  Video download failed for {p['name']}: {e}")

    return {
        "game_pk": game_pk,
        "date": game_date,
        "home": home,
        "away": away,
        "opponent": opponent,
        "pitchers": pitchers,
    }


def generate_outing_summary(name, stats_str):
    """Generate 2-sentence summary."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    client = anthropic.Anthropic(api_key=key)
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content":
                f"Write exactly 2 sentences summarizing {name}'s pitching outing. "
                f"Stats: {stats_str}. "
                f"Be direct and factual. Reference specific stats. "
                f"No hashtags, emojis, @, dashes, hyphens, or character counts. "
                f"Output ONLY the 2 sentences."
            }],
        )
        text = message.content[0].text.strip()
        text = re.sub(r'\s*\(\d+ characters?\)\s*$', '', text)
        return text
    except:
        return None


def main(game_date="2026-03-26"):
    game = find_reds_game(game_date)
    if not game:
        return

    pitchers = game["pitchers"]
    opponent = game["opponent"]
    date_display = pd.to_datetime(game_date).strftime("%B %d, %Y")
    starter = pitchers[0]["name"]  # Most pitches = starter

    print(f"\nCIN vs {opponent} | {date_display}")
    print(f"Starter: {starter}")
    print(f"Pitchers: {len(pitchers)}")

    # Pull Pitch Profiler data for cards
    print("\nPulling Pitch Profiler 2026 data...")
    season_df = get_season_pitchers(2026)
    pitches_df = get_season_pitches(2026)

    # Setup Twitter
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

    def post_tweet(text, reply_to=None, image_path=None, video_path=None):
        media_ids = []
        if image_path:
            m = v1.media_upload(filename=str(image_path))
            v1.create_media_metadata(m.media_id, alt_text="Pitching summary")
            media_ids.append(m.media_id)
        if video_path and video_path.exists():
            try:
                vm = v1.chunked_upload(filename=str(video_path), media_category="tweet_video")
                media_ids.append(vm.media_id)
            except Exception as e:
                print(f"  Video upload failed: {e}")

        kwargs = {"text": text}
        if reply_to:
            kwargs["in_reply_to_tweet_id"] = reply_to
        if media_ids:
            kwargs["media_ids"] = media_ids
        resp = client.create_tweet(**kwargs)
        return resp.data["id"]

    # ── Tweet 1: Header ──
    header = (
        f"Reds Pitching Summary Thread\n"
        f"CIN vs {opponent} | {date_display}\n\n"
        f"Starter: {starter}\n"
        f"{len(pitchers)} pitchers used\n\n"
        f"Full breakdown of every arm used below\n\n"
        f"@TJStats #MLB #Statcast #Reds"
    )
    t1 = post_tweet(header)
    print(f"\nHeader: {t1}")
    last_id = t1

    # ── Post each pitcher ──
    for i, p in enumerate(pitchers):
        name = p["name"]
        print(f"\n--- {name} ({p['pitches']} pitches) ---")

        # Generate card if pitcher is in Pitch Profiler data
        image_path = None
        if name in season_df["pitcher_name"].values:
            print("  Generating card...")
            image_path = plot_pitching_summary(
                name, season_df, pitches_df,
                team=TEAM, player_id=p["id"], level="MLB",
            )
            if image_path:
                print(f"  Card: {image_path}")

        # Build tweet text
        role = "Starter" if i == 0 else "Reliever"
        line = f"{p['pitches']} pitches, {p['ks']} K, {p['bbs']} BB, {p['hits']} H"

        # Generate 2-sentence summary
        summary = generate_outing_summary(name, line)

        if summary:
            tweet_text = f"{name} | {role}\n{line}\n\n{summary}"
        else:
            tweet_text = f"{name} | {role}\n{line}"

        # Post with card and/or video
        if image_path and p.get("video_path"):
            # Post card first, then video reply
            tid = post_tweet(tweet_text, reply_to=last_id, image_path=image_path)
            print(f"  Card tweet: {tid}")
            # Video reply
            vid_tid = post_tweet(f"{name}", reply_to=tid, video_path=p["video_path"])
            print(f"  Video tweet: {vid_tid}")
            last_id = vid_tid
        elif image_path:
            tid = post_tweet(tweet_text, reply_to=last_id, image_path=image_path)
            print(f"  Card tweet: {tid}")
            last_id = tid
        elif p.get("video_path"):
            tid = post_tweet(tweet_text, reply_to=last_id, video_path=p["video_path"])
            print(f"  Video tweet: {tid}")
            last_id = tid
        else:
            tid = post_tweet(tweet_text, reply_to=last_id)
            print(f"  Text tweet: {tid}")
            last_id = tid

    print(f"\nThread: https://x.com/BachTalk1/status/{t1}")


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-03-26"
    main(date)
