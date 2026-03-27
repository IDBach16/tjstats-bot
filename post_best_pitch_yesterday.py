"""Post best pitch from yesterday to X."""
import sys, os, re, logging
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, r"C:\Users\IDBac\tjstats-bot")
os.chdir(r"C:\Users\IDBac\tjstats-bot")

from dotenv import load_dotenv
load_dotenv()

from src import pitch_profiler
from src.config import MLB_SEASON, MLB_API_BASE, SCREENSHOTS_DIR
from src.charts import plot_best_pitch_card
from src.video_clips import _download_mp4
import requests, pandas as pd, tweepy, numpy as np, re
from pathlib import Path
from datetime import date, timedelta

PITCH_NAMES = {"FF": "4-Seam Fastball", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "SV": "Sweeper", "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle Curve",
    "CH": "Changeup", "FS": "Splitter", "CT": "Cutter"}

yesterday = date.today() - timedelta(days=1)
display_date = yesterday.strftime("%m/%d/%Y")

# Get games
url = f"{MLB_API_BASE}/schedule?sportId=1&date={yesterday.strftime('%Y-%m-%d')}"
resp = requests.get(url, timeout=15)
pks = [g["gamePk"] for d in resp.json().get("dates", []) for g in d.get("games", [])
       if "Final" in g.get("status", {}).get("detailedState", "")]
print(f"Yesterday: {len(pks)} games")

gpp = pitch_profiler.get_game_pitches(MLB_SEASON)
yest = gpp[(gpp["game_type"] == "R") & (gpp["game_pk"].isin(pks))]
yest = yest.copy()
yest["thrown"] = pd.to_numeric(yest["thrown"], errors="coerce")
qual = yest[yest["thrown"] >= 10].copy()
for col in ["stuff_plus", "whiff_rate", "called_strikes_plus_whiffs_percentage",
            "velocity", "spin_rate", "ivb", "hb", "chase_percentage",
            "zone_percentage", "woba", "run_value_per_100_pitches"]:
    if col in qual.columns:
        qual[col] = pd.to_numeric(qual[col], errors="coerce")

qual["score"] = (qual["stuff_plus"].fillna(100) * 0.4
    + qual["whiff_rate"].fillna(0) * 100 * 0.3
    + qual["called_strikes_plus_whiffs_percentage"].fillna(0) * 100 * 0.3)

best = qual.sort_values("score", ascending=False).iloc[0]
name = str(best["pitcher_name"])
pt = str(best["pitch_type"])
pid = int(best["pitcher_id"])
gpk = int(best["game_pk"])
pitch_display = PITCH_NAMES.get(pt, pt)

print(f"Best: {name} {pitch_display} stuff+={best.get('stuff_plus', 0):.0f} whiff={float(best.get('whiff_rate', 0)) * 100:.1f}%")

# Get team from Statcast game data or MLB API
team = ""
try:
    # Try from the game data first
    game_pitches = gpp[(gpp['game_type']=='R') & (gpp['game_pk']==gpk)]
    pitcher_rows = game_pitches[game_pitches['pitcher_name']==name]
    # Fall back to MLB API
    mlb_resp = requests.get(f'https://statsapi.mlb.com/api/v1/people/{pid}?hydrate=currentTeam', timeout=10)
    mlb_data = mlb_resp.json()
    team_name = mlb_data['people'][0].get('currentTeam',{}).get('name','')
    # Map full name to abbreviation
    _NAME_TO_ABBREV = {
        "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
        "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
        "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
        "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
        "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
        "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
        "New York Yankees":"NYY","Oakland Athletics":"OAK","Philadelphia Phillies":"PHI",
        "Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF",
        "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TB",
        "Texas Rangers":"TEX","Toronto Blue Jays":"TOR","Washington Nationals":"WSH",
    }
    team = _NAME_TO_ABBREV.get(team_name, "")
    print(f"Team: {team_name} -> {team}")
except Exception as e:
    print(f"Team lookup error: {e}")

# Build card
out = plot_best_pitch_card(
    pitcher_name=name, pitch_type=pt, team=team, player_id=pid,
    game_date=display_date, pitch_data=best.to_dict(),
    title="Best Pitch Last Night",
)
if not out:
    print("Card generation failed!")
    sys.exit(1)
out = str(out)
print(f"Card: {out}")

# Get video
video_path = None
try:
    pbp = requests.get(f"https://statsapi.mlb.com/api/v1/game/{gpk}/playByPlay", timeout=15).json()
    for play in pbp.get("allPlays", []):
        if play.get("matchup", {}).get("pitcher", {}).get("id") != pid:
            continue
        if "strikeout" not in (play.get("result", {}).get("event", "")).lower():
            continue
        for pe in reversed(play.get("playEvents", [])):
            play_id = pe.get("playId")
            if play_id:
                sr = requests.get(f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}", timeout=10)
                mp4s = re.findall(r'https?://sporty-clips\.mlb\.com/[^\s"<>]+\.mp4', sr.text)
                if mp4s:
                    vp = Path(SCREENSHOTS_DIR).parent / "data" / "clips" / f"best_pitch_{pid}.mp4"
                    vp.parent.mkdir(parents=True, exist_ok=True)
                    if _download_mp4(mp4s[0], vp):
                        video_path = vp
                        print(f"Video: {vp}")
                break
        if video_path:
            break
except Exception as e:
    print(f"Video error: {e}")

# Post
auth = tweepy.OAuth1UserHandler(os.environ["X_API_KEY"], os.environ["X_API_SECRET"],
    os.environ["X_ACCESS_TOKEN"], os.environ["X_ACCESS_TOKEN_SECRET"])
v1 = tweepy.API(auth)
client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"],
    consumer_key=os.environ["X_API_KEY"], consumer_secret=os.environ["X_API_SECRET"],
    access_token=os.environ["X_ACCESS_TOKEN"], access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"])

parts = []
if best.get("velocity"): parts.append(f"{float(best['velocity']):.1f} mph")
if best.get("stuff_plus"): parts.append(f"Stuff+ {float(best['stuff_plus']):.0f}")
if best.get("whiff_rate"): parts.append(f"{float(best['whiff_rate']) * 100:.1f}% whiff rate")

tweet = (f"Best Pitch Last Night\n\n"
         f"{name}'s {pitch_display}\n"
         f"{', '.join(parts)}\n\n"
         f"@TJStats @PitchProfiler #MLB #Statcast")

media = v1.media_upload(filename=out)
v1.create_media_metadata(media.media_id, alt_text=f"Best pitch card for {name}")
resp = client.create_tweet(text=tweet, media_ids=[media.media_id])
t1 = resp.data["id"]
print(f"Tweet 1: {t1}")

if video_path and video_path.exists():
    vm = v1.chunked_upload(filename=str(video_path), media_category="tweet_video")
    resp2 = client.create_tweet(text=f"{name} strikeout clip",
                                media_ids=[vm.media_id], in_reply_to_tweet_id=t1)
    print(f"Tweet 2: {resp2.data['id']}")

print(f"\nPosted: https://x.com/BachTalk1/status/{t1}")
