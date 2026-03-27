"""Post best pitch of the day — v4 with strike zone + clean design."""
import sys, os, re, logging, io
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, r"C:\Users\IDBac\tjstats-bot")
os.chdir(r"C:\Users\IDBac\tjstats-bot")
from dotenv import load_dotenv; load_dotenv()

from src import pitch_profiler
from src.charts import _fetch_headshot, _headshot_cache, _draw_watermark, SCREENSHOTS_DIR
from src.config import MLB_SEASON, MLB_API_BASE
from src.video_clips import _download_mp4
from pybaseball import statcast_pitcher
import requests, pandas as pd, tweepy, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image as PILImage
from io import BytesIO
from pathlib import Path
from datetime import date, timedelta

PITCH_NAMES = {"FF":"4-Seam Fastball","SI":"Sinker","FC":"Cutter","SL":"Slider",
    "SV":"Sweeper","ST":"Sweeper","CU":"Curveball","KC":"Knuckle Curve",
    "CH":"Changeup","FS":"Splitter","CT":"Cutter"}
PITCH_COLORS = {"FF":"#d62828","SI":"#f77f00","FC":"#8338ec","SL":"#3a86ff",
    "SV":"#00b4d8","ST":"#00b4d8","CU":"#2ec4b6","KC":"#06d6a0",
    "CH":"#ffbe0b","FS":"#fb5607","CT":"#8338ec"}
RESULT_COLORS = {
    "swinging_strike": "#ff6b6b", "swinging_strike_blocked": "#ff6b6b",
    "called_strike": "#3a86ff", "foul": "#ffbe0b", "foul_tip": "#ffbe0b",
    "ball": "#8b949e", "hit_into_play": "#2ec4b6",
    "hit_into_play_no_out": "#2ec4b6", "hit_into_play_score": "#2ec4b6",
}

yesterday = date.today() - timedelta(days=1)
display_date = yesterday.strftime("%m/%d/%Y")

# ── Find best pitch ──
print("Getting game data...")
url = f"{MLB_API_BASE}/schedule?sportId=1&date={yesterday.strftime('%Y-%m-%d')}"
pks = [g["gamePk"] for d in requests.get(url,timeout=15).json().get("dates",[])
       for g in d.get("games",[]) if "Final" in g.get("status",{}).get("detailedState","")]

gpp = pitch_profiler.get_game_pitches(MLB_SEASON)
yest = gpp[(gpp["game_type"]=="R") & (gpp["game_pk"].isin(pks))].copy()
yest["thrown"] = pd.to_numeric(yest["thrown"], errors="coerce")
qual = yest[yest["thrown"]>=10].copy()
for col in ["stuff_plus","whiff_rate","called_strikes_plus_whiffs_percentage"]:
    qual[col] = pd.to_numeric(qual[col], errors="coerce")
qual["score"] = qual["stuff_plus"].fillna(100)*0.4 + qual["whiff_rate"].fillna(0)*100*0.3 + qual["called_strikes_plus_whiffs_percentage"].fillna(0)*100*0.3

best = qual.sort_values("score", ascending=False).iloc[0]
name = str(best["pitcher_name"])
pt = str(best["pitch_type"])
pid = int(best["pitcher_id"])
gpk = int(best["game_pk"])
pitch_display = PITCH_NAMES.get(pt, pt)
accent = PITCH_COLORS.get(pt, "#3a86ff")
p_throws = str(best.get("p_throws", "R"))

# Get team
team = ""
try:
    mlb_resp = requests.get(f'https://statsapi.mlb.com/api/v1/people/{pid}?hydrate=currentTeam', timeout=10)
    team_name = mlb_resp.json()['people'][0].get('currentTeam',{}).get('name','')
    _N2A = {"Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
        "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
        "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
        "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
        "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
        "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
        "New York Yankees":"NYY","Oakland Athletics":"OAK","Philadelphia Phillies":"PHI",
        "Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF",
        "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TB",
        "Texas Rangers":"TEX","Toronto Blue Jays":"TOR","Washington Nationals":"WSH"}
    team = _N2A.get(team_name, "")
except: pass

print(f"Best: {name} {pitch_display} ({team}) stuff+={best.get('stuff_plus',0):.0f}")

# Get pitch locations from Statcast
print("Pulling Statcast locations...")
sc = statcast_pitcher(yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d"), pid)
pitch_locs = sc[sc["pitch_type"] == pt].copy() if sc is not None and not sc.empty else pd.DataFrame()
print(f"  {len(pitch_locs)} {pt} pitches with location data")

# Get headshot + logo
_headshot_cache.clear()
headshot = _fetch_headshot(pid, accent=accent)

logo_slugs = {"ARI":"ari","ATL":"atl","BAL":"bal","BOS":"bos","CHC":"chc","CWS":"chw",
    "CIN":"cin","CLE":"cle","COL":"col","DET":"det","HOU":"hou","KC":"kc",
    "LAA":"laa","LAD":"lad","MIA":"mia","MIL":"mil","MIN":"min","NYM":"nym",
    "NYY":"nyy","OAK":"oak","PHI":"phi","PIT":"pit","SD":"sd","SF":"sf",
    "SEA":"sea","STL":"stl","TB":"tb","TEX":"tex","TOR":"tor","WSH":"wsh"}
team_logo = None
slug = logo_slugs.get(team)
if slug:
    try:
        lr = requests.get(f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/{slug}.png&h=500&w=500",timeout=10,allow_redirects=True)
        team_logo = PILImage.open(BytesIO(lr.content)).convert("RGBA")
    except: pass

hand = "LHP" if p_throws == "L" else "RHP"

# ── Build card ──
fig = plt.figure(figsize=(22, 18), dpi=150)
fig.set_facecolor("white")

gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 5, 0.4], hspace=0.25, wspace=0.25,
                       left=0.05, right=0.95, top=0.92, bottom=0.04)

# ── Header (spanning both columns) ──
if headshot is not None:
    ax_hs = fig.add_axes([0.03, 0.88, 0.06, 0.08])
    ax_hs.imshow(headshot); ax_hs.axis("off")
if team_logo:
    ax_logo = fig.add_axes([0.90, 0.88, 0.06, 0.08])
    ax_logo.set_xlim(0,1.3); ax_logo.set_ylim(0,1)
    ax_logo.imshow(team_logo, extent=[0.3,1.3,0,1], origin="upper"); ax_logo.axis("off")

fig.text(0.11, 0.955, name, fontsize=38, fontweight="bold", color="#1a1a2e", va="top")
fig.text(0.11, 0.925, f"{pitch_display}  |  {hand}  |  {team_name}", fontsize=16, color="#555555", va="top")
fig.text(0.11, 0.9, f"Best Pitch Last Night  |  {display_date}", fontsize=13, color=accent, va="top", fontweight="bold")

# Hero stats bar
hero = [
    (f"{best.get('velocity',0):.1f}", "VELO"),
    (f"{best.get('stuff_plus',0):.0f}", "STUFF+"),
    (f"{float(best.get('whiff_rate',0))*100:.1f}%", "WHIFF%"),
    (f"{float(best.get('run_value_per_100_pitches',0)):.1f}", "RV/100"),
    (f"{best.get('woba',0):.3f}", "wOBA"),
    (f"{int(best.get('thrown',0))}", "PITCHES"),
]
for i, (val, label) in enumerate(hero):
    x = 0.07 + i * 0.15
    fig.text(x, 0.855, val, fontsize=28, fontweight="bold", color="#1a1a2e", ha="center", va="top")
    fig.text(x, 0.825, label, fontsize=10, color=accent, ha="center", va="top", fontweight="bold")

fig.add_artist(plt.Line2D([0.03, 0.97], [0.815, 0.815], transform=fig.transFigure, color="#e0e0e0", linewidth=2))

# ── Movement Chart (bottom left) ──
ax_move = fig.add_subplot(gs[1, 0])
ax_move.set_facecolor("#fafafa")
ax_move.set_title("Pitch Movement Profile", fontsize=20, fontweight="bold", color="#1a1a2e", pad=15)
ax_move.axhline(0, color="#cccccc", linewidth=1)
ax_move.axvline(0, color="#cccccc", linewidth=1)

hb = float(best.get("hb",0) or 0)
ivb = float(best.get("ivb",0) or 0)
ax_move.scatter(hb, ivb, c=accent, s=800, edgecolors="#1a1a2e", linewidths=2.5, zorder=5, alpha=0.9)
ax_move.annotate(pitch_display, (hb, ivb), textcoords="offset points", xytext=(12, 12),
                 fontsize=14, fontweight="bold", color=accent,
                 arrowprops=dict(arrowstyle="-", color=accent, alpha=0.5))

ax_move.set_xlim(-22, 22); ax_move.set_ylim(-15, 25)
ax_move.set_xlabel("Horizontal Break (in)", fontsize=14, fontweight="bold", color="#333333")
ax_move.set_ylabel("Induced Vertical Break (in)", fontsize=14, fontweight="bold", color="#333333")
ax_move.tick_params(labelsize=12, colors="#666666")
ax_move.grid(True, alpha=0.2, linewidth=0.8)
ax_move.set_aspect("equal", adjustable="box")
for spine in ax_move.spines.values(): spine.set_color("#dddddd"); spine.set_linewidth(1.5)

# ── Strike Zone (bottom right) ──
ax_zone = fig.add_subplot(gs[1, 1])
ax_zone.set_facecolor("#fafafa")
ax_zone.set_title("Strike Zone — Catcher's View", fontsize=20, fontweight="bold", color="#1a1a2e", pad=15)

# Draw zone box
zone = Rectangle((-0.83, 1.5), 1.66, 2.0, linewidth=2.5, edgecolor="#1a1a2e", facecolor="none", zorder=3)
ax_zone.add_patch(zone)
# Inner grid
for x in [-0.277, 0.277]:
    ax_zone.plot([x, x], [1.5, 3.5], color="#cccccc", linewidth=0.8, alpha=0.5)
for y in [2.167, 2.833]:
    ax_zone.plot([-0.83, 0.83], [y, y], color="#cccccc", linewidth=0.8, alpha=0.5)

# Plot pitch locations
if not pitch_locs.empty:
    for _, p in pitch_locs.iterrows():
        px = p.get("plate_x", 0)
        pz = p.get("plate_z", 0)
        desc = str(p.get("description", ""))
        color = RESULT_COLORS.get(desc, "#888888")
        marker = "o"
        size = 120 if "swinging" in desc else 100
        alpha = 0.85
        ax_zone.scatter(px, pz, c=color, s=size, edgecolors="white", linewidths=1,
                       alpha=alpha, zorder=4, marker=marker)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0],marker="o",color="none",markerfacecolor="#ff6b6b",markeredgecolor="white",markersize=8,label="Whiff"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor="#3a86ff",markeredgecolor="white",markersize=8,label="Called K"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor="#8b949e",markeredgecolor="white",markersize=8,label="Ball"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor="#2ec4b6",markeredgecolor="white",markersize=8,label="In Play"),
    ]
    ax_zone.legend(handles=legend_els, loc="lower center", ncol=4, fontsize=10,
                   framealpha=0.9, edgecolor="#dddddd", bbox_to_anchor=(0.5, -0.08))

ax_zone.set_xlim(-2.5, 2.5); ax_zone.set_ylim(-0.5, 5.0)
ax_zone.set_aspect("equal")
ax_zone.set_xlabel("Plate Side (ft)", fontsize=14, fontweight="bold", color="#333333")
ax_zone.set_ylabel("Height (ft)", fontsize=14, fontweight="bold", color="#333333")
ax_zone.tick_params(labelsize=12, colors="#666666")
for spine in ax_zone.spines.values(): spine.set_color("#dddddd"); spine.set_linewidth(1.5)

# ── Footer ──
fig.text(0.03, 0.015, "By: @BachTalk1", fontsize=16, fontweight="bold", color="#1a1a2e")
fig.text(0.5, 0.015, "Data: Pitch Profiler + Baseball Savant  |  Images: MLB, ESPN", fontsize=11, color="#888888", ha="center")
fig.text(0.97, 0.015, "Best Pitch Last Night", fontsize=11, color="#888888", ha="right")

# Watermark
_draw_watermark(fig, alpha=0.06, scale=0.3, dark_bg=False)

out = str(SCREENSHOTS_DIR / "best_pitch_v4.png")
fig.savefig(out, facecolor="white", dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Card: {out}")

# ── Get video ──
video_path = None
try:
    pbp = requests.get(f"https://statsapi.mlb.com/api/v1/game/{gpk}/playByPlay", timeout=15).json()
    for play in pbp.get("allPlays",[]):
        if play.get("matchup",{}).get("pitcher",{}).get("id") != pid: continue
        if "strikeout" not in (play.get("result",{}).get("event","")).lower(): continue
        for pe in reversed(play.get("playEvents",[])):
            play_id = pe.get("playId")
            if play_id:
                sr = requests.get(f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}", timeout=10)
                mp4s = re.findall(r'https?://sporty-clips\.mlb\.com/[^\s"<>]+\.mp4', sr.text)
                if mp4s:
                    vp = Path(SCREENSHOTS_DIR).parent/"data"/"clips"/f"best_pitch_{pid}.mp4"
                    vp.parent.mkdir(parents=True, exist_ok=True)
                    if _download_mp4(mp4s[0], vp):
                        video_path = vp; print(f"Video: {vp}")
                break
        if video_path: break
except Exception as e:
    print(f"Video error: {e}")

# ── Post ──
auth = tweepy.OAuth1UserHandler(os.environ["X_API_KEY"],os.environ["X_API_SECRET"],
    os.environ["X_ACCESS_TOKEN"],os.environ["X_ACCESS_TOKEN_SECRET"])
v1 = tweepy.API(auth)
client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"],
    consumer_key=os.environ["X_API_KEY"],consumer_secret=os.environ["X_API_SECRET"],
    access_token=os.environ["X_ACCESS_TOKEN"],access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"])

parts = []
if best.get("velocity"): parts.append(f"{float(best['velocity']):.1f} mph")
if best.get("stuff_plus"): parts.append(f"Stuff+ {float(best['stuff_plus']):.0f}")
if best.get("whiff_rate"): parts.append(f"{float(best['whiff_rate'])*100:.1f}% whiff rate")

tweet = f"Best Pitch Last Night\n\n{name}'s {pitch_display}\n{', '.join(parts)}\n\n@TJStats @PitchProfiler #MLB #Statcast"

media = v1.media_upload(filename=out)
v1.create_media_metadata(media.media_id, alt_text=f"Best pitch card for {name}")
resp = client.create_tweet(text=tweet, media_ids=[media.media_id])
t1 = resp.data["id"]
print(f"Tweet 1: {t1}")

if video_path and video_path.exists():
    vm = v1.chunked_upload(filename=str(video_path), media_category="tweet_video")
    resp2 = client.create_tweet(text=f"{name} strikeout clip", media_ids=[vm.media_id], in_reply_to_tweet_id=t1)
    print(f"Tweet 2: {resp2.data['id']}")

print(f"\nPosted: https://x.com/BachTalk1/status/{t1}")
