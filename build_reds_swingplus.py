"""Build Reds Swing+ leaderboard visualization."""
import io
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage, ImageDraw

MLB_SEASON = 2026
FEATURES = [
    "bat_speed", "squared_up_rate", "squared_up_speed_rate",
    "swing_length", "sweetspot_speed_high", "hit_into_play_rate",
    "swords", "brl_percent", "anglesweetspotpercent", "ev95percent",
]
COL_MAP = {
    "avg_bat_speed": "bat_speed",
    "squared_up_per_bat_contact": "squared_up_rate",
    "blast_per_bat_contact": "squared_up_speed_rate",
    "swing_length": "swing_length",
    "hard_swing_rate": "sweetspot_speed_high",
    "batted_ball_event_per_swing": "hit_into_play_rate",
    "swords": "swords",
}

# Reds roster
roster = requests.get(
    "https://statsapi.mlb.com/api/v1/teams/113/roster?rosterType=active",
    timeout=15,
).json()
reds_ids = set()
reds_names = {}
for p in roster.get("roster", []):
    person = p.get("person", {})
    pos = p.get("position", {}).get("abbreviation", "")
    if pos not in ("P", "TWP"):
        pid = person.get("id")
        reds_ids.add(pid)
        reds_names[pid] = person.get("fullName")
print(f"Reds hitters on roster: {len(reds_ids)}")

# Bat tracking - qualified for training
url50 = (
    f"https://baseballsavant.mlb.com/leaderboard/bat-tracking?attackZone=&batSide=&"
    f"contactType=&count=&dateStart=&dateEnd=&gameType=&isHardHit=&minSwings=50&"
    f"minGroupSwings=1&pitchHand=&pitchType=&playerPool=All&season={MLB_SEASON}&"
    f"seasonStart=&seasonEnd=&team=&type=batter&csv=true"
)
bt50 = pd.read_csv(io.StringIO(requests.get(url50, timeout=30).text))

# Bat tracking - all hitters
url1 = url50.replace("minSwings=50", "minSwings=1")
bt1 = pd.read_csv(io.StringIO(requests.get(url1, timeout=30).text))

# Barrels
brl_url = (
    f"https://baseballsavant.mlb.com/leaderboard/statcast?type=batter&"
    f"year={MLB_SEASON}&position=&team=&min=1&csv=true"
)
brl_df = pd.read_csv(io.StringIO(requests.get(brl_url, timeout=30).text))
brl_df["_pid"] = pd.to_numeric(brl_df["player_id"], errors="coerce")

# Launch quality from pybaseball
from pybaseball import statcast_batter_exitvelo_barrels
ev_df = statcast_batter_exitvelo_barrels(MLB_SEASON, minBBE=10)
ev_df["_pid"] = pd.to_numeric(ev_df["player_id"], errors="coerce")

# xwOBA
xw_url = (
    f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&"
    f"year={MLB_SEASON}&position=&team=&min=20&csv=true"
)
xw = pd.read_csv(io.StringIO(requests.get(xw_url, timeout=30).text))
xw["_pid"] = pd.to_numeric(xw["player_id"], errors="coerce")


def prep(bt):
    missing = [k for k in COL_MAP if k not in bt.columns]
    for k in missing:
        if k + "_qualified" in bt.columns:
            bt[k] = bt[k + "_qualified"]
    r = bt.rename(columns=COL_MAP)
    nc = next((c for c in ["name", "batter_name", "player_name"] if c in r.columns), None)
    ic = next((c for c in ["id", "batter_id", "player_id", "savant_batter_id"] if c in r.columns), None)
    r["name_fg"] = r[nc].apply(
        lambda x: " ".join(str(x).split(", ")[::-1]).strip() if ", " in str(x) else str(x).strip()
    )
    r["_pid"] = pd.to_numeric(r[ic], errors="coerce")
    r = r.merge(brl_df[["_pid", "brl_percent"]], on="_pid", how="left")
    r = r.merge(ev_df[["_pid", "anglesweetspotpercent", "ev95percent"]], on="_pid", how="left")
    for f in FEATURES:
        r[f] = pd.to_numeric(r[f], errors="coerce")
    return r


train_bt = prep(bt50.copy()).dropna(subset=FEATURES)
all_bt = prep(bt1.copy())

# Train
train_merged = (
    train_bt.merge(xw[["_pid", "est_woba"]], on="_pid", how="inner")
    .rename(columns={"est_woba": "xwOBA"})
    .dropna(subset=["xwOBA"])
)
X_train = train_merged[FEATURES].values
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=min(10, len(train_merged))).fit(
    X_train_s, train_merged["xwOBA"].values
)
pred_train = ridge.predict(X_train_s)
pm, ps = pred_train.mean(), pred_train.std()

# Score Reds
reds_data = []
for pid in reds_ids:
    row = all_bt[all_bt["_pid"] == pid]
    if row.empty:
        continue
    r = row.iloc[0]
    feats = [0.0 if pd.isna(r.get(f, np.nan)) else r[f] for f in FEATURES]
    X_h = scaler.transform([feats])
    pred_h = ridge.predict(X_h)
    sp = round(100 + ((pred_h[0] - pm) / ps) * 15, 1)
    reds_data.append({
        "pid": pid,
        "name": reds_names.get(pid, str(pid)),
        "swing_plus": sp,
        "bat_speed": r.get("bat_speed", 0) or 0,
        "brl_percent": r.get("brl_percent", 0) if not pd.isna(r.get("brl_percent", 0)) else 0,
        "squared_up_rate": r.get("squared_up_rate", 0) or 0,
        "swing_length": r.get("swing_length", 0) or 0,
    })

reds_df = pd.DataFrame(reds_data).sort_values("swing_plus", ascending=False).reset_index(drop=True)
print(f"\nReds hitters with bat tracking data: {len(reds_df)}")
print(reds_df[["name", "swing_plus", "bat_speed", "brl_percent"]].to_string())


def fetch_headshot(pid):
    try:
        url = f"https://securea.mlb.com/mlb/images/players/head_shot/{int(pid)}.jpg"
        resp = requests.get(url, timeout=10, allow_redirects=True)
        if resp.status_code != 200:
            return None
        img = PILImage.open(io.BytesIO(resp.content)).convert("RGBA")
        size = min(img.size)
        l = (img.width - size) // 2
        t = (img.height - size) // 2
        img = img.crop((l, t, l + size, t + size)).resize((100, 100), PILImage.LANCZOS)
        mask = PILImage.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, 100, 100), fill=255)
        img.putalpha(mask)
        return np.array(img)
    except Exception:
        return None


n = len(reds_df)
fig, ax = plt.subplots(figsize=(11, max(7, n * 0.55)))
bg = "#0a0e1a"
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

reds_red = "#c6011f"
white = "#ffffff"
gray = "#888888"

max_sp = reds_df["swing_plus"].max()
min_sp = reds_df["swing_plus"].min()
# Bar baseline sits below the lowest Swing+ value so every bar renders
bar_left = float(np.floor(min(min_sp, 90) / 5.0) * 5 - 5)
x_max = max(max_sp + 8, 130)
hs_x = bar_left - 3
rank_x = bar_left - 6
left_pad = 8  # extra room so rank labels don't clip
y_positions = np.arange(n - 1, -1, -1)

ax.axvline(x=100, color="#666666", linestyle="--", linewidth=0.8, zorder=1, alpha=0.5)
ax.text(100, n - 0.3, "100 = MLB Avg", ha="center", va="bottom", fontsize=8, color="#888888")

for i in range(n):
    row = reds_df.iloc[i]
    y = y_positions[i]
    sp = row["swing_plus"]
    name = row["name"]
    bat_spd = row["bat_speed"]
    brl = row["brl_percent"]

    color = reds_red if sp >= 100 else "#555555"

    ax.barh(y, sp - bar_left, left=bar_left, height=0.65, color=color,
            edgecolor="none", zorder=2)

    ax.text(rank_x, y, f"#{i+1}", ha="center", va="center",
            fontsize=10, fontweight="bold", color=gray)

    hs = fetch_headshot(row["pid"])
    if hs is not None:
        imagebox = OffsetImage(hs, zoom=0.28)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (hs_x, y), frameon=False,
                            box_alignment=(0.5, 0.5), zorder=4)
        ax.add_artist(ab)

    ax.text(bar_left + 0.3, y + 0.15, name, ha="left", va="center",
            fontsize=11, fontweight="bold", color=white, zorder=5)

    subtitle = f"{bat_spd:.1f} mph  \u00b7  {brl:.1f}% BRL"
    ax.text(bar_left + 0.3, y - 0.18, subtitle, ha="left", va="center",
            fontsize=8, color="#cccccc", zorder=5)

    ax.text(sp + 0.5, y, f"{sp:.1f}", ha="left", va="center",
            fontsize=11, fontweight="bold", color=white, zorder=5)

ax.set_xlim(rank_x - left_pad, x_max)
ax.set_ylim(-0.6, n - 0.4)
ax.set_yticks([])
ax.set_xlabel("Swing+", fontsize=11, color="#999999")
ax.tick_params(axis="x", colors="#666666", labelsize=9)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#333333")

fig.suptitle("Cincinnati Reds \u2014 Swing+ Leaderboard",
             fontsize=20, fontweight="bold", color=white, y=0.97)
ax.set_title(f"{MLB_SEASON} Season  \u00b7  Mechanics + Barrel Model",
             fontsize=10, color="#888888", pad=12)

fig.text(0.03, 0.01, "Data: Baseball Savant", fontsize=7, color="#555555")
fig.text(0.97, 0.01, "@BachTalk1", fontsize=7, color="#555555", ha="right")

out = Path("C:/Users/IDBac/tjstats-bot/screenshots/reds_swing_plus.png")
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig(out, dpi=150, facecolor=bg, bbox_inches="tight", pad_inches=0.3)
plt.close()
print(f"\nSaved: {out}")
