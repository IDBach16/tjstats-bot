"""Test adding launch quality features to Swing+ model."""
import io
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pybaseball import statcast_batter_exitvelo_barrels

MLB_SEASON = 2026
FEATURES_BASE = [
    "bat_speed", "squared_up_rate", "squared_up_speed_rate",
    "swing_length", "sweetspot_speed_high", "hit_into_play_rate",
    "swords", "brl_percent",
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

url = (
    f"https://baseballsavant.mlb.com/leaderboard/bat-tracking?attackZone=&batSide=&"
    f"contactType=&count=&dateStart=&dateEnd=&gameType=&isHardHit=&minSwings=50&"
    f"minGroupSwings=1&pitchHand=&pitchType=&playerPool=All&season={MLB_SEASON}&"
    f"seasonStart=&seasonEnd=&team=&type=batter&csv=true"
)
bt = pd.read_csv(io.StringIO(requests.get(url, timeout=30).text))
missing = [k for k in COL_MAP if k not in bt.columns]
for k in missing:
    if k + "_qualified" in bt.columns:
        bt[k] = bt[k + "_qualified"]
bt = bt.rename(columns=COL_MAP)
ic = next((c for c in ["id", "batter_id", "player_id"] if c in bt.columns), None)
bt["_pid"] = pd.to_numeric(bt[ic], errors="coerce")

brl_url = (
    f"https://baseballsavant.mlb.com/leaderboard/statcast?type=batter&"
    f"year={MLB_SEASON}&position=&team=&min=20&csv=true"
)
brl_df = pd.read_csv(io.StringIO(requests.get(brl_url, timeout=30).text))
brl_df["_pid"] = pd.to_numeric(brl_df["player_id"], errors="coerce")
bt = bt.merge(brl_df[["_pid", "brl_percent"]], on="_pid", how="left")

ev_df = statcast_batter_exitvelo_barrels(MLB_SEASON, minBBE=10)
ev_df["_pid"] = pd.to_numeric(ev_df["player_id"], errors="coerce")
bt = bt.merge(
    ev_df[["_pid", "anglesweetspotpercent", "ev95percent", "avg_hit_speed"]],
    on="_pid", how="left",
)

xw_url = (
    f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&"
    f"year={MLB_SEASON}&position=&team=&min=20&csv=true"
)
xw = pd.read_csv(io.StringIO(requests.get(xw_url, timeout=30).text))
xw["_pid"] = pd.to_numeric(xw["player_id"], errors="coerce")

NEW_FEATS = ["anglesweetspotpercent", "ev95percent", "avg_hit_speed"]
for f in FEATURES_BASE + NEW_FEATS:
    bt[f] = pd.to_numeric(bt[f], errors="coerce")

merged = (
    bt.merge(xw[["_pid", "est_woba"]], on="_pid", how="inner")
    .rename(columns={"est_woba": "xwOBA"})
    .dropna(subset=["xwOBA"] + FEATURES_BASE + NEW_FEATS)
)
print(f"Hitters with all features: {len(merged)}")
print()

y = merged["xwOBA"].values

configs = {
    "8 features (current)": FEATURES_BASE,
    "+ anglesweetspotpercent": FEATURES_BASE + ["anglesweetspotpercent"],
    "+ ev95percent": FEATURES_BASE + ["ev95percent"],
    "+ both": FEATURES_BASE + ["anglesweetspotpercent", "ev95percent"],
}

hayes_id = 663647
hayes_idx = merged.index[merged["_pid"] == hayes_id]

print("=== Model fit comparison ===")
for label, feats in configs.items():
    X = merged[feats].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=10)
    cv_scores = cross_val_score(ridge, X_s, y, cv=10, scoring="r2")
    ridge.fit(X_s, y)
    train_r2 = ridge.score(X_s, y)
    pred = ridge.predict(X_s)
    pm, ps = pred.mean(), pred.std()
    swing_plus = 100 + ((pred - pm) / ps) * 15

    print(f"{label}:")
    print(f"  Train R2:    {train_r2:.4f}")
    print(f"  CV R2:       {cv_scores.mean():.4f}")
    print(f"  Overfit gap: {train_r2 - cv_scores.mean():.4f}")
    if len(hayes_idx) > 0:
        i = merged.index.get_loc(hayes_idx[0])
        rank = int((swing_plus > swing_plus[i]).sum() + 1)
        print(f"  Hayes Swing+: {swing_plus[i]:.1f}  (rank #{rank}/{len(merged)})")
    print()

print("=== Correlation with xwOBA (new features) ===")
for f in NEW_FEATS:
    r = merged[f].corr(merged["xwOBA"])
    print(f"  {f}: r={r:.3f}")

print()
print("=== Hayes raw values for new features ===")
if len(hayes_idx) > 0:
    hrow = merged.loc[hayes_idx[0]]
    for f in NEW_FEATS:
        v = hrow[f]
        avg = merged[f].mean()
        pct = (merged[f] < v).mean() * 100
        print(f"  {f}: {v:.2f}  (avg {avg:.2f}, {pct:.0f}th pctl)")
