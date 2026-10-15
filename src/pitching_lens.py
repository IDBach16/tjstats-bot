"""How to read a pitcher -- the analytical lens every pitching prompt shares.

Distilled from Lance Brozdowski's Substack and YouTube work (2018-2026; the
full study with sources is in Desktop\\Pitching_PHIL: PHILOSOPHY.md,
knowledge.jsonl and "Pitching Metrics Best Practices.xlsx"). Every benchmark
below is one he states, for MLB pitchers.

The numbers are for INTERPRETING a pitcher, not for quoting: posts may only
cite the pitcher's own stats from the data they are handed, so a fact-checker
or reader never sees a benchmark presented as that pitcher's number.

Also home of the derived numbers the lens needs (fastball-relative velocity
and drop gaps), computed in code because the model should explain numbers,
not do arithmetic on them.
"""

from __future__ import annotations

LENS = """HOW TO READ A PITCHER (Lance Brozdowski's framework; MLB benchmarks)

Start with the fastball, and with velocity.
- Velocity buys margin for everything else, and its value is nonlinear: ~94 mph is where four-seam quality jumps, and at 96+ mph shape barely matters. A 93-95 mph fastball needs unusual shape to stand out.
- A change under ~1 mph is noise. A ~2 mph drop alongside falling whiffs is a red flag.
- Sliders grade well at 85+ mph almost regardless of shape.

Never read movement in raw inches; read it against context.
- Four-seam IVB (ride): ~16 in is MLB average from a ~5.8 ft release, ~12-13 in is normal from ~5 ft. Ride above what the arm slot predicts is what hitters miss; ride that matches the slot is stock. A lower release flattens the approach angle, so average ride can play up.
- Flat vertical approach angle is what lets a four-seam live at the top of the zone. Ride fastballs belong up.
- Lower slots trade ride for arm-side run; spin efficiency decides how much run a four-seam has.
- Spin is capacity, judged per mph (~25 rpm per mph is normal), not as a raw number.

Grade secondaries against the pitcher's own fastball.
- Changeups and splitters: the drop gap (vertical separation from the fastball) matters more than the velocity gap for whiffs. The ideal whiff changeup is hard, about 7-9 mph off the fastball, with 10+ in more drop. Under 7 mph AND under 10 in of separation tends to be average or worse. 10-12 in is above average, 13+ in is great. Velocity-gap changeups get grounders; drop-gap changeups get whiffs.
- Breaking balls: velocity against movement is the core tradeoff, and the rare pitch that has both is the special one. A hard slider that adds drop while holding velocity beats a big, slow sweep.
- Sweepers are same-handed weapons and cooler on opposite-handed hitters; shorter, harder sliders and downer shapes work against both sides.
- Splitters are the best whiff and barrel-suppressing pitch in baseball, and roughly platoon neutral.

Bat-missing is the north star; compare every rate to that pitch type's average.
- Average whiff rates: four-seam ~22%, slider ~32-34%, changeup ~31%, splitter ~35%. A 25% fastball and a 25% slider mean opposite things.
- Chase above 30% is good for a secondary.
- Swinging strikes back strikeouts: about 13-14% SwStr supports a ~25% K rate. A strikeout jump without SwStr support is a mirage.

Strikes and command.
- Strike-throwing is sticky: ~65% strikes is the starter baseline. Zone rate ~49% is MLB average, and under ~48% rarely produces an above-average K-BB.
- A low zone rate works only while hitters keep chasing and whiffing.
- Pitches miss in the direction they move. Much of what looks like a pitch "getting worse" is a pitch going somewhere new.
- Command lets a pitcher punch above his stuff.

Results versus underlying skill.
- K-BB% is the simplest strong stat: ~15% is good for a starter, 20%+ is great (pitchers above 20% rarely fail), 30% is elite.
- Trust FIP/xFIP/SIERA over ERA. A gap of ~0.4-0.5 runs is normal, while a full-run gap demands an explanation (luck, defense, velocity, or a rare profile that beats its peripherals).
- Damage: barrels around 8-9% are average. High barrels are a location problem first and a velocity problem second.
- Stuff+ (100 = average, 112 = plus, 124 = plus-plus) is the best starting point and shows potential, not a guarantee. It undersells lefties, changeups, cutters, short hard sliders and odd release points, so when whiffs and contact disagree with the grade, trust what hitters do. As results pile up, the grade matters less.

Usage is the cheapest lever, and it reveals intent.
- Four-seamers allow the most damage, especially when behind in the count. "Make the four-seam better, then throw it less" is the modern template; 45%+ four-seam usage needs an exceptional fastball.
- Throw the worst pitch less. Move usage toward what misses bats and away from what is getting hit.
- With two strikes, the pitch with the most chase and whiff should finish hitters.
- Deep mixes (several pitches at 10%+ to each side) keep hitters guessing, because hitters improve against a pitch each time they see it.

How to write it.
- Lead with the one thing that makes this pitcher interesting, and explain WHY in the terms above: the fastball relative to its slot, a secondary relative to the fastball, bat-missing relative to its pitch-type average, results relative to underlying skill.
- Cite only this pitcher's own numbers from the data given. Use the benchmarks above to judge, in words ("well above average for a slider", "more drop than his fastball suggests"), never as figures attributed to him.
- If the data cannot support a read (for example no release height to judge ride against), don't make that read.
- Never mention this framework or whose it is. Write the analysis as your own.
"""

# The model every pitching prompt writes with. Chosen 2026-10-15 by testing
# the lens on five real pitchers: Haiku 4.5 invented facts ("elite spin
# efficiency", an age) and quoted benchmark numbers; Haiku 5.5 mislabeled a
# curveball as a changeup; Sonnet 5.5 applied the lens correctly every time.
ANALYST_MODEL = "claude-sonnet-5-5"


def analyst_request() -> dict:
    """Shared kwargs for client.messages.create(**analyst_request(), messages=...).

    Sonnet 5.5 thinks by default and max_tokens covers thinking + text, so the
    old 80-150 token caps would end before any words; low effort keeps a
    two-sentence take quick. The server-side refusal fallback is sent raw
    (header + body) so the unpinned SDK needn't know the parameter.
    """
    return dict(
        model=ANALYST_MODEL,
        max_tokens=3000,
        output_config={"effort": "low"},
        system=LENS,
        extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
        extra_body={"fallbacks": "default"},
    )


def reply_text(message) -> str:
    """Text of a reply, read by block type (a reply can open with a thinking
    block). Empty on a refusal or a reply cut off by max_tokens."""
    if getattr(message, "stop_reason", None) in ("refusal", "max_tokens"):
        return ""
    return "".join(b.text for b in message.content if getattr(b, "type", "") == "text").strip()


# Primary-fastball candidates, in preference order when usage ties.
_FASTBALLS = ("FF", "SI", "FC")
_OFFSPEED = ("CH", "FS", "FO", "SC")
_BREAKING = ("SL", "ST", "CU", "KC", "SV", "CS")


# Pitch-type whiff averages he states (MLB). Only these four: he gives no
# sinker / cutter / curveball figure, and an invented one would be worse.
WHIFF_AVG = {"FF": 22.0, "SL": 33.0, "CH": 31.0, "FS": 35.0}


def whiff_tag(code: str, whiff: float | None) -> str:
    """'above average for a FF' style label, so the model judges whiff rate
    against its pitch type without being handed a number to quote."""
    avg = WHIFF_AVG.get(code)
    if avg is None or whiff is None:
        return ""
    diff = whiff - avg
    if diff >= 6:
        return f"well above average for a {code}"
    if diff >= 2:
        return f"above average for a {code}"
    if diff > -2:
        return f"about average for a {code}"
    if diff > -6:
        return f"below average for a {code}"
    return f"well below average for a {code}"


def _drop_verdict(drop: float) -> str:
    # His offspeed bands: under 10 in sits below the whiff threshold,
    # 10-12 above average, 13+ great.
    if drop >= 13:
        return "great drop separation"
    if drop >= 10:
        return "above-average drop separation"
    return "drop separation below the ~10 in whiff threshold"


def separations(pitches: list[dict]) -> list[str]:
    """Fastball-relative velocity and drop gaps WITH his verdicts, computed
    here so the model explains them instead of doing the subtraction or
    misjudging the band.

    Offspeed is graded against the four-seam when he throws one (his rule is
    separation from the four-seam), otherwise against his primary fastball.
    `pitches` items need pitch_name (a Statcast code), usage, and optionally
    velocity and ivb. Returns readable lines, empty when there is no fastball.
    """
    fbs = [p for p in pitches if p.get("pitch_name") in _FASTBALLS and p.get("velocity")]
    if not fbs:
        return []
    primary = max(fbs, key=lambda p: (p.get("usage") or 0, -_FASTBALLS.index(p["pitch_name"])))
    ff = next((p for p in fbs if p["pitch_name"] == "FF"), None)
    out = []
    for p in pitches:
        code = p.get("pitch_name")
        if code not in _OFFSPEED + _BREAKING or not p.get("velocity"):
            continue
        fb = ff if (code in _OFFSPEED and ff) else primary
        gap = fb["velocity"] - p["velocity"]
        parts = [f"{code} vs {fb['pitch_name']}: {gap:.1f} mph slower"]
        if p.get("ivb") is not None and fb.get("ivb") is not None:
            drop = fb["ivb"] - p["ivb"]
            parts.append(f"{drop:.1f} in more drop")
            if code in _OFFSPEED:
                verdict = _drop_verdict(drop)
                if gap < 7 and drop < 10:
                    verdict += "; under 7 mph and under 10 in together usually means an average-or-worse changeup"
                elif gap <= 9 and drop >= 10:
                    verdict += "; hard with real drop, the whiff-getting shape"
                elif gap > 9 and drop < 10:
                    verdict += "; a velocity-gap changeup, more ground balls than whiffs"
                parts.append(f"({verdict})")
        out.append(", ".join(parts))
    return out
