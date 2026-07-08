"""Hero stat card for a newsroom thread: a clean white BachTalk card (matplotlib).

Optional by design — if rendering fails the thread still posts (just without the
chart). The lead tweet carries the game video, so the card rides on a reply.
"""

from __future__ import annotations

import logging
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ...config import SCREENSHOTS_DIR, MLB_SEASON
from .feeds import Lead

log = logging.getLogger(__name__)

NAVY = "#0b1f3a"
GOLD = "#c9a227"
INK = "#1b1b1b"
GRAY = "#6b7280"
TRACK = "#e9edf2"
RED = "#c8102e"

_KIND_TAG = {
    "overperformer": "REGRESSION WATCH",
    "underperformer": "DUE FOR A BREAKOUT",
    "nasty_pitch": "NASTIEST PITCH",
    "bat_speed": "FASTEST BAT",
}


def _f3(v) -> str:
    s = f"{float(v):.3f}"
    return s[1:] if s.startswith("0.") else s


def _spec(lead: Lead):
    """(marquee_value, marquee_label, rows[(label, player, league|None, fmt)])."""
    f = lead.facts
    k = lead.kind
    if k in ("overperformer", "underperformer"):
        g = f.get("gap") or 0.0
        marquee = ("+" if g >= 0 else "−") + _f3(abs(g))
        return marquee, "xwOBA − wOBA gap", [
            ("wOBA", f.get("woba"), f.get("league_woba"), _f3),
            ("xwOBA", f.get("est_woba"), f.get("league_woba"), _f3),
        ]
    if k == "nasty_pitch":
        return f"{f.get('whiff_percent')}%", f"Whiff rate — {f.get('pitch_name')}", [
            ("Whiff %", f.get("whiff_percent"), f.get("league_whiff"), lambda v: f"{v:.1f}"),
            ("Put-away %", f.get("put_away"), None, lambda v: f"{v:.1f}"),
            ("Opp AVG", f.get("opp_ba"), None, _f3),
        ]
    if k == "bat_speed":
        return f"{f.get('avg_bat_speed')}", "mph average bat speed", [
            ("Bat speed", f.get("avg_bat_speed"), f.get("league_bat_speed"), lambda v: f"{v:.1f}"),
            ("Blast %", f.get("blast_per_swing"), None, lambda v: f"{v:.1f}"),
        ]
    return "", "", []


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _render_source_card(lead: Lead):
    """Render a clean 'worth a read' source card for an article reaction."""
    import textwrap
    try:
        f = lead.facts
        outlet = f.get("outlet", "")
        author = f.get("author") or "staff"
        title = f.get("title", lead.subject)

        fig = plt.figure(figsize=(12, 6.75), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.patch.set_facecolor("white")
        ax.add_patch(FancyBboxPatch((0, 0), 0.018, 1, boxstyle="square,pad=0",
                                    facecolor=GOLD, edgecolor="none"))

        ax.text(0.05, 0.90, "WORTH A READ", color=GOLD, fontsize=15, fontweight="bold")
        ax.text(0.965, 0.905, "BachTalk", color=GOLD, fontsize=17,
                fontweight="bold", ha="right")

        lines = textwrap.wrap(title, width=32)[:4]
        y = 0.70
        for line in lines:
            ax.text(0.05, y, line, color=NAVY, fontsize=31, fontweight="bold")
            y -= 0.12
        ax.text(0.05, max(y - 0.02, 0.15), f"{outlet}  ·  {author}",
                color=GRAY, fontsize=19, fontweight="bold")

        ax.text(0.05, 0.045, f"BachTalk  ·  via {outlet}", color=GRAY, fontsize=12)

        out = SCREENSHOTS_DIR / f"newsroom_article_{_safe(lead.subject)}.png"
        fig.savefig(out, facecolor="white")
        plt.close(fig)
        log.info("rendered source card: %s", out.name)
        return out
    except Exception:
        log.warning("source card render failed", exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def render_stat_card(fact_sheet: dict, lead: Lead):
    """Render the card and return its Path, or None on failure."""
    if lead.kind == "article":
        return _render_source_card(lead)
    try:
        marquee, marquee_label, rows = _spec(lead)
        rows = [r for r in rows if r[1] is not None]

        fig = plt.figure(figsize=(12, 6.75), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.patch.set_facecolor("white")

        ax.add_patch(FancyBboxPatch((0, 0), 0.018, 1, boxstyle="square,pad=0",
                                    facecolor=GOLD, edgecolor="none"))

        tag = _KIND_TAG.get(lead.kind, "")
        ax.text(0.05, 0.90, tag, color=(RED if lead.is_red else GOLD),
                fontsize=15, fontweight="bold")
        ax.text(0.05, 0.795, lead.subject, color=NAVY, fontsize=34, fontweight="bold")
        ax.text(0.965, 0.905, "BachTalk", color=GOLD, fontsize=17,
                fontweight="bold", ha="right")
        if lead.is_red:
            ax.text(0.965, 0.845, "CINCINNATI REDS", color=RED, fontsize=11,
                    fontweight="bold", ha="right")

        # Marquee stat
        ax.text(0.05, 0.585, str(marquee), color=GOLD, fontsize=76, fontweight="bold")
        ax.text(0.055, 0.475, marquee_label, color=GRAY, fontsize=17)

        # Comparison bars
        x0, x1 = 0.34, 0.95
        span = x1 - x0
        y = 0.36
        for label, pval, lval, fmt in rows:
            rowmax = max(pval, (lval or 0)) * 1.18 or 1
            ax.text(0.05, y + 0.012, label, color=INK, fontsize=15, va="center")
            ax.add_patch(FancyBboxPatch((x0, y - 0.006), span, 0.036,
                         boxstyle="round,pad=0,rounding_size=0.01",
                         facecolor=TRACK, edgecolor="none"))
            wp = span * min(pval / rowmax, 1)
            ax.add_patch(FancyBboxPatch((x0, y - 0.006), wp, 0.036,
                         boxstyle="round,pad=0,rounding_size=0.01",
                         facecolor=NAVY, edgecolor="none"))
            val_txt = fmt(pval)
            if lval is not None:
                lx = x0 + span * min(lval / rowmax, 1)
                ax.plot([lx, lx], [y - 0.018, y + 0.048], color=GOLD, lw=2.5)
                val_txt += f"   (lg {fmt(lval)})"
            ax.text(x0 + wp + 0.01, y + 0.012, val_txt, color=INK, fontsize=13,
                    va="center", fontweight="bold")
            y -= 0.105

        ax.text(0.05, 0.045, f"BachTalk  ·  Data: Baseball Savant {MLB_SEASON}",
                color=GRAY, fontsize=12)

        out = SCREENSHOTS_DIR / f"newsroom_{_safe(lead.subject)}.png"
        fig.savefig(out, facecolor="white")
        plt.close(fig)
        log.info("rendered stat card: %s", out.name)
        return out
    except Exception:
        log.warning("stat card render failed", exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass
        return None
