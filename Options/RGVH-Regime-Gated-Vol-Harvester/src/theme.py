"""
Institutional research-publication theme for matplotlib + plotly.

Inspired by the visual language of GS / JPM / Bridgewater research notes:
  - Pure-white backgrounds, near-black text
  - Deep institutional navy as primary, restrained gold as accent
  - Subtle horizontal gridlines only, no top/right spines
  - Generous whitespace, clear hierarchy
  - Publication-grade DPI (200) and typography
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color palette — institutional research aesthetic
# ---------------------------------------------------------------------------
COLORS = {
    # Brand
    "navy":        "#0A2540",     # primary brand
    "navy_dark":   "#061A2F",
    "navy_light":  "#3B5573",
    "gold":        "#C9A227",     # accent
    "gold_light":  "#E0C870",
    "gold_dark":   "#9A7E1F",

    # Semantic
    "positive":    "#1B5E20",     # deep green for gains/wins
    "negative":    "#A92424",     # deep red for losses
    "neutral":     "#455A64",     # slate grey
    "subtle":      "#90A4AE",     # very light slate
    "muted":       "#CFD8DC",     # subtle gridlines / shading

    # Text
    "ink":         "#0A1929",     # body text
    "ink_soft":    "#37474F",     # secondary text
    "ink_muted":   "#78909C",     # caption/footnote

    # Background
    "bg":          "#FFFFFF",
    "bg_alt":      "#FAFBFC",     # very subtle grey for shaded panels

    # Comparators
    "spy":         "#1565C0",     # buy-and-hold benchmark blue
    "rgvh":        "#0A2540",
    "baseline":    "#90A4AE",

    # Heatmap / regime palette
    "regime_iv":   "#E0982C",
    "regime_vxn":  "#A92424",
    "regime_curve":"#1565C0",
}


def apply_theme(font_size: int = 10, dpi: int = 200) -> None:
    """Apply the institutional theme to matplotlib globally."""
    mpl.rcParams.update({
        "figure.dpi":          dpi,
        "savefig.dpi":         dpi,
        "savefig.bbox":        "tight",
        "savefig.facecolor":   COLORS["bg"],
        "savefig.edgecolor":   "none",

        "figure.facecolor":    COLORS["bg"],
        "axes.facecolor":      COLORS["bg"],

        # Typography
        "font.family":         "DejaVu Sans",
        "font.size":           font_size,
        "font.weight":         "regular",
        "axes.titlesize":      font_size + 3,
        "axes.titleweight":    "semibold",
        "axes.labelsize":      font_size,
        "axes.labelcolor":     COLORS["ink_soft"],
        "axes.labelweight":    "regular",
        "xtick.labelsize":     font_size - 1,
        "ytick.labelsize":     font_size - 1,
        "xtick.color":         COLORS["ink_soft"],
        "ytick.color":         COLORS["ink_soft"],
        "legend.fontsize":     font_size - 1,

        # Spines / axes
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.spines.left":    False,
        "axes.spines.bottom":  True,
        "axes.edgecolor":      COLORS["ink_muted"],
        "axes.linewidth":      0.8,

        # Gridlines — very subtle, horizontal only
        "axes.grid":           True,
        "axes.grid.axis":      "y",
        "grid.color":          COLORS["muted"],
        "grid.alpha":          0.45,
        "grid.linewidth":      0.6,
        "grid.linestyle":      "-",

        # Ticks
        "xtick.major.size":    3.0,
        "xtick.major.width":   0.6,
        "ytick.major.size":    0.0,    # no y-tick marks (let gridline do it)
        "ytick.major.width":   0,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # Lines
        "lines.linewidth":     2.0,
        "lines.solid_capstyle":"round",

        # Legend
        "legend.frameon":      False,
        "legend.handletextpad":0.7,
        "legend.columnspacing":1.2,
    })


# ---------------------------------------------------------------------------
# Helper: title block (title + subtitle + footer source)
# ---------------------------------------------------------------------------
def title_block(
    fig,
    title: str,
    subtitle: str | None = None,
    source: str | None = None,
    title_fontsize: int = 14,
    subtitle_fontsize: int = 10,
    pad_top: float = 0.04,
):
    """Add a left-aligned title block at the top of a figure plus a footer source."""
    fig.suptitle("", y=1)  # prevent matplotlib's default title from grabbing space
    fig.text(0.04, 1 - pad_top, title,
             ha="left", va="top",
             fontsize=title_fontsize, fontweight="bold",
             color=COLORS["ink"])
    if subtitle:
        fig.text(0.04, 1 - pad_top - 0.045, subtitle,
                 ha="left", va="top",
                 fontsize=subtitle_fontsize, fontweight="regular",
                 color=COLORS["ink_soft"], style="italic")
    if source:
        fig.text(0.04, 0.01, source,
                 ha="left", va="bottom",
                 fontsize=8, color=COLORS["ink_muted"], style="italic")


def callout(ax, x, y, text, color=None, ha="center"):
    """Place an annotation box at (x, y) on an axis."""
    color = color or COLORS["navy"]
    ax.annotate(
        text, xy=(x, y), xytext=(x, y),
        ha=ha, va="center",
        fontsize=8.5, color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.45", fc=COLORS["bg"], ec=color, lw=0.8),
    )


def crisis_marker(ax, date, label, ymin=0, ymax=1, color=None):
    """Vertical line at `date` with a label tag at the top."""
    color = color or COLORS["negative"]
    ax.axvline(date, color=color, lw=0.8, ls="--", alpha=0.55, zorder=1)
    ax.text(date, ymax, f" {label}", ha="left", va="top",
            fontsize=8, color=color, alpha=0.85, fontweight="medium",
            transform=ax.get_xaxis_transform())


# Standardized crisis events for annotation
CRISES = [
    ("2018-02-05", "Volmageddon"),
    ("2020-03-12", "COVID crash"),
    ("2022-06-13", "Fed hike cycle"),
    ("2024-08-05", "Yen carry unwind"),
]


def format_dollars(x, pos):
    """Tick formatter — $1,234 with thousand separators."""
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.1f}k"
    return f"${x:,.0f}"


def format_pct(x, pos):
    return f"{x:.1%}"


def format_bps(x, pos):
    return f"{x*10000:.0f} bps"
