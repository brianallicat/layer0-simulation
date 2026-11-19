"""Plot helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import genome as gn

TISSUE_COLORS = ["#ffffff", "#c9b38f", "#5a4a3a", "#d7263d", "#1b998b", "#f4d35e", "#3f88c5"]
CMAP = ListedColormap(TISSUE_COLORS)
ORG_STYLE = {"C": ("*", "k"), "A": ("^", "m"), "P": ("v", "b")}


def show(ax, pattern, orgs=None, title=None, crop=None):
    p = pattern
    ax.imshow(p, cmap=CMAP, vmin=0, vmax=6, interpolation="nearest")
    if orgs:
        for name, pts in orgs.items():
            m, c = ORG_STYLE[name]
            for (y, x) in pts:
                ax.plot(x, y, m, color=c, ms=9, mec="w", mew=1.0)
    if crop is not None:
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[1], crop[0])
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def legend(fig):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=TISSUE_COLORS[i], label=gn.TISSUE_NAMES[i]) for i in range(1, 7)]
    h += [Line2D([], [], marker=ORG_STYLE[k][0], color=ORG_STYLE[k][1], ls="", label=f"organizer {k}") for k in "CAP"]
    fig.legend(handles=h, loc="lower center", ncol=9, fontsize=8, frameon=False)
