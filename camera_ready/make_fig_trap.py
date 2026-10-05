"""Camera-ready figure: the trap and its escape (k=50, greedy alert-only).
Numbers are the canonical macOS results (camera_ready/results: key, base_cmp, boot_bulk)."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT = sys.argv[1]
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#444444"})
ORANGE, NAVY, RED, GREEN = "#be8442", "#1f3050", "#a8423a", "#3d7a5a"
cats = ["Static\n(frozen)", "Bulk\nfeedback", "Window\n(AEGIS)", "All alerts,\nno history", "History +\nalerts"]
cold = [14, 98, 10, 46, 79]
real = [113, 120, 93, 74, 130]
x = np.arange(len(cats)); w = 0.38
fig, ax = plt.subplots(figsize=(3.4, 1.85))
b1 = ax.bar(x - w/2, cold, w, color=ORANGE, label="Cold-start base (step 34)", zorder=3)
b2 = ax.bar(x + w/2, real, w, color=NAVY, label="Realistic base (step 42)", zorder=3)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 2, f"{int(r.get_height())}",
                ha="center", va="bottom", fontsize=5.6)
ax.axhline(12.98, color="#8e9aaf", lw=0.8, ls=":", zorder=2)
ax.text(5.02, 15, "random", color="#8e9aaf", fontsize=5.2, ha="right")
ax.set_xlim(-0.6, 5.08)
# bracket over the alert-only categories
ax.annotate("", xy=(1.55, 152), xytext=(4.45, 152),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#444444"))
ax.text(3.0, 155, "alert-only feedback (greedy)", ha="center", fontsize=5.6, color="#444444")
ax.axvline(1.5, color="#bbbbbb", lw=0.5, ls="--", zorder=1)
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=5.8)
ax.set_ylabel("Illicit caught (of 169)", fontsize=6.5)
ax.set_ylim(0, 168); ax.set_yticks([0, 40, 80, 120, 160])
ax.tick_params(labelsize=5.8, width=0.5, length=2)
ax.grid(axis="y", alpha=0.3, lw=0.4, zorder=0)
ax.legend(fontsize=5.4, frameon=False, loc="upper left", borderaxespad=0.2, handlelength=1.2)
fig.tight_layout(pad=0.25)
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=400)
