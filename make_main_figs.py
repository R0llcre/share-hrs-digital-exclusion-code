#!/usr/bin/env python
"""
Main-text figures. Values are the published estimates printed in the committed
outputs (SHARE_output.txt: multi-wave pre-trend + before/during blocks + subgroup
block; HRS_output.txt: multi-wave pre-trend block); this script only draws them.
  Figure 1 -> fig1_pretrend.pdf/.png   (wave trajectories, SHARE + HRS panels)
  Figure 2 -> fig2_gradient.pdf/.png   (need gradient)
  Supp     -> fig2_needladder.pdf/.png (M0-M3 ladder, both cohorts)
  Supp     -> figS1_temporal.pdf/.png  (SCS1 vs SCS2)
Run:  venv/bin/python make_main_figs.py   (writes into ../../manuscript/figures_pos/)
"""
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../../manuscript/figures_pos"
BLUE, TEAL, ORANGE = "#1f4e79", "#1d8a7a", "#c0562f"
def r2(v): return str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---- Figure 1: wave trajectories ----
share_any = {"x": [2015, 2019, 2021], "rr": [0.973, 0.968, 0.887],
             "lo": [0.963, 0.958, 0.848], "hi": [0.984, 0.977, 0.928]}
share_4p  = {"x": [2019, 2021], "rr": [0.941, 0.887], "lo": [0.921, 0.848], "hi": [0.962, 0.928]}
hrs = {"x": [2014, 2016, 2018, 2020], "rr": [0.930, 0.918, 0.914, 0.902],
       "lo": [0.917, 0.907, 0.901, 0.888], "hi": [0.944, 0.931, 0.926, 0.917]}

fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=False)

a.axhline(1.0, color="#999999", lw=0.9, ls="--")
a.axvspan(2020.0, 2021.6, color="#f5dede", alpha=0.55, zorder=0)
a.errorbar(share_any["x"], share_any["rr"],
           yerr=[np.array(share_any["rr"]) - np.array(share_any["lo"]),
                 np.array(share_any["hi"]) - np.array(share_any["rr"])],
           fmt="o-", color=BLUE, ms=6, lw=1.6, capsize=3, zorder=3)
a.errorbar(share_4p["x"], share_4p["rr"],
           yerr=[np.array(share_4p["rr"]) - np.array(share_4p["lo"]),
                 np.array(share_4p["hi"]) - np.array(share_4p["rr"])],
           fmt="s--", color=TEAL, ms=6, lw=1.3, capsize=3, mfc="white", zorder=2)
# value labels anchored above the upper whisker (never on markers or lines)
a.annotate("0.97", (2015, 0.984), xytext=(0, 5), textcoords="offset points",
           ha="center", fontsize=9, fontweight="bold", color=BLUE)
a.annotate("0.97", (2019, 0.977), xytext=(0, 5), textcoords="offset points",
           ha="center", fontsize=9, fontweight="bold", color=BLUE)
a.annotate("0.89", (2021, 0.887), xytext=(-26, -4), textcoords="offset points",
           ha="center", fontsize=9, fontweight="bold", color=BLUE)
a.annotate("0.94", (2019, 0.941), xytext=(-14, -4), textcoords="offset points",
           ha="right", fontsize=9, fontweight="bold", color=TEAL)
a.annotate("any visit (pre)", (2015, 0.973), xytext=(14, -14), textcoords="offset points",
           fontsize=8, color=BLUE)
a.annotate("4+ visits (pre, non-ceiling)", (2019, 0.921), xytext=(-12, -14),
           textcoords="offset points", ha="right", fontsize=8, color=TEAL)
a.annotate("in-person (pandemic)", (2021, 0.848), xytext=(-30, -14),
           textcoords="offset points", ha="right", fontsize=8, color=BLUE)
a.set_title("A. SHARE: pandemic-specific in-person contact", fontsize=9.5,
            fontweight="bold", loc="left")
a.set_xticks([2015, 2019, 2021]); a.set_xlim(2013.9, 2021.9)
a.set_ylim(0.82, 1.02); a.set_ylabel("Adjusted RR", fontsize=9)

b.axhline(1.0, color="#999999", lw=0.9, ls="--")
b.axvspan(2019.4, 2020.8, color="#f5dede", alpha=0.55, zorder=0)
b.errorbar(hrs["x"], hrs["rr"],
           yerr=[np.array(hrs["rr"]) - np.array(hrs["lo"]),
                 np.array(hrs["hi"]) - np.array(hrs["rr"])],
           fmt="s-", color=ORANGE, ms=6, lw=1.6, capsize=3, zorder=3)
for x, hi, lab in zip(hrs["x"], hrs["hi"], ["0.93", "0.92", "0.91", "0.90"]):
    b.annotate(lab, (x, hi), xytext=(0, 5), textcoords="offset points",
               ha="center", fontsize=9, fontweight="bold", color=ORANGE)
b.set_title("B. HRS: broad doctor-contact measure", fontsize=9.5,
            fontweight="bold", loc="left")
b.set_xticks(hrs["x"]); b.set_xlim(2013.4, 2020.8)
b.set_ylim(0.82, 1.02); b.set_ylabel("Adjusted RR", fontsize=9)
b.set_xlabel("Survey year", fontsize=9)

for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig1_pretrend.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote fig1_pretrend.pdf/.png")

# ---- Figure 2: need gradient ----
rows = [("By multimorbidity", None, None, None),
        ("None", 0.827, 0.769, 0.889),
        ("2+ conditions", 0.927, 0.892, 0.964),
        ("By age", None, None, None),
        ("50–64 y", 0.819, 0.753, 0.891),
        ("65–79 y", 0.905, 0.863, 0.949),
        ("80+ y", 0.913, 0.865, 0.964)]
OVERALL = 0.887

fig, ax = plt.subplots(figsize=(7.0, 3.4))
ys = np.arange(len(rows))[::-1]
ax.axvline(1.0, color="#999999", lw=0.9, ls="--")
ax.axvline(OVERALL, color=BLUE, lw=1.0, alpha=0.45)
for (lab, r, lo, hi), y in zip(rows, ys):
    if r is None:
        ax.text(0.735, y, lab, fontsize=9, fontweight="bold", va="center")
        continue
    ax.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="o", color=BLUE, ms=5.5,
                capsize=2.5, lw=1.4)
    ax.text(hi + 0.012, y, f"{r:.2f}", va="center", fontsize=9)
ax.text(OVERALL, len(rows) - 0.35, "overall 0.89", fontsize=8, color=BLUE,
        ha="center", va="bottom", style="italic",
        bbox=dict(fc="white", ec="none", pad=1.0))
ax.set_yticks(ys)
ax.set_yticklabels([("" if r is None else lab) for (lab, r, lo, hi) in rows], fontsize=9)
ax.set_xlim(0.72, 1.06)
ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05])
ax.set_xlabel("Adjusted during-pandemic in-person contact RR (non-users vs users), SHARE",
              fontsize=9)
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(left=False, labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig2_gradient.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote fig2_gradient.pdf/.png")

# ---- Supplement figure: need-adjustment ladder (M0-M3), both cohorts ----
lad_x = [0, 1, 2, 3]
lad_lab = ["M0\nbase", "M1\n+health need", "M2\n+function", "M3\n+baseline\nutilisation"]
share_l = {"rr": [0.905, 0.882, 0.887, 0.899], "lo": [0.868, 0.844, 0.848, 0.860], "hi": [0.943, 0.922, 0.928, 0.940]}
hrs_l   = {"rr": [0.914, 0.902, 0.901, 0.927], "lo": [0.900, 0.888, 0.887, 0.913], "hi": [0.929, 0.917, 0.916, 0.942]}
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.axhline(1.0, color="#999999", lw=0.9, ls="--")
for d, col, mk, lab in [(share_l, BLUE, "o", "SHARE (in-person, 2021)"), (hrs_l, ORANGE, "s", "HRS (2020 wave)")]:
    ax.errorbar(lad_x, d["rr"], yerr=[np.array(d["rr"]) - np.array(d["lo"]),
                                      np.array(d["hi"]) - np.array(d["rr"])],
                fmt=f"{mk}-", color=col, ms=5.5, lw=1.5, capsize=3, label=lab)
ax.set_xticks(lad_x); ax.set_xticklabels(lad_lab, fontsize=8.5)
ax.set_ylim(0.83, 1.01); ax.set_ylabel("Adjusted RR (non-users vs users)", fontsize=9)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig2_needladder.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote fig2_needladder.pdf/.png")

# ---- Supplement figure: SCS1 vs SCS2 temporal robustness ----
tem = [("SCS2 (2021)", 0.887, 0.848, 0.928), ("SCS1 (2020)", 0.853, 0.813, 0.896)]
fig, ax = plt.subplots(figsize=(6.2, 1.9))
ax.axvline(1.0, color="#999999", lw=0.9, ls="--")
for (lab, r, lo, hi), y in zip(tem, [0, 1]):
    ax.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="o", color=BLUE, ms=6, capsize=3, lw=1.5)
    ax.text(hi + 0.008, y, f"{r2(r)} ({r2(lo)}–{r2(hi)})", va="center", fontsize=8.5)
ax.set_yticks([0, 1]); ax.set_yticklabels([t[0] for t in tem], fontsize=9)
ax.set_ylim(-0.6, 1.6); ax.set_xlim(0.79, 1.02)
ax.set_xlabel("Adjusted RR of in-person contact (non-users vs users)", fontsize=9)
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(left=False, labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS1_temporal.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote figS1_temporal.pdf/.png")
