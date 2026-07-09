#!/usr/bin/env python
"""
PAPER 2 main-text figures. Values are the committed estimates printed in
PAPER2_FIGDATA.txt (4dp, half-up 2dp labels) and PAPER2_ANALYSIS.txt S3
(timing; labels reproduced verbatim at 3dp, never re-rounded).
  Figure 1 -> fig1_trajectory.pdf/.png  (same-instrument trajectory + flagged
                                         in-person point, panels any / 4+)
  Figure 2 -> fig2_timing.pdf/.png      (fieldwork-timing dose-response)
Run:  venv/bin/python paper2_make_figs.py   (writes into ../../paper2/figures/)
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../../paper2/figures"
os.makedirs(OUT, exist_ok=True)
BLUE, TEAL, ORANGE = "#1f4e79", "#1d8a7a", "#c0562f"
def META(title): return {"Title": title, "Author": "Zihan Zhu, Kairan Shen",
                         "Creator": "", "Producer": "", "CreationDate": None}

# ---------- Figure 1: same-instrument trajectory (PAPER2_FIGDATA.txt) ----------
# hc602 12-month CAPI series, estimated inside the W9 analytic frame
any_s = {"x": [2015, 2019, 2022], "rr": [0.9749, 0.9604, 0.9648],
         "lo": [0.9631, 0.9507, 0.9525], "hi": [0.9869, 0.9703, 0.9772],
         "lab": ["0.97", "0.96", "0.96"]}
fp_s  = {"x": [2015, 2019, 2022], "rr": [1.0106, 0.9429, 0.9449],
         "lo": [0.9861, 0.9225, 0.9305], "hi": [1.0358, 0.9639, 0.9595],
         "lab": ["1.01", "0.94", "0.94"]}
ip = {"x": 2020.9, "rr": 0.8871, "lo": 0.8471, "hi": 0.9290, "lab": "0.89"}

fig, (a, bx) = plt.subplots(2, 1, figsize=(7.0, 6.4))
for ax in (a, bx):
    ax.axhline(1.0, color="#999999", lw=0.9, ls="--")
    ax.axvspan(2020.0, 2021.6, color="#f5dede", alpha=0.55, zorder=0)
    ax.set_xticks([2015, 2019, 2022])
    ax.set_xticklabels(["2015", "2019", "2021–22"])
    ax.set_xlim(2013.9, 2022.9)
    ax.set_ylim(0.80, 1.06)
    ax.set_ylabel("Adjusted RR (non-users vs users)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

a.errorbar(any_s["x"], any_s["rr"],
           yerr=[np.array(any_s["rr"]) - np.array(any_s["lo"]),
                 np.array(any_s["hi"]) - np.array(any_s["rr"])],
           fmt="o-", color=BLUE, ms=6, lw=1.6, capsize=3, zorder=3)
for x, hi, lab in zip(any_s["x"], any_s["hi"], any_s["lab"]):
    a.annotate(lab, (x, hi), xytext=(0, 5), textcoords="offset points",
               ha="center", fontsize=9, fontweight="bold", color=BLUE)
a.errorbar([ip["x"]], [ip["rr"]], yerr=[[ip["rr"] - ip["lo"]], [ip["hi"] - ip["rr"]]],
           fmt="D", color=ORANGE, ms=6.5, capsize=3, mfc="white", zorder=3)
a.annotate(ip["lab"], (ip["x"], ip["lo"]), xytext=(0, -13), textcoords="offset points",
           ha="center", fontsize=9, fontweight="bold", color=ORANGE)
a.annotate("any doctor visit\n(identical 12-month item)", (2015, 0.9749),
           xytext=(10, -26), textcoords="offset points", fontsize=8, color=BLUE)
a.text(ip["x"], 1.052, "in-person visit, summer 2021\n(telephone survey, different instrument)",
       ha="center", va="top", fontsize=8, color=ORANGE)
a.set_title("A. Any doctor visit", fontsize=9.5, fontweight="bold", loc="left")

bx.errorbar(fp_s["x"], fp_s["rr"],
            yerr=[np.array(fp_s["rr"]) - np.array(fp_s["lo"]),
                  np.array(fp_s["hi"]) - np.array(fp_s["rr"])],
            fmt="s-", color=TEAL, ms=6, lw=1.6, capsize=3, zorder=3)
for x, hi, lab in zip(fp_s["x"], fp_s["hi"], fp_s["lab"]):
    bx.annotate(lab, (x, hi), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold", color=TEAL)
bx.annotate("4 or more doctor visits\n(prevalence-matched, non-ceiling margin)",
            (2019, 0.9225), xytext=(0, -26), textcoords="offset points",
            ha="center", fontsize=8, color=TEAL)
bx.set_title("B. Four or more doctor visits", fontsize=9.5, fontweight="bold", loc="left")
bx.set_xlabel("Survey wave (outcome year)", fontsize=9)

plt.tight_layout()
for ext in ("pdf", "png"):
    kw = {"metadata": META("Figure 1")} if ext == "pdf" else {}
    fig.savefig(f"{OUT}/fig1_trajectory.{ext}", dpi=300, bbox_inches="tight", **kw)
plt.close(fig)
print("wrote fig1_trajectory.pdf/.png")

# ---------- Figure 2: fieldwork-timing dose-response (PAPER2_ANALYSIS.txt S3) ----------
Q = ["2021Q4", "2022Q1", "2022Q2", "2022Q3+"]
N = ["11,661", "15,224", "8,635", "1,858"]
rr_q = {"rr": [0.985, 0.955, 0.963, 0.941], "lo": [0.965, 0.931, 0.947, 0.909],
        "hi": [1.006, 0.978, 0.981, 0.974]}
ct_q = {"rr": [1.021, 0.995, 1.005, 0.977], "lo": [0.996, 0.973, 0.980, 0.938],
        "hi": [1.048, 1.018, 1.031, 1.018]}
BASE2019, OVERALL = 0.960, 1.005

fig, (l, r) = plt.subplots(1, 2, figsize=(7.0, 3.3), sharex=True)
xs = np.arange(4)
for ax in (l, r):
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{q}\nn={n}" for q, n in zip(Q, N)], fontsize=7.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    ax.set_xlim(-0.5, 3.95)

l.axhline(1.0, color="#999999", lw=0.9, ls="--")
l.axhline(BASE2019, color=BLUE, lw=1.0, alpha=0.45)
l.errorbar(xs, rr_q["rr"], yerr=[np.array(rr_q["rr"]) - np.array(rr_q["lo"]),
                                 np.array(rr_q["hi"]) - np.array(rr_q["rr"])],
           fmt="o", color=BLUE, ms=5.5, capsize=3, lw=1.4)
for x, hi, v in zip(xs, rr_q["hi"], rr_q["rr"]):
    l.annotate(f"{v:.3f}", (x, hi), xytext=(0, 4), textcoords="offset points",
               ha="center", fontsize=8, color=BLUE)
l.text(3.88, BASE2019, "2019 baseline\n0.960", fontsize=7.5, color=BLUE,
       va="center", ha="right", style="italic",
       bbox=dict(fc="white", ec="none", pad=1.0))
l.set_ylim(0.88, 1.07)
l.set_ylabel("Adjusted 2021–22 RR", fontsize=9)
l.set_title("A. W9 gap by fieldwork quarter", fontsize=9.5, fontweight="bold", loc="left")

r.axhline(1.0, color="#999999", lw=0.9, ls="--")
r.axhline(OVERALL, color=BLUE, lw=1.0, alpha=0.45)
r.errorbar(xs, ct_q["rr"], yerr=[np.array(ct_q["rr"]) - np.array(ct_q["lo"]),
                                 np.array(ct_q["hi"]) - np.array(ct_q["rr"])],
           fmt="s", color=TEAL, ms=5.5, capsize=3, lw=1.4, mfc="white")
for x, hi, v in zip(xs, ct_q["hi"], ct_q["rr"]):
    r.annotate(f"{v:.3f}", (x, hi), xytext=(0, 4), textcoords="offset points",
               ha="center", fontsize=8, color=TEAL)
r.text(3.88, OVERALL, "overall\n1.005", fontsize=7.5, color=BLUE, va="center",
       ha="right", style="italic", bbox=dict(fc="white", ec="none", pad=1.0))
r.set_ylim(0.88, 1.07)
r.set_ylabel("Ratio of RRs, 2021–22 vs 2019", fontsize=9)
r.set_title("B. Period contrast by fieldwork quarter", fontsize=9.5,
            fontweight="bold", loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    kw = {"metadata": META("Figure 2")} if ext == "pdf" else {}
    fig.savefig(f"{OUT}/fig2_timing.{ext}", dpi=300, bbox_inches="tight", **kw)
plt.close(fig)
print("wrote fig2_timing.pdf/.png")
