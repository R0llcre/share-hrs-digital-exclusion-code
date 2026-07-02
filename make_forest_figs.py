#!/usr/bin/env python
"""
Forest figures for the manuscript supplement (recomputed from data, not hard-coded):
  (1) subgroup forest (exploratory effect modification; ADL-only functional strata,
      labelled as such) -> fig3_hetero.pdf/.png
  (2) per-country forest with country-FE pooled and DerSimonian-Laird random-effects
      pooled estimates -> figS_country.pdf/.png
Estimates match share_analysis.py -> SHARE_output.txt. Annotations use half-up rounding.
Run:  venv/bin/python make_forest_figs.py   (writes into ../../manuscript/figures_pos/)
"""
import pyreadstat, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

U, R = "../unzipped", "9-0-0"
OUT = "../../manuscript/figures_pos"
def n2n(s): return s.where(s >= 0, np.nan)
def yn(s):  return n2n(s).map({1: 1.0, 5: 0.0})
def rd(fn, c): return pyreadstat.read_dta(f"{U}/{fn}", usecols=c)[0]
def r2(v):  return str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---- analytic sample (mirrors share_analysis.py) ----
it8 = rd(f"sharew8_rel{R}_it.dta", ["mergeid", "it004_"])
it8["excl"] = np.where(it8["it004_"] == 5, 1.0, np.where(it8["it004_"] == 1, 0.0, np.nan))
cvr, meta = pyreadstat.read_dta(f"{U}/sharew8_rel{R}_cv_r.dta", usecols=["mergeid", "country", "gender", "age2020"])
cvr = cvr.drop_duplicates("mergeid")
CLAB = meta.variable_value_labels.get("country", {})
isc = rd(f"sharew8_rel{R}_gv_isced.dta", ["mergeid", "isced1997_r"])
gh  = rd(f"sharew8_rel{R}_gv_health.dta", ["mergeid", "sphus", "chronicw8", "eurod", "adl", "iadl"])
hc  = rd(f"sharew8_rel{R}_hc.dta", ["mergeid", "hc602_", "hc012_", "hc889_"])
s2  = rd(f"sharew9ca_rel{R}_ca.dta", ["mergeid", "caq120_"])
b = (cvr.merge(it8[["mergeid", "excl"]], on="mergeid").merge(isc, on="mergeid", how="left")
        .merge(gh, on="mergeid", how="left").merge(hc, on="mergeid", how="left"))
b["age"] = n2n(b["age2020"])
b = b[(b["age"] >= 50) & (b["country"] != 25) & b["excl"].notna()].copy()
b["agecat"] = pd.cut(b["age"], [49, 64, 79, 200], labels=["a", "b", "c"])
b["sex"] = n2n(b["gender"]).map({1: "m", 2: "f"})
e = n2n(b["isced1997_r"]).where(lambda s: s <= 6)
b["edu3"] = pd.cut(e, [-1, 2, 4, 6], labels=["lo", "mi", "hi"])
for c in ["sphus", "chronicw8", "eurod", "adl", "iadl"]: b[c] = n2n(b[c])
b["docv8"] = n2n(b["hc602_"]); b["hosp8"] = yn(b["hc012_"])
b["lowlit"] = (n2n(b["hc889_"]) <= 2).astype(float); b.loc[n2n(b["hc889_"]).isna(), "lowlit"] = np.nan
b["cc"] = b["country"].astype(int).astype(str)
d = b.merge(s2, on="mergeid", how="inner"); d["visit"] = yn(d["caq120_"])
DROP = ["visit","excl","agecat","sex","edu3","sphus","chronicw8","eurod","adl","iadl","docv8","hosp8","cc"]
dd = d.dropna(subset=DROP).copy()
print("analytic N =", len(dd))

full  = "C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl"
noage = "C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl"
noedu = "C(agecat)+C(sex)+sphus+chronicw8+eurod+adl+iadl"
nosex = "C(agecat)+C(edu3)+sphus+chronicw8+eurod+adl+iadl"
def rrc(x, cov):
    m = smf.glm(f"visit ~ excl + {cov}+C(cc)", data=x, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": x["cc"]})
    ci = m.conf_int().loc["excl"]
    return float(np.exp(m.params["excl"])), float(np.exp(ci[0])), float(np.exp(ci[1]))

# ---- Figure: subgroup forest ----
dl = dd.dropna(subset=["lowlit"])
GROUPS = [
    ("Overall", dd, full),
    ("Age 50–64", dd[dd.agecat == "a"], noage), ("Age 65–79", dd[dd.agecat == "b"], noage),
    ("Age 80+", dd[dd.agecat == "c"], noage),
    ("Women", dd[dd.sex == "f"], nosex), ("Men", dd[dd.sex == "m"], nosex),
    ("Low education", dd[dd.edu3 == "lo"], noedu), ("High education", dd[dd.edu3 == "hi"], noedu),
    ("Not multimorbid", dd[dd.chronicw8 < 2], full), ("Multimorbid (2+)", dd[dd.chronicw8 >= 2], full),
    ("No ADL limitation", dd[dd.adl == 0], full), ("Any ADL limitation", dd[dd.adl > 0], full),
    ("Higher health literacy", dl[dl.lowlit == 0], full), ("Low health literacy", dl[dl.lowlit == 1], full),
]
rows = []
for lab, x, cov in GROUPS:
    r, lo, hi = rrc(x, cov)
    rows.append((lab, r, lo, hi, len(x)))
    print(f"  {lab:<24} RR {r:.3f} [{lo:.3f},{hi:.3f}] N={len(x):,}")

fig, ax = plt.subplots(figsize=(8.4, 6.2))
ys = np.arange(len(rows))[::-1]
for (lab, r, lo, hi, n), y in zip(rows, ys):
    bold = lab == "Overall"
    ax.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="s" if bold else "o",
                color="#1f4e79" if bold else "#4a4a4a", ms=7 if bold else 5, capsize=2.5, lw=1.4)
    ax.text(1.24, y, f"{r2(r)} ({r2(lo)}–{r2(hi)})", va="center", fontsize=8.5)
    ax.text(1.78, y, f"{n:,}", va="center", fontsize=8.5, ha="right")
ax.axvline(1.0, color="#999999", lw=0.9, ls="--")
ax.set_yticks(ys); ax.set_yticklabels([g[0] for g in rows], fontsize=9)
ax.set_xlim(0.62, 1.80); ax.set_xlabel("Adjusted RR (non-users vs users), in-person visit, SHARE 2021", fontsize=9)
ax.text(1.24, len(rows) + 0.1, "RR (95% CI)", fontsize=8.5, fontweight="bold")
ax.text(1.78, len(rows) + 0.1, "N", fontsize=8.5, fontweight="bold", ha="right")
ax.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1])
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(left=False)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig3_hetero.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote fig3_hetero.pdf/.png")

# ---- Figure: per-country forest + pooled ----
crows = []
for c in sorted(dd.cc.unique(), key=int):
    x = dd[dd.cc == c]
    if x.excl.nunique() < 2 or len(x) < 150: continue
    m = smf.glm(f"visit ~ excl + {full}", data=x, family=sm.families.Poisson()).fit()
    name = CLAB.get(float(c), CLAB.get(int(c), f"Country {c}"))
    crows.append((str(name), float(m.params["excl"]), float(m.bse["excl"]), len(x)))
crows.sort(key=lambda t: t[1])
th = np.array([r[1] for r in crows]); se = np.array([r[2] for r in crows]); w = 1 / se**2
k = len(crows)
thbar = (w * th).sum() / w.sum(); Q = (w * (th - thbar) ** 2).sum()
tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
wre = 1 / (se**2 + tau2)
thre = (wre * th).sum() / wre.sum(); sere = np.sqrt(1 / wre.sum())
fe = rrc(dd, full)   # country-FE pooled (the focal model)
re = (float(np.exp(thre)), float(np.exp(thre - 1.96 * sere)), float(np.exp(thre + 1.96 * sere)))
print(f"  country-FE pooled RR {fe[0]:.3f} [{fe[1]:.3f},{fe[2]:.3f}]")
print(f"  random-effects pooled RR {re[0]:.3f} [{re[1]:.3f},{re[2]:.3f}] (tau^2={tau2:.4f})")

fig, ax = plt.subplots(figsize=(8.4, 8.6))
ys = np.arange(k + 2)[::-1]
for (name, t, s, n), y in zip(crows, ys[:k]):
    r, lo, hi = np.exp(t), np.exp(t - 1.96 * s), np.exp(t + 1.96 * s)
    ax.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="o", color="#4a4a4a", ms=4, capsize=2, lw=1.1)
    ax.text(2.55, y, f"{r2(r)} ({r2(lo)}–{r2(hi)})", va="center", fontsize=8)
    ax.text(5.6, y, f"{n:,}", va="center", fontsize=8, ha="right")
for lab, (r, lo, hi), y, col in [("Pooled (country fixed effects)", fe, ys[k], "#1f4e79"),
                                 ("Pooled (random effects)", re, ys[k + 1], "#8b1a1a")]:
    ax.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="D", color=col, ms=7, capsize=3, lw=1.6)
    ax.text(2.55, y, f"{r2(r)} ({r2(lo)}–{r2(hi)})", va="center", fontsize=8, fontweight="bold")
ax.axvline(1.0, color="#999999", lw=0.9, ls="--")
labels = [c[0] for c in crows] + ["Pooled (country FE)", "Pooled (random effects)"]
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xscale("log"); ax.set_xlim(0.35, 5.8)
ax.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0]); ax.set_xticklabels(["0.5", "0.7", "1.0", "1.4", "2.0"], fontsize=8.5)
from matplotlib.ticker import NullFormatter, NullLocator
ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Adjusted RR (non-users vs users), in-person visit, SHARE 2021 (log scale)", fontsize=9)
ax.text(2.55, k + 2.2, "RR (95% CI)", fontsize=8, fontweight="bold")
ax.text(5.6, k + 2.2, "N", fontsize=8, fontweight="bold", ha="right")
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(left=False)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS_country.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote figS_country.pdf/.png")

# ---- Figure: leave-one-country-out ----
loco = []
for c in sorted(dd.cc.unique(), key=int):
    sub = dd[dd.cc != c]
    r, lo, hi = rrc(sub, full)
    name = CLAB.get(float(c), CLAB.get(int(c), f"Country {c}"))
    loco.append((str(name), r))
loco.sort(key=lambda t: t[1], reverse=True)
full_rr = rrc(dd, full)[0]
print(f"  LOCO range {min(r for _, r in loco):.3f}--{max(r for _, r in loco):.3f}")

BLUE_ = "#1f4e79"
fig, ax = plt.subplots(figsize=(6.6, 7.2))
ys = np.arange(len(loco))[::-1]
ax.axvline(full_rr, color=BLUE_, lw=1.1)
for (name, r), y in zip(loco, ys):
    ax.plot(r, y, "o", color="#4a4a4a", ms=4.5)
ax.set_yticks(ys); ax.set_yticklabels([n for n, _ in loco], fontsize=8.5)
ax.set_xlim(0.855, 0.925)
ax.set_xlabel("Pooled focal RR with that country removed", fontsize=9)
ax.text(full_rr, len(loco) + 0.1, f"full-sample RR {full_rr:.3f}", fontsize=8,
        color=BLUE_, ha="center", va="bottom",
        bbox=dict(fc="white", ec="none", pad=1.0))
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(left=False, labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS2_loco.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote figS2_loco.pdf/.png")
