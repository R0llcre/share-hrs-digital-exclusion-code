#!/usr/bin/env python
"""
PAPER 2 MAIN ANALYSIS: "Transient widening, durable gradient" (SHARE W9).
Implements the must-do package of paper2/NARRATIVE_SPEC.md section 4
(everything except the wild cluster bootstrap -> paper2_bootstrap.py):

  S1  core estimates: W9 RRs, like-for-like W8->W9 contrasts, period-specific
      RRs from the fully-interacted model (incl. survivor-sample 2019 RR)
  S2  TOST equivalence against pre-specified margins
  S3  interview-timing dose-response (int_year/int_month, W9 cv_r)
  S4  retention/mortality package: status decomposition, mortality RR by
      exposure, IPW for W9 retention, Manski-style extreme bounds
  S6  exposure-drift sensitivity: stable-exposure groups, 4-group table
  S7  country heterogeneity: per-country RRs, DL random effects, LOCO
  S8  SCS2 instrument bridge: caq120 in-person gap INSIDE the W9 sample
  S9  threshold ladder 1+/2+/3+/4+/6+
  S10 cciw_w9 calibrated-weight sensitivity

Run from this directory:  venv/bin/python paper2_analysis.py
Writes PAPER2_ANALYSIS.txt to the current directory.
"""
import pyreadstat, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
import patsy
from scipy import stats
np.seterr(all="ignore")

U, R = "../unzipped", "9-0-0"
OUT = []
def P(s=""):
    print(s, flush=True); OUT.append(s)
def n2n(s): return s.where(s >= 0, np.nan)
def yn(s):  return n2n(s).map({1: 1.0, 5: 0.0})
def rd(fn, c): return pyreadstat.read_dta(f"{U}/{fn}", usecols=c)[0]

# ================= base: identical to paper2_feasibility.py =================
it8 = rd(f"sharew8_rel{R}_it.dta", ["mergeid", "it004_"])
assert it8["mergeid"].is_unique
it8["excl"] = np.where(it8["it004_"] == 5, 1.0, np.where(it8["it004_"] == 1, 0.0, np.nan))
cvr = rd(f"sharew8_rel{R}_cv_r.dta", ["mergeid", "country", "gender", "age2020"]).drop_duplicates("mergeid")
isc = rd(f"sharew8_rel{R}_gv_isced.dta", ["mergeid", "isced1997_r"])
gh  = rd(f"sharew8_rel{R}_gv_health.dta", ["mergeid", "sphus", "chronicw8", "eurod", "adl", "iadl"])
hc8 = rd(f"sharew8_rel{R}_hc.dta", ["mergeid", "hc602_", "hc012_"])
for df in (isc, gh, hc8):
    assert df["mergeid"].is_unique
b = (cvr.merge(it8[["mergeid", "excl"]], on="mergeid").merge(isc, on="mergeid", how="left")
        .merge(gh, on="mergeid", how="left").merge(hc8, on="mergeid", how="left"))
b["age"] = n2n(b["age2020"])
b = b[(b["age"] >= 50) & (b["country"] != 25) & b["excl"].notna()].copy()
b["agecat"] = pd.cut(b["age"], [49, 64, 79, 200], labels=["a", "b", "c"])
b["sex"] = n2n(b["gender"]).map({1: "m", 2: "f"})
e = n2n(b["isced1997_r"]).where(lambda s: s <= 6)
b["edu3"] = pd.cut(e, [-1, 2, 4, 6], labels=["lo", "mi", "hi"])
for c in ["sphus", "chronicw8", "eurod", "adl", "iadl"]:
    b[c] = n2n(b[c])
b["docv8"] = n2n(b["hc602_"]); b["hosp8"] = yn(b["hc012_"])
b["cc"] = b["country"].astype(int).astype(str)
N_expo = len(b)

hc9 = rd(f"sharew9_rel{R}_hc.dta", ["mergeid", "hc602_"]).rename(columns={"hc602_": "hc602w9"})
it9 = rd(f"sharew9_rel{R}_it.dta", ["mergeid", "it004_"]).rename(columns={"it004_": "it004w9"})
cv9 = rd(f"sharew9_rel{R}_cv_r.dta", ["mergeid", "interview", "deceased", "int_year", "int_month"]).drop_duplicates("mergeid")
xt9 = rd(f"sharew9_rel{R}_xt.dta", ["mergeid", "xt009_"]).drop_duplicates("mergeid")
for df in (hc9, it9):
    assert df["mergeid"].is_unique

d = b.merge(hc9, on="mergeid", how="inner")
N_w9 = len(d)
d["docv9"] = n2n(d["hc602w9"])
d["v9_any"] = np.where(d["docv9"] >= 1, 1.0, np.where(d["docv9"] >= 0, 0.0, np.nan))
d["v9_4p"]  = np.where(d["docv9"] >= 4, 1.0, np.where(d["docv9"] >= 0, 0.0, np.nan))
DROP = ["v9_any", "excl", "agecat", "sex", "edu3", "sphus", "chronicw8",
        "eurod", "adl", "iadl", "docv8", "hosp8", "cc"]
dd = d.dropna(subset=DROP).copy()
dd = dd.merge(it9, on="mergeid", how="left").merge(
    cv9[["mergeid", "int_year", "int_month"]], on="mergeid", how="left")
dd["pre_any"] = np.where(dd["docv8"] >= 1, 1.0, 0.0)
dd["pre_4"]   = np.where(dd["docv8"] >= 4, 1.0, 0.0)

fM2 = "C(agecat)+C(sex)+C(edu3)+C(cc)+sphus+chronicw8+eurod+adl+iadl"
COVCOLS = ["excl", "agecat", "sex", "edu3", "sphus", "chronicw8", "eurod", "adl", "iadl", "cc"]

def rr(col, x, formula_rhs=None, weights=None):
    """Modified-Poisson RR for excl, country-clustered SE. Returns (rr, lo, hi, n)."""
    sub = x.dropna(subset=[col]).copy()
    rhs = formula_rhs or fM2
    kw = {}
    if weights is not None:
        kw["var_weights"] = sub[weights]
    m = smf.glm(f"{col} ~ excl + {rhs}", data=sub, family=sm.families.Poisson(), **kw
                ).fit(cov_type="cluster", cov_kwds={"groups": sub["cc"]})
    ci = m.conf_int().loc["excl"]
    return np.exp(m.params["excl"]), np.exp(ci[0]), np.exp(ci[1]), int(m.nobs if weights is None else len(sub))

def stacked_fit(x, precol, postcol, weights=None):
    """Fully-interacted stacked Poisson. Returns dict with contrast theta,
    period-specific RRs (pre = survivor-sample baseline RR), se_log, n."""
    sub = x.dropna(subset=[precol, postcol] + ([] if weights is None else [weights]))
    lo = pd.concat([sub[COVCOLS + [precol] + ([weights] if weights else [])].rename(columns={precol: "y"}).assign(per=0.0),
                    sub[COVCOLS + [postcol] + ([weights] if weights else [])].rename(columns={postcol: "y"}).assign(per=1.0)],
                   ignore_index=True)
    X = patsy.dmatrix("excl*per + (C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl+C(cc))*per",
                      lo, return_type="dataframe")
    kw = {} if weights is None else {"var_weights": lo[weights].values}
    m = sm.GLM(lo["y"].astype(float).values, X.values, family=sm.families.Poisson(), **kw).fit(
        cov_type="cluster", cov_kwds={"groups": lo["cc"].values})
    cols = list(X.columns)
    je, ji = cols.index("excl"), cols.index("excl:per")
    V = m.cov_params()
    th, se = np.exp(m.params[ji]), np.sqrt(V[ji, ji])
    pre = np.exp(m.params[je]); pre_se = np.sqrt(V[je, je])
    post_lb = m.params[je] + m.params[ji]
    post_se = np.sqrt(V[je, je] + V[ji, ji] + 2 * V[je, ji])
    z = stats.norm.ppf(0.975)
    return {"theta": th, "lo": np.exp(np.log(th) - z * se), "hi": np.exp(np.log(th) + z * se),
            "p": m.pvalues[ji], "se_log": se,
            "pre": pre, "pre_lo": np.exp(np.log(pre) - z * pre_se), "pre_hi": np.exp(np.log(pre) + z * pre_se),
            "post": np.exp(post_lb), "post_lo": np.exp(post_lb - z * post_se), "post_hi": np.exp(post_lb + z * post_se),
            "n": len(sub)}

P("=" * 96)
P("PAPER 2 MAIN ANALYSIS -- SHARE W9 like-for-like reversion (run 2026-07-02)")
P("=" * 96)
P(f"S0 FLOW: W8 exposure-known 50+ (excl. Israel) {N_expo:,} -> appears in W9 hc module {N_w9:,}")
P(f"        -> covariate-complete analytic sample {len(dd):,}")
n_noout = int(d["v9_any"].isna().sum())
n_cov = N_w9 - n_noout - len(dd)
P(f"   complete-case decomposition of the {N_w9 - len(dd):,} excluded: "
  f"{n_noout:,} missing the W9 outcome; {n_cov:,} outcome present but missing a focal covariate")
P(f"   4+ prevalence check: W9 {dd['v9_4p'].mean()*100:.1f}% vs W8-in-sample {(dd['docv8']>=4).mean()*100:.1f}% (matched baselines)")
P(f"   any-visit:           W9 {dd['v9_any'].mean()*100:.1f}% vs W8-in-sample {(dd['docv8']>=1).mean()*100:.1f}%")

# ================= S1 core + S5 survivor-sample pre RR ======================
P(); P("== S1 CORE ESTIMATES ==")
for col, lab in [("v9_any", "W9 any doctor visit "), ("v9_4p", "W9 4+ doctor visits")]:
    r = rr(col, dd)
    P(f"  cross-sectional {lab}: RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] (n={r[3]:,})")
RES = {}
for precol, postcol, lab in [("pre_any", "v9_any", "any-visit"), ("pre_4", "v9_4p", "4+-visit ")]:
    s = stacked_fit(dd, precol, postcol)
    RES[postcol] = s
    P(f"  {lab} fully-interacted model (n={s['n']:,}):")
    P(f"    2019 RR within W9 survivors (S5): {s['pre']:.3f} [{s['pre_lo']:.3f},{s['pre_hi']:.3f}]"
      f"   <- companion (n=30,244, SCS2-conditioned) anchors: 0.968 / 0.941")
    P(f"    2021-22 RR (same model):          {s['post']:.3f} [{s['post_lo']:.3f},{s['post_hi']:.3f}]")
    P(f"    ratio-of-RR (period contrast):    {s['theta']:.3f} [{s['lo']:.3f},{s['hi']:.3f}] p={s['p']:.2f}")

# ================= S2 TOST equivalence ======================================
P(); P("== S2 EQUIVALENCE (one-sided z-tests on log ratio-of-RR; H0: theta <= margin) ==")
P("   margins: 0.92/0.94 = full persistence of the IN-PERSON pandemic contrasts;")
P("   0.959/0.969 = half persistence (log scale); 0.960 = full persistence of the")
P("   TOTAL-CONTACT-implied widening (0.93/0.968); 0.980 = half of that (primary).")
MARGINS = {"v9_any": [0.92, 0.959, 0.960, 0.980], "v9_4p": [0.94, 0.969, 0.980]}
for postcol, lab in [("v9_any", "any-visit"), ("v9_4p", "4+-visit ")]:
    s = RES[postcol]
    for mgn in MARGINS[postcol]:
        zst = (np.log(s["theta"]) - np.log(mgn)) / s["se_log"]
        p1 = 1 - stats.norm.cdf(zst)
        P(f"  {lab}: H0 theta<={mgn:.3f} rejected? z={zst:5.2f}, one-sided p={p1:.2e}")
    bound95 = np.exp(np.log(s["theta"]) - 1.645 * s["se_log"])
    P(f"  {lab}: one-sided 95% lower bound on theta = {bound95:.3f} (equivalence boundary)")

# ================= S3 interview-timing dose-response ========================
P(); P("== S3 INTERVIEW-TIMING DOSE-RESPONSE (W9 fieldwork Oct 2021 - late 2022) ==")
dd["iy"] = n2n(dd["int_year"]); dd["im"] = n2n(dd["int_month"])
dd["ms"] = (dd["iy"] - 2021) * 12 + (dd["im"] - 10)   # months since Oct 2021
known = dd.dropna(subset=["ms"])
P(f"  interview date known: {len(known):,}/{len(dd):,}; months-since-Oct-2021: "
  f"p10 {known['ms'].quantile(.1):.0f}, median {known['ms'].median():.0f}, p90 {known['ms'].quantile(.9):.0f}")
qcut = pd.cut(known["ms"], [-1, 2, 5, 8, 24], labels=["2021Q4", "2022Q1", "2022Q2", "2022Q3+"])
for q in ["2021Q4", "2022Q1", "2022Q2", "2022Q3+"]:
    sub = known[qcut == q]
    if len(sub) < 800:
        P(f"  {q}: n={len(sub):,} -- too small, skipped"); continue
    r = rr("v9_any", sub)
    s = stacked_fit(sub, "pre_any", "v9_any")
    P(f"  {q}: n={len(sub):,} ({sub['cc'].nunique()} countries) | W9 any RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]"
      f" | contrast {s['theta']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]")
half = {"early(Oct21-Mar22)": known[known["ms"] <= 5], "late(Apr22+)      ": known[known["ms"] >= 6]}
for lab, sub in half.items():
    r = rr("v9_any", sub); r4 = rr("v9_4p", sub)
    s = stacked_fit(sub, "pre_any", "v9_any"); s4 = stacked_fit(sub, "pre_4", "v9_4p")
    P(f"  {lab}: n={len(sub):,} | any RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] contrast {s['theta']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]"
      f" | 4+ RR {r4[0]:.3f} contrast {s4['theta']:.3f} [{s4['lo']:.3f},{s4['hi']:.3f}]")
known = known.assign(ms_c=(known["ms"] - known["ms"].mean()) / 6.0)
m = smf.glm(f"v9_any ~ excl*ms_c + {fM2}", data=known.dropna(subset=["v9_any"]),
            family=sm.families.Poisson()).fit(cov_type="cluster",
            cov_kwds={"groups": known.dropna(subset=["v9_any"])["cc"]})
ci = m.conf_int().loc["excl:ms_c"]
P(f"  continuous: excl x (per 6 months of fieldwork) RR-ratio {np.exp(m.params['excl:ms_c']):.3f}"
  f" [{np.exp(ci[0]):.3f},{np.exp(ci[1]):.3f}] p={m.pvalues['excl:ms_c']:.2f}")
P("  NOTE: timing strata are country-confounded (staggered fieldwork); country FE kept within strata.")

# ================= S4 retention / mortality package =========================
P(); P("== S4 RETENTION AND MORTALITY (base: W8-covariate-complete pool) ==")
CC8 = ["excl", "agecat", "sex", "edu3", "sphus", "chronicw8", "eurod", "adl", "iadl", "docv8", "hosp8", "cc"]
bb = b.dropna(subset=CC8).copy()
bb = bb.merge(cv9, on="mergeid", how="left").merge(xt9, on="mergeid", how="left")
bb["analytic"] = bb["mergeid"].isin(dd["mergeid"]).astype(float)
bb["dead"] = np.where((bb["deceased"] == 1) | (bb["interview"] == 2), 1.0, 0.0)
bb["status"] = np.select(
    [bb["analytic"] == 1, bb["dead"] == 1, bb["interview"].isin([0, 1])],
    ["analytic", "deceased", "alive-nonresponse/incomplete"],
    default="not-in-W9-coverscreen")
P(f"  pool n={len(bb):,} (W8 covariate-complete out of {N_expo:,})")
tab = bb.groupby(["status"])["excl"].agg(["size"]).join(
    bb.groupby("status").apply(lambda g: pd.Series({
        "nonuser%": g.loc[g.excl == 1].shape[0] / max(bb[bb.excl == 1].shape[0], 1) * 100,
        "user%":    g.loc[g.excl == 0].shape[0] / max(bb[bb.excl == 0].shape[0], 1) * 100})))
for st, row in tab.iterrows():
    P(f"  {st:<30} n={int(row['size']):>6,}  among non-users {row['nonuser%']:5.1f}%  among users {row['user%']:5.1f}%")
mort = rr("dead", bb)
P(f"  adjusted mortality RR (non-user vs user, M2): {mort[0]:.3f} [{mort[1]:.3f},{mort[2]:.3f}]")
# genuine survivor-test comparator: 2019 RRs on the FULL covariate-complete W8 pool
bb["pre_any"] = np.where(bb["docv8"] >= 1, 1.0, 0.0)
bb["pre_4"]   = np.where(bb["docv8"] >= 4, 1.0, 0.0)
for col, lab in [("pre_any", "2019 any doctor visit "), ("pre_4", "2019 4+ doctor visits")]:
    r = rr(col, bb)
    P(f"  {lab} RR on the FULL W8 covariate-complete pool (n={r[3]:,}): {r[0]:.4f} [{r[1]:.4f},{r[2]:.4f}]")
yr = bb[bb.dead == 1].copy(); yr["dyr"] = n2n(yr["xt009_"])
if yr["dyr"].notna().sum() > 100:
    t = yr.groupby(yr["dyr"].clip(2019, 2023))["excl"].agg(["size", "mean"])
    P("  deaths by year (xt): " + "; ".join(f"{int(y)}: n={int(r['size'])}, non-user {r['mean']*100:.0f}%"
                                            for y, r in t.iterrows()))
# IPW for retention
ret = smf.glm(f"analytic ~ excl + {fM2} + docv8 + hosp8", data=bb,
              family=sm.families.Binomial()).fit()
bb["p_ret"] = ret.predict(bb)
an = bb[bb["analytic"] == 1].copy()
an["ipw"] = (an["p_ret"].mean() / an["p_ret"]).clip(
    lower=(an["p_ret"].mean() / an["p_ret"]).quantile(0.01),
    upper=(an["p_ret"].mean() / an["p_ret"]).quantile(0.99))
ddw = dd.merge(an[["mergeid", "ipw"]], on="mergeid", how="inner")
P(f"  IPW distribution: mean {ddw['ipw'].mean():.2f}, median {ddw['ipw'].median():.2f}, max {ddw['ipw'].max():.2f}")
for col, pre, lab in [("v9_any", "pre_any", "any-visit"), ("v9_4p", "pre_4", "4+-visit ")]:
    r = rr(col, ddw, weights="ipw")
    s = stacked_fit(ddw, pre, col, weights="ipw")
    P(f"  IPW-weighted {lab}: W9 RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] | contrast {s['theta']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]")
# extreme (Manski-style) bounds on the RAW W9 any-visit RR
lost = bb[(bb["analytic"] == 0)]
obs_nu, obs_u = dd[dd.excl == 1], dd[dd.excl == 0]
for scen, nu_y, u_y, universe in [
        ("alive lost only: non-users 0 visits, users all visit (worst for reversion)", 0, 1, lost[lost.dead == 0]),
        ("alive lost only: non-users all visit, users 0 visits (best)",                1, 0, lost[lost.dead == 0]),
        ("alive lost + deceased: non-users 0, users 1 (most extreme)",                 0, 1, lost)]:
    n_nu = universe[universe.excl == 1].shape[0]; n_u = universe[universe.excl == 0].shape[0]
    p_nu = (obs_nu["v9_any"].sum() + nu_y * n_nu) / (len(obs_nu) + n_nu)
    p_u  = (obs_u["v9_any"].sum() + u_y * n_u) / (len(obs_u) + n_u)
    P(f"  extreme bound [{scen}]: raw W9 any-visit RR = {p_nu/p_u:.3f}")
P("  (observed raw W9 any-visit RR = "
  f"{obs_nu['v9_any'].mean()/obs_u['v9_any'].mean():.3f}; bounds are diagnostic extremes, not estimates)")

# ================= S6 exposure drift ========================================
P(); P("== S6 EXPOSURE DRIFT (W8 -> W9 internet transitions inside analytic sample) ==")
dd["excl9"] = np.where(dd["it004w9"] == 5, 1.0, np.where(dd["it004w9"] == 1, 0.0, np.nan))
dk = dd.dropna(subset=["excl9"]).copy()
grp = np.select([(dk.excl == 0) & (dk.excl9 == 0), (dk.excl == 1) & (dk.excl9 == 1),
                 (dk.excl == 1) & (dk.excl9 == 0), (dk.excl == 0) & (dk.excl9 == 1)],
                ["stable-user", "stable-nonuser", "catch-up", "drop-off"], default="?")
dk["grp"] = pd.Categorical(grp, categories=["stable-user", "stable-nonuser", "catch-up", "drop-off"])
P(f"  {'group':<16}{'n':>8}{'any%':>8}{'4+%':>8}   adjusted RR vs stable-user (any | 4+)")
f_grp = "C(grp)+C(agecat)+C(sex)+C(edu3)+C(cc)+sphus+chronicw8+eurod+adl+iadl"
ma = smf.glm(f"v9_any ~ {f_grp}", data=dk.dropna(subset=['v9_any']), family=sm.families.Poisson()
             ).fit(cov_type="cluster", cov_kwds={"groups": dk.dropna(subset=['v9_any'])["cc"]})
m4 = smf.glm(f"v9_4p ~ {f_grp}", data=dk.dropna(subset=['v9_4p']), family=sm.families.Poisson()
             ).fit(cov_type="cluster", cov_kwds={"groups": dk.dropna(subset=['v9_4p'])["cc"]})
for g in ["stable-user", "stable-nonuser", "catch-up", "drop-off"]:
    sub = dk[dk.grp == g]
    if g == "stable-user":
        ra = r4 = "1.00 (ref)"
    else:
        k = f"C(grp)[T.{g}]"
        ca, c4 = ma.conf_int().loc[k], m4.conf_int().loc[k]
        ra = f"{np.exp(ma.params[k]):.3f} [{np.exp(ca[0]):.3f},{np.exp(ca[1]):.3f}]"
        r4 = f"{np.exp(m4.params[k]):.3f} [{np.exp(c4[0]):.3f},{np.exp(c4[1]):.3f}]"
    P(f"  {g:<16}{len(sub):>8,}{sub['v9_any'].mean()*100:>8.1f}{sub['v9_4p'].mean()*100:>8.1f}   {ra} | {r4}")
st = dk[dk.grp.isin(["stable-user", "stable-nonuser"])]
r = rr("v9_any", st); r4 = rr("v9_4p", st)
s = stacked_fit(st, "pre_any", "v9_any"); s4 = stacked_fit(st, "pre_4", "v9_4p")
P(f"  stable-exposure only (undiluted): any RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}], 4+ RR {r4[0]:.3f} [{r4[1]:.3f},{r4[2]:.3f}]")
P(f"    contrasts: any {s['theta']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]; 4+ {s4['theta']:.3f} [{s4['lo']:.3f},{s4['hi']:.3f}]")

# ================= S7 country heterogeneity + LOCO ==========================
P(); P("== S7 COUNTRY HETEROGENEITY (W9 any-visit RR) ==")
f_nocc = "C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl"
logs, ses, ccs = [], [], []
for c in sorted(dd["cc"].unique(), key=int):
    sub = dd[(dd.cc == c)].dropna(subset=["v9_any"])
    try:
        m = smf.glm(f"v9_any ~ excl + {f_nocc}", data=sub, family=sm.families.Poisson()).fit(cov_type="HC1")
        lg, se = m.params["excl"], m.bse["excl"]
        if np.isfinite(lg) and np.isfinite(se) and se < 1:
            logs.append(lg); ses.append(se); ccs.append(c)
    except Exception:
        P(f"  country {c}: model failed, skipped")
logs, ses = np.array(logs), np.array(ses)
w_fe = 1 / ses**2
mu_fe = (w_fe * logs).sum() / w_fe.sum()
Q = (w_fe * (logs - mu_fe)**2).sum(); dfq = len(logs) - 1
I2 = max(0.0, (Q - dfq) / Q) * 100
tau2 = max(0.0, (Q - dfq) / (w_fe.sum() - (w_fe**2).sum() / w_fe.sum()))
w_re = 1 / (ses**2 + tau2)
mu_re = (w_re * logs).sum() / w_re.sum(); se_re = np.sqrt(1 / w_re.sum())
P(f"  {len(logs)} countries | below 1: {(np.exp(logs) < 1).sum()} | range {np.exp(logs.min()):.2f}-{np.exp(logs.max()):.2f}")
P(f"  FE pooled {np.exp(mu_fe):.3f}; DL-RE pooled {np.exp(mu_re):.3f} "
  f"[{np.exp(mu_re-1.96*se_re):.3f},{np.exp(mu_re+1.96*se_re):.3f}]; Q={Q:.1f} (df={dfq}, p={1-stats.chi2.cdf(Q,dfq):.2f}); I2={I2:.0f}%; tau2={tau2:.4f}")
P("  LOCO on the any-visit period contrast (26 refits):")
locos = []
for c in sorted(dd["cc"].unique(), key=int):
    s = stacked_fit(dd[dd.cc != c], "pre_any", "v9_any")
    locos.append((c, s["theta"]))
ths = [t for _, t in locos]
cmin = min(locos, key=lambda x: x[1]); cmax = max(locos, key=lambda x: x[1])
P(f"  LOCO range {min(ths):.3f} (drop {cmin[0]}) - {max(ths):.3f} (drop {cmax[0]})  [full {RES['v9_any']['theta']:.3f}]")

# ================= S8 SCS2 instrument bridge ================================
P(); P("== S8 SCS2 BRIDGE: in-person gap re-estimated INSIDE the W9 analytic sample ==")
ca2 = rd(f"sharew9ca_rel{R}_ca.dta", ["mergeid", "caq120_", "caq118_"]).drop_duplicates("mergeid")
br = dd.merge(ca2, on="mergeid", how="inner")
br["visit_ip"] = yn(br["caq120_"])
br["remote_n"] = n2n(br["caq118_"])
br["total_c"] = np.where((br["visit_ip"] == 1) | (br["remote_n"] >= 1), 1.0,
                         np.where(br["visit_ip"].notna() & br["remote_n"].notna(), 0.0, np.nan))
r = rr("visit_ip", br)
P(f"  SCS2 (summer 2021) in-person visit RR within overlap (n={r[3]:,}): {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]"
  f"   <- companion full-sample 0.887")
rt = rr("total_c", br)
P(f"  SCS2 total contact (in-person or remote) RR within overlap (n={rt[3]:,}): {rt[0]:.3f} [{rt[1]:.3f},{rt[2]:.3f}]"
  f"   <- companion full-sample 0.93")
s = stacked_fit(br.dropna(subset=["visit_ip"]), "pre_any", "v9_any")
P(f"  same overlap sample, W8->W9 hc602 contrast: {s['theta']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]")
P("  => if the in-person gap holds inside the very sample showing reversion, survivor")
P("     composition cannot explain the reversion (challenge replicated within-sample).")

# ================= S9 threshold ladder ======================================
P(); P("== S9 THRESHOLD LADDER (W9 RRs; W8-in-sample prevalence matched) ==")
for k in [1, 2, 3, 4, 6]:
    dd[f"v9_{k}"] = np.where(dd["docv9"] >= k, 1.0, np.where(dd["docv9"] >= 0, 0.0, np.nan))
    r = rr(f"v9_{k}", dd)
    P(f"  >={k} visits: W9 prev {dd[f'v9_{k}'].mean()*100:4.1f}% (W8 {(dd['docv8']>=k).mean()*100:4.1f}%)"
      f"  RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]")

# ================= S10 calibrated-weight sensitivity ========================
P(); P("== S10 CALIBRATED CROSS-SECTIONAL WEIGHT (cciw_w9) SENSITIVITY ==")
w9 = rd(f"sharew9_rel{R}_gv_weights.dta", ["mergeid", "cciw_w9"]).drop_duplicates("mergeid")
dw = dd.merge(w9, on="mergeid", how="left")
dw = dw[dw["cciw_w9"].notna() & (dw["cciw_w9"] > 0)]
for col, pre, lab in [("v9_any", "pre_any", "any-visit"), ("v9_4p", "pre_4", "4+-visit ")]:
    r = rr(col, dw, weights="cciw_w9")
    s = stacked_fit(dw, pre, col, weights="cciw_w9")
    P(f"  weighted {lab} (n={len(dw):,}): W9 RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] | contrast {s['theta']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]")

open("PAPER2_ANALYSIS.txt", "w").write("\n".join(OUT) + "\n")
P(); P("written: PAPER2_ANALYSIS.txt")
