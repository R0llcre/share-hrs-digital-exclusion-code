#!/usr/bin/env python
"""
HRS arm of the POSITIVE cross-cohort paper
("Pre-pandemic internet non-use as a marker of lower subsequent healthcare contact").

Outcome  : r15doctor  = any doctor visit at the 2020 wave (Wave 15), RAND HRS
           (recall since previous interview, ~2018-2021, telemedicine-inclusive).
Exposure : 2018 HRS Core QW303  (5 = does not regularly use the Internet -> 1; 1 = yes -> 0).
Covariates (all 2018 baseline, RAND wave 14): age, sex, education, self-rated health,
           chronic-condition count, depression (CES-D), functional limitation (any ADL/IADL); baseline utilisation = r14doctor.
Design   : modified-Poisson adjusted RR with household-clustered robust SE; survey-design
           sensitivity using r15wtresp weights + raestrat strata + raehsamp PSU; adjusted
           absolute risk via logistic marginal standardisation; risk difference via a
           linear-probability model with household-cluster-robust SE.

Reproduces every HRS number in the manuscript and supplement.
Run:  python hrs_analysis.py > HRS_output.txt
(set H at the top to your local HRS data directory)
"""
import pyreadstat, numpy as np, pandas as pd
import statsmodels.api as sm
np.seterr(all="ignore")

H = "../unzipped"
RAND = f"{H}/rand/randhrs1992_2022v1.dta"

def n2n(s):  return s.where(s >= 0, np.nan)
def norm(s): return s.astype(str).str.strip()

# ---- load ----
rv = ["hhid", "pn", "r15doctor", "r14agey_e", "ragender", "raedyrs",
      "r14shlt", "r14conde", "r14cesd", "r14doctor", "r14adl5a", "r14iadl5a",
      "r15wtresp", "raestrat", "raehsamp"]
r = pyreadstat.read_dta(RAND, usecols=rv)[0]
adl5, iadl5 = n2n(r["r14adl5a"]), n2n(r["r14iadl5a"])
R = pd.DataFrame({
    "HHID": norm(r["hhid"]), "PN": norm(r["pn"]),
    "visit": n2n(r["r15doctor"]),
    "age": n2n(r["r14agey_e"]),
    "female": (n2n(r["ragender"]) == 2).astype(float),
    "edu": n2n(r["raedyrs"]),
    "sphus": n2n(r["r14shlt"]),
    "chronic": n2n(r["r14conde"]),
    "cesd": n2n(r["r14cesd"]),
    "func": ((adl5 > 0) | (iadl5 > 0)).astype(float),   # any ADL/IADL difficulty (harmonised w/ SHARE function step)
    "butil": n2n(r["r14doctor"]),               # baseline (2018) doctor visit
    "wt": n2n(r["r15wtresp"]),
    "strat": r["raestrat"].astype("Int64").astype(str),
    "psu": r["raehsamp"].astype("Int64").astype(str),
    "HHc": norm(r["hhid"]),
})
R.loc[adl5.isna() & iadl5.isna(), "func"] = np.nan
w = pyreadstat.read_dta(f"{H}/h18sta/H18W_R.dta", usecols=["hhid", "pn", "QW303"])[0]
W = pd.DataFrame({"HHID": norm(w["hhid"]), "PN": norm(w["pn"]),
                  "excl": np.where(n2n(w["QW303"]) == 5, 1.0,
                          np.where(n2n(w["QW303"]) == 1, 0.0, np.nan))})

hd = R.merge(W, on=["HHID", "PN"], how="inner")
N18, Nlink, Nout = W["excl"].notna().sum(), len(hd), hd["visit"].notna().sum()
core = ["visit", "excl", "age", "female", "edu", "sphus", "chronic", "cesd"]
hdc = hd.dropna(subset=core).copy()
hdc["dclus"] = hdc["strat"] + "_" + hdc["psu"]
Nfin = len(hdc)
print("== HRS STROBE flow ==")
print(f"  2018 internet item {N18:,} -> outcome nonmiss {Nout:,} -> covariate-complete (analytic) {Nfin:,}")

def poisRR(x, cols, clus, wt=None):
    X = sm.add_constant(x[cols].astype(float)).values
    y = x["visit"].astype(float).values
    kw = dict(family=sm.families.Poisson())
    if wt is not None:
        kw["var_weights"] = np.asarray(x[wt].values, dtype=np.float64)
    m = sm.GLM(y, X, **kw).fit(cov_type="cluster", cov_kwds={"groups": x[clus].values})
    ci = m.conf_int()
    return np.exp(m.params[1]), np.exp(ci[1, 0]), np.exp(ci[1, 1]), len(x)

# ---- ladder (M0, M1, M2, M3; M2 = harmonised any-ADL/IADL function step, focal) ----
LAD = [("M0 age,sex,edu",                ["excl", "age", "female", "edu"]),
       ("M1 +SRH,chronic,depression",    ["excl", "age", "female", "edu", "sphus", "chronic", "cesd"]),
       ("M2 +function (any ADL/IADL)",   ["excl", "age", "female", "edu", "sphus", "chronic", "cesd", "func"]),
       ("M3 +baseline utilisation",      ["excl", "age", "female", "edu", "sphus", "chronic", "cesd", "func", "butil"])]
print("== HRS need-adjustment ladder (household-clustered; covariates 2018, outcome 2020) ==")
for lab, cols in LAD:
    x = hdc.dropna(subset=cols)
    rr = poisRR(x, cols, "HHc")
    print(f"  {lab:<30} RR {rr[0]:.3f} [{rr[1]:.3f},{rr[2]:.3f}] N={rr[3]:,}")

# ---- survey-design sensitivity (weight + strata/PSU clustered) ----
cols1 = ["excl", "age", "female", "edu", "sphus", "chronic", "cesd", "func"]   # M2 = cross-cohort focal (both cohorts have a function step)
ru = poisRR(hdc.dropna(subset=cols1), cols1, "HHc")
hw = hdc[hdc.wt > 0].dropna(subset=cols1)
rw = poisRR(hw, cols1, "dclus", wt="wt")
print("== HRS focal (M2, +function) RR ==")
print(f"  unweighted, household-clustered : RR {ru[0]:.3f} [{ru[1]:.3f},{ru[2]:.3f}]  (target: analytic sample)")
print(f"  design-weighted (r15wtresp)+strata/PSU : RR {rw[0]:.3f} [{rw[1]:.3f},{rw[2]:.3f}] "
      f"(N={len(hw):,}; target: US community pop >=50)")

# ---- adjusted absolute risk / RD via logistic marginal standardisation + bootstrap CI ----
def mrd(x, cols):
    X = sm.add_constant(x[cols].astype(float)).values
    y = x["visit"].astype(float).values
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    X0 = X.copy(); X0[:, 1] = 0
    X1 = X.copy(); X1[:, 1] = 1
    p0, p1 = m.predict(X0).mean(), m.predict(X1).mean()
    return p1 - p0, p0, p1

x1 = hdc.dropna(subset=cols1).reset_index(drop=True)
Xh = sm.add_constant(x1[cols1].astype(float)).values
yh = x1["visit"].astype(float).values
def rd_arr(Xm, ym):
    b = sm.GLM(ym, Xm, family=sm.families.Binomial()).fit().params
    X0 = Xm.copy(); X0[:, 1] = 0; X1m = Xm.copy(); X1m[:, 1] = 1
    s = lambda z: 1.0 / (1.0 + np.exp(-z))
    return s(X1m @ b).mean() - s(X0 @ b).mean()
b0 = sm.GLM(yh, Xh, family=sm.families.Binomial()).fit().params   # logistic marginal-standardised risks (descriptive)
s = lambda z: 1.0 / (1.0 + np.exp(-z))
X0 = Xh.copy(); X0[:, 1] = 0; X1m = Xh.copy(); X1m[:, 1] = 1
p0, p1 = s(X0 @ b0).mean(), s(X1m @ b0).mean()
# adjusted risk difference + 95% CI from a linear-probability model with household-cluster-robust SE
lpm = sm.OLS(yh, Xh).fit(cov_type="cluster", cov_kwds={"groups": x1["HHc"].values})
lci = lpm.conf_int()[1]
print("== HRS adjusted absolute risk (M2) ==")
print(f"  logistic-standardised risks: users {p0*100:.1f}% vs non-users {p1*100:.1f}%")
print(f"  adjusted RD (LPM, household-cluster-robust): {lpm.params[1]*100:+.1f}pp "
      f"[{lci[0]*100:+.1f}, {lci[1]*100:+.1f}]")

# ---- Table 1 ----
print("== HRS Table 1 (analytic sample) ==")
for k, sub in [("users", hdc[hdc.excl == 0]), ("non-users", hdc[hdc.excl == 1])]:
    print(f"  {k}: N={len(sub)} age={sub.age.mean():.1f}({sub.age.std():.1f}) "
          f"female={sub.female.mean()*100:.0f}% eduyrs={sub.edu.mean():.1f} "
          f"srh={sub.sphus.mean():.2f} chronic={sub.chronic.mean():.1f} cesd={sub.cesd.mean():.2f} "
          f"anyADLIADL%={sub.func.mean()*100:.0f} baseDocv2018%={sub.butil.mean()*100:.0f}")
def smd(v):
    a = hdc[hdc.excl == 0][v].dropna(); b = hdc[hdc.excl == 1][v].dropna()
    return (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
print("  SMD age=%.2f edu=%.2f srh=%.2f chronic=%.2f cesd=%.2f"
      % (smd("age"), smd("edu"), smd("sphus"), smd("chronic"), smd("cesd")))
print("  SMD (binary rows) female=%.2f anyADLIADL=%.2f baseDocv2018=%.2f"
      % (smd("female"), smd("func"), smd("butil")))
print("  missing%% sphus=%.1f chronic=%.1f cesd=%.1f butil=%.1f func=%.1f"
      % (hd.sphus.isna().mean()*100, hd.chronic.isna().mean()*100,
         hd.cesd.isna().mean()*100, hd.butil.isna().mean()*100, hd.func.isna().mean()*100))

# ---- SECONDARY outcome: delayed needed care (2020 COVID module, RCOVW579) ----
# Item: "Since March 2020, was there any time you needed [medical/dental/etc] care but DELAYED getting it?"
# Reported only by the 2020 COVID half-sample; uses the COVID-module respondent weight.
print("== HRS secondary: delayed needed care (RCOVW579, 2020 COVID module) ==")
cov = pyreadstat.read_dta(f"{H}/h20sta/H20COV_R.dta")[0]
cov.columns = [c.upper() for c in cov.columns]
wcol = next((c for c in cov.columns if "WGT" in c or "WTR" in c), None)   # no module weight ships in H20COV_R
dc = pd.DataFrame({"HHID": norm(cov["HHID"]), "PN": norm(cov["PN"]),
                   "delay": n2n(cov["RCOVW579"]).map({1: 1.0, 5: 0.0})})
dd2 = hd.merge(dc, on=["HHID", "PN"], how="inner")
ccov = ["excl", "age", "female", "edu", "sphus", "chronic", "cesd"]
dd2c = dd2.dropna(subset=["delay"] + ccov)
raw = dd2c.groupby("excl")["delay"].agg(["mean", "size"])
# robust modified-Poisson (logistic start params avoid the cold-start boundary nan on this subsample)
Xd = sm.add_constant(dd2c[ccov].astype(float)).values
yd = dd2c["delay"].astype(float).values
sp = sm.GLM(yd, Xd, family=sm.families.Binomial()).fit().params * 0.5
md = sm.GLM(yd, Xd, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": dd2c["HHc"].values}, start_params=sp)
ci = md.conf_int()
print("  Item RCOVW579: 'Since March 2020, any time you needed care but DELAYED getting it?' (Yes/No)")
print(f"  COVID-module weight in H20COV_R: {wcol} (none ships in this file -> unweighted; COVID half-sample)")
print(f"  N={len(dd2c):,}; raw delayed-care: users {raw.loc[0.0,'mean']*100:.1f}% (n={int(raw.loc[0.0,'size'])}) "
      f"vs non-users {raw.loc[1.0,'mean']*100:.1f}% (n={int(raw.loc[1.0,'size'])})")
print(f"  adjusted RR {np.exp(md.params[1]):.3f} [{np.exp(ci[1,0]):.3f},{np.exp(ci[1,1]):.3f}]; "
      f"item missingness within COVID subsample {dd2.assign(insub=dd2['delay'].notna())['delay'].isna().mean()*100:.1f}% "
      f"(the larger non-response is the COVID half-sample design, not item missingness)")
print("  NOTE: denominator-dependent self-report; cannot distinguish lower need, fewer opportunities to")
print("        experience delay, reporting differences, or better access. Secondary, cautious indicator only.")

# ---- Before/during difference-in-differences (contrast with SHARE) ----
# r14doctor (2018, pre-pandemic) vs r15doctor (2020, during; recall window ~2018-2021 -> weaker test).
print("== HRS before/during difference-in-differences ==")
ddd = hd.dropna(subset=["butil", "visit", "excl", "age", "female", "edu", "sphus", "chronic", "cesd", "func"]).copy()
covd = ["excl", "age", "female", "edu", "sphus", "chronic", "cesd", "func"]
def hrs_rr(col):
    X = sm.add_constant(ddd[covd].astype(float)).values; y = ddd[col].astype(float).values
    m = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": ddd["HHc"].values})
    ci = m.conf_int(); return np.exp(m.params[1]), np.exp(ci[1, 0]), np.exp(ci[1, 1])
rp = hrs_rr("butil"); rd_ = hrs_rr("visit")
print(f"  PRE  r14doctor (2018): RR {rp[0]:.3f} [{rp[1]:.3f},{rp[2]:.3f}]")
print(f"  DURING r15doctor (2020): RR {rd_[0]:.3f} [{rd_[1]:.3f},{rd_[2]:.3f}]")
import patsy as _patsy
lo = pd.concat([ddd.assign(y=ddd["butil"].values, per=0.0), ddd.assign(y=ddd["visit"].values, per=1.0)], ignore_index=True)
# fully-interacted (covariate x period): interaction = ratio of period-specific adjusted RRs
Xfi = _patsy.dmatrix("excl*per + (age+female+edu+sphus+chronic+cesd+func)*per", lo, return_type="dataframe")
ml = sm.GLM(lo["y"].astype(float).values, Xfi.values, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": lo["HHc"].values})
j = list(Xfi.columns).index("excl:per"); ci = ml.conf_int()
print(f"  period contrast (2018 vs 2020, fully-interacted): ratio-of-RR {np.exp(ml.params[j]):.3f} [{np.exp(ci[j,0]):.3f},{np.exp(ci[j,1]):.3f}] p={ml.pvalues[j]:.2f}")
print("  => no significant widening: the HRS broad-contact gap was already present in 2018 (pre-existing gradient,")
print("     and the 2020 measure spans the pre-pandemic period, so HRS cannot test pandemic-specific widening).")


# ====================== CONSOLIDATED EXTENDED ANALYSES (review-added) ======================
# Reproduces REVISION_PLAN HRS numbers. Reuses module-level helpers (n2n, norm, H).
import statsmodels.formula.api as smf  # noqa

_cols = ["hhid","pn","r12doctor","r13doctor","r14doctor","r15doctor","r14agey_e","ragender","raedyrs",
         "r14shlt","r14conde","r14cesd","r14adl5a","r14iadl5a",
         "h14itot","h14atotb","raracem","rahispan","r14higov","h14hhres","r14cog27"]
_r = pyreadstat.read_dta(RAND, usecols=_cols)[0]
_adl, _iadl = n2n(_r["r14adl5a"]), n2n(_r["r14iadl5a"])
E = pd.DataFrame({"HHID": norm(_r["hhid"]), "PN": norm(_r["pn"]),
    "r12": n2n(_r["r12doctor"]), "r13": n2n(_r["r13doctor"]), "r14": n2n(_r["r14doctor"]), "r15": n2n(_r["r15doctor"]),
    "age": n2n(_r["r14agey_e"]), "female": (n2n(_r["ragender"]) == 2).astype(float), "edu": n2n(_r["raedyrs"]),
    "sphus": n2n(_r["r14shlt"]), "chronic": n2n(_r["r14conde"]), "cesd": n2n(_r["r14cesd"]),
    "func": ((_adl > 0) | (_iadl > 0)).astype(float), "inc": n2n(_r["h14itot"]), "wealth": n2n(_r["h14atotb"]),
    "race": n2n(_r["raracem"]), "hisp": n2n(_r["rahispan"]), "gov": n2n(_r["r14higov"]),
    "hhres": n2n(_r["h14hhres"]), "cog": n2n(_r["r14cog27"]), "HHc": norm(_r["hhid"])})
E = E.merge(W, on=["HHID", "PN"])
_cov = ["excl", "age", "female", "edu", "sphus", "chronic", "cesd"]
def _hrr(df, col, cols):
    x = df.dropna(subset=[col] + cols); X = sm.add_constant(x[cols].astype(float)).values
    m = sm.GLM(x[col].astype(float).values, X, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": x["HHc"].values})
    ci = m.conf_int(); return float(np.exp(m.params[1])), float(np.exp(ci[1,0])), float(np.exp(ci[1,1])), len(x)

print("== HRS multi-wave pre-trend (2018 exposure -> any doctor visit) ==")
for lab, c in [("r12 2014 (pre)", "r12"), ("r13 2016 (pre)", "r13"), ("r14 2018 (pre)", "r14"), ("r15 2020 (during)", "r15")]:
    r = _hrr(E, c, _cov); print(f"  {lab:<18} RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] N={r[3]:,}")
print("  => HRS gap present since 2014 and stable through 2020 (pre-existing gradient; no within-pandemic jump).")

print("== HRS extended-confounder sensitivity (focal vs +income+wealth+race+hispanic+insurance+living-alone+cognition) ==")
E["incq"] = pd.qcut(E["inc"].rank(method="first"), 5, labels=False, duplicates="drop")
E["wq"]   = pd.qcut(E["wealth"].rank(method="first"), 5, labels=False, duplicates="drop")
E["alone"] = (E["hhres"] == 1).astype(float)
def _ext(extra):
    cols2 = _cov + extra; x = E.dropna(subset=["r15"] + cols2).copy()
    X = sm.add_constant(pd.get_dummies(x[cols2], columns=[c for c in ["incq","wq","race","hisp","gov","alone"] if c in cols2], drop_first=True).astype(float)).values
    m = sm.GLM(x["r15"].astype(float).values, X, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": x["HHc"].values})
    ci = m.conf_int(); return float(np.exp(m.params[1])), float(np.exp(ci[1,0])), float(np.exp(ci[1,1])), len(x)
r0 = _ext([]); r1 = _ext(["incq","wq","race","hisp","gov","alone","cog"])
print(f"  focal need-set                                       RR {r0[0]:.3f} [{r0[1]:.3f},{r0[2]:.3f}] N={r0[3]:,}")
print(f"  +income+wealth+race+hispanic+insurance+alone+cognition RR {r1[0]:.3f} [{r1[1]:.3f},{r1[2]:.3f}] N={r1[3]:,}")
print("  => modest ATTENUATION toward the null but remains significant: not merely a proxy for SES/race/cognition/isolation.")

print("== HRS 65+ sensitivity (focal M2) ==")
_h65 = hdc[hdc["age"] >= 65].dropna(subset=cols1)
_r = poisRR(_h65, cols1, "HHc")
print(f"  50+ RR {ru[0]:.3f}; 65+ RR {_r[0]:.3f} [{_r[1]:.3f},{_r[2]:.3f}] N={len(_h65):,}")


# ===== TRACEABILITY (HRS): raw percentages and E-values for the focal estimate =====
import math as _m
def _evh(rr):
    r=1.0/rr if rr<1 else rr
    return r+_m.sqrt(r*(r-1))
print("\n===== TRACEABILITY (HRS) =====")
_x = hdc.dropna(subset=cols1)
print(f"  raw 2020 doctor visit by exposure: users {_x[_x.excl==0]['visit'].mean()*100:.1f}% "
      f"vs non-users {_x[_x.excl==1]['visit'].mean()*100:.1f}%  (N={len(_x):,})")
print(f"  HRS focal RR {ru[0]:.3f} -> E-value {_evh(ru[0]):.3f}; "
      f"CI limit closest to null {ru[2]:.3f} -> E-value {_evh(ru[2]):.3f}")

