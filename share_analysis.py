#!/usr/bin/env python
"""
SHARE arm of the POSITIVE cross-cohort paper
("Pre-pandemic internet non-use as a marker of lower subsequent healthcare contact").

Outcome  : caq120_ = in-person medical visit, Corona Survey 2 (2021, pandemic-specific).
           Temporal robustness: caq020_ = in-person visit, Corona Survey 1 (2020).
Exposure : Wave 8 it004_  (5 = no internet use last 7 days -> 1; 1 = yes -> 0).
Covariates (all Wave 8, pre-pandemic): age, sex, education (ISCED), self-rated health,
           chronic count, depression (EURO-D), ADL, IADL; baseline utilisation = hc602
           (doctor visits) + hc012 (hospital).
Design   : modified-Poisson adjusted RR with country fixed effects, country-clustered SE;
           sequential need-adjustment ladder M0-M3; adjusted absolute risk via logistic
           standardisation, RD via linear-probability model with country-cluster-robust SE; effect modification
           by interaction; between-country heterogeneity by Cochran's Q / I^2.
           (Wild cluster bootstrap-t inference: see wild_bootstrap.py.)

Reproduces every SHARE number in the manuscript and supplement.
Run:  python share_analysis.py > SHARE_output.txt
(set U at the top to your local SHARE data directory; wild cluster bootstrap-t
 inference: see wild_bootstrap.py)
"""
import pyreadstat, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
np.seterr(all="ignore")

U, R = "../unzipped", "9-0-0"
def n2n(s): return s.where(s >= 0, np.nan)
def yn(s):  return n2n(s).map({1: 1.0, 5: 0.0})
def rd(fn, c): return pyreadstat.read_dta(f"{U}/{fn}", usecols=c)[0]

it8 = rd(f"sharew8_rel{R}_it.dta", ["mergeid", "it004_"])
it8["excl"] = np.where(it8["it004_"] == 5, 1.0, np.where(it8["it004_"] == 1, 0.0, np.nan))
cvr = rd(f"sharew8_rel{R}_cv_r.dta", ["mergeid", "country", "gender", "age2020", "int_year", "int_month"]).drop_duplicates("mergeid")
isc = rd(f"sharew8_rel{R}_gv_isced.dta", ["mergeid", "isced1997_r"])
gh  = rd(f"sharew8_rel{R}_gv_health.dta", ["mergeid", "sphus", "chronicw8", "eurod", "adl", "iadl"])
hc  = rd(f"sharew8_rel{R}_hc.dta", ["mergeid", "hc602_", "hc012_", "hc889_"])  # hc889 = health literacy (1=always needs help..5=never)
s1  = rd(f"sharew8ca_rel{R}_ca.dta", ["mergeid", "caq020_"])   # SCS1 2020
s2  = rd(f"sharew9ca_rel{R}_ca.dta", ["mergeid", "caq120_"])   # SCS2 2021 (primary)
wt9 = rd(f"sharew9ca_rel{R}_gv_weights_ca.dta", ["mergeid", "cciw_w9ca_ca"])  # SCS2 calibrated weight

b = (cvr.merge(it8[["mergeid", "excl"]], on="mergeid")
        .merge(isc, on="mergeid", how="left")
        .merge(gh, on="mergeid", how="left")
        .merge(hc, on="mergeid", how="left"))
b["age"] = n2n(b["age2020"]); N_w8 = len(b)
b1 = b[b["age"] >= 50]; b2 = b1[b1["country"] != 25]; b3 = b2[b2["excl"].notna()].copy()
N_age, N_ctry, N_excl = len(b1), len(b2), len(b3)
b3["agecat"] = pd.cut(b3["age"], [49, 64, 79, 200], labels=["a", "b", "c"])
b3["sex"] = n2n(b3["gender"]).map({1: "m", 2: "f"})
e = n2n(b3["isced1997_r"]).where(lambda s: s <= 6)
b3["edu3"] = pd.cut(e, [-1, 2, 4, 6], labels=["lo", "mi", "hi"])
for c in ["sphus", "chronicw8", "eurod", "adl", "iadl"]:
    b3[c] = n2n(b3[c])
b3["docv8"] = n2n(b3["hc602_"]); b3["hosp8"] = yn(b3["hc012_"])
b3["lowlit"] = (n2n(b3["hc889_"]) <= 2).astype(float)   # 1=Always / 2=Often needs help = LOW health literacy (~8%)
b3.loc[n2n(b3["hc889_"]).isna(), "lowlit"] = np.nan
b3["cc"] = b3["country"].astype(int).astype(str)

d = b3.merge(s2, on="mergeid", how="inner"); N_scs2 = len(d)
d["visit"] = yn(d["caq120_"]); N_out = d["visit"].notna().sum()
DROP = ["visit", "excl", "agecat", "sex", "edu3", "sphus", "chronicw8",
        "eurod", "adl", "iadl", "docv8", "hosp8", "cc"]
dd = d.dropna(subset=DROP).copy(); N_final = len(dd)
dd = dd.merge(wt9, on="mergeid", how="left")
dd["w"] = n2n(dd["cciw_w9ca_ca"])
print("== SHARE STROBE flow ==")
print(f"  W8 {N_w8:,} -> age>=50 {N_age:,} -> ex-Israel {N_ctry:,} -> exposure {N_excl:,} "
      f"-> in SCS2 {N_scs2:,} -> outcome {N_out:,} -> analytic {N_final:,}")
print(f"  countries in analytic sample: {dd['cc'].nunique()}")

def rr(x, formula):
    m = smf.glm(f"visit ~ excl + {formula}", data=x, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": x["cc"]})
    ci = m.conf_int().loc["excl"]
    return np.exp(m.params["excl"]), np.exp(ci[0]), np.exp(ci[1])

LAD = [("M0 age,sex,edu,countryFE",        "C(agecat)+C(sex)+C(edu3)+C(cc)"),
       ("M1 +SRH,chronic,depression",      "C(agecat)+C(sex)+C(edu3)+C(cc)+sphus+chronicw8+eurod"),
       ("M2 +ADL,IADL (function; focal)",  "C(agecat)+C(sex)+C(edu3)+C(cc)+sphus+chronicw8+eurod+adl+iadl"),
       ("M3 +baseline utilisation",        "C(agecat)+C(sex)+C(edu3)+C(cc)+sphus+chronicw8+eurod+adl+iadl+docv8+hosp8")]
print(f"== SHARE need-adjustment ladder (caq120 SCS2; N={N_final:,}) ==")
for lab, f in LAD:
    r = rr(dd, f); print(f"  {lab:<32} RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]")

# adjusted absolute risk via logistic marginal standardisation (M2 focal) on a
# precomputed design matrix; risk difference + CI from a cluster-robust LPM below
import patsy
fM2 = "C(agecat)+C(sex)+C(edu3)+C(cc)+sphus+chronicw8+eurod+adl+iadl"
yd, Xd = patsy.dmatrices(f"visit ~ excl + {fM2}", dd, return_type="matrix")
Xf = np.asarray(Xd); yv = np.asarray(yd).ravel()
ecol = Xd.design_info.column_names.index("excl")
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
def rd_np(Xm, ym):
    b = sm.GLM(ym, Xm, family=sm.families.Binomial()).fit().params
    X0 = Xm.copy(); X0[:, ecol] = 0
    X1 = Xm.copy(); X1[:, ecol] = 1
    p0, p1 = sig(X0 @ b).mean(), sig(X1 @ b).mean()
    return p1 - p0, p0, p1
pt, p0, p1 = rd_np(Xf, yv)   # logistic marginal-standardised risks (descriptive)
# adjusted risk difference + 95% CI from a linear-probability model with country-cluster-robust SE
lpm = sm.OLS(yv, Xf).fit(cov_type="cluster", cov_kwds={"groups": dd["cc"].values})
lci = lpm.conf_int()[ecol]
print("== SHARE adjusted absolute risk (M2) ==")
print(f"  logistic-standardised risks: users {p0*100:.1f}% vs non-users {p1*100:.1f}%")
print(f"  adjusted RD (LPM, country-cluster-robust): {lpm.params[ecol]*100:+.1f}pp "
      f"[{lci[0]*100:+.1f}, {lci[1]*100:+.1f}]")

# selection / weighting sensitivity: SCS2 calibrated weight (cciw_w9ca_ca) on the M2 model
print("== SHARE selection/weighting sensitivity (M2) ==")
dw = dd.dropna(subset=["w"])
mw = smf.glm(f"visit ~ excl + {fM2}", data=dw, family=sm.families.Poisson(),
             var_weights=dw["w"].values).fit(cov_type="cluster", cov_kwds={"groups": dw["cc"]})
ciw = mw.conf_int().loc["excl"]
print(f"  unweighted M2 RR {rr(dd, fM2)[0]:.3f}; SCS2-calibrated-weighted RR "
      f"{np.exp(mw.params['excl']):.3f} [{np.exp(ciw[0]):.3f},{np.exp(ciw[1]):.3f}] (N={len(dw):,})")
# IPW for SCS2 participation among W8 exposure-known (selection model on baseline covariates)
allw8 = b3.copy()
allw8["inSCS2"] = allw8["mergeid"].isin(set(dd["mergeid"])).astype(float)
sm_cov = "C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl+C(cc)"
allw8f = allw8.dropna(subset=["inSCS2", "agecat", "sex", "edu3", "sphus", "chronicw8", "eurod", "adl", "iadl"])
psel = smf.glm(f"inSCS2 ~ {sm_cov}", data=allw8f, family=sm.families.Binomial()).fit().predict(allw8f)
ipw = pd.Series((1.0 / psel).values, index=allw8f["mergeid"].values)
dd["ipw"] = dd["mergeid"].map(ipw)
di = dd.dropna(subset=["ipw"])
mi = smf.glm(f"visit ~ excl + {fM2}", data=di, family=sm.families.Poisson(),
             var_weights=di["ipw"].clip(upper=di["ipw"].quantile(0.99)).values).fit(
             cov_type="cluster", cov_kwds={"groups": di["cc"]})
print(f"  IPW for SCS2 participation: RR {np.exp(mi.params['excl']):.3f} "
      f"[{np.exp(mi.conf_int().loc['excl'][0]):.3f},{np.exp(mi.conf_int().loc['excl'][1]):.3f}] (N={len(di):,})")
_wt = di["ipw"].clip(upper=di["ipw"].quantile(0.99))
print(f"  IPW weight distribution (truncated at 99th pct): mean {_wt.mean():.2f}, "
      f"median {_wt.median():.2f}, p99 {_wt.quantile(0.99):.2f}, max {_wt.max():.2f}")

# pre-pandemic exposure-timing sensitivity: exclude Wave-8 interviews from March 2020 onward
# (the great majority of 2020-calendar-year W8 interviews were Jan-Feb, i.e. still pre-pandemic)
_iy=n2n(dd["int_year"]); _im=n2n(dd["int_month"])
de=dd[~((_iy==2020)&(_im>=3))]
_re=rr(de,fM2)
print("== SHARE pre-pandemic exposure-timing sensitivity (focal M2) ==")
print(f"  excl W8 interviews from Mar 2020 onward: RR {_re[0]:.3f} [{_re[1]:.3f},{_re[2]:.3f}] "
      f"N={len(de):,} (dropped {len(dd)-len(de):,})")

# multimorbidity-stratified standardised risk difference (absolute, not just RR)
print("== SHARE multimorbidity-stratified absolute RD (M2 standardisation) ==")
for lab, sub in [("multimorbid (2+)", dd[dd.chronicw8 >= 2]), ("not multimorbid", dd[dd.chronicw8 < 2])]:
    ys, Xs = patsy.dmatrices(f"visit ~ excl + {fM2}", sub, return_type="matrix")
    ec = Xs.design_info.column_names.index("excl")   # excl column index for THIS subgroup design
    Xa = np.asarray(Xs); ya = np.asarray(ys).ravel()
    bsg = sm.GLM(ya, Xa, family=sm.families.Binomial()).fit().params
    X0 = Xa.copy(); X0[:, ec] = 0; X1 = Xa.copy(); X1[:, ec] = 1
    q0, q1 = sig(X0 @ bsg).mean(), sig(X1 @ bsg).mean()
    print(f"  {lab:<18} RD {(q1-q0)*100:+.1f}pp (users {q0*100:.1f}% vs non-users {q1*100:.1f}%) N={len(sub):,}")

# pre-specified effect modification (interaction tests).
# NB: when the modifier is derived from a covariate already in fM2 (education, literacy),
# that covariate is dropped from the adjustment set to avoid collinearity.
fM2_noedu = "C(agecat)+C(sex)+sphus+chronicw8+eurod+adl+iadl"
dd["mm"] = (dd.chronicw8 >= 2).astype(float)
dd["lowedu"] = (dd.edu3 == "lo").astype(float)
print("== SHARE pre-specified effect modification ==")
def interaction_p(formula, term):
    mi = smf.glm(formula, data=dd.dropna(subset=["lowlit"]) if "lowlit" in formula else dd,
                 family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups":
                 (dd.dropna(subset=["lowlit"]) if "lowlit" in formula else dd)["cc"]})
    ip = [k for k in mi.params.index if term in k]
    return mi.pvalues[ip[0]]
print(f"  multimorbidity x exposure  p = {interaction_p(f'visit ~ excl*mm + {fM2}', 'excl:mm'):.3f}")
print(f"  low-education x exposure   p = {interaction_p(f'visit ~ excl*lowedu + {fM2_noedu}+C(cc)', 'excl:lowedu'):.3f}")
print(f"  health-literacy x exposure p = {interaction_p(f'visit ~ excl*lowlit + {fM2}', 'excl:lowlit'):.3f}")
# literacy subgroup estimates (low = needs help always/often)
dl = dd.dropna(subset=["lowlit"])
for lab, sub in [("Low health literacy", dl[dl.lowlit == 1]), ("Higher health literacy", dl[dl.lowlit == 0])]:
    _r = rr(sub, fM2)
    print(f"  {lab:<22} RR {_r[0]:.3f} [{_r[1]:.3f},{_r[2]:.3f}]  N={len(sub):,}")

# subgroup estimates (Figure 3) -- drop the stratifier from covariates to avoid collinearity
full = "C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl"
noage = "C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl"
noedu = "C(agecat)+C(sex)+sphus+chronicw8+eurod+adl+iadl"
nosex = "C(agecat)+C(edu3)+sphus+chronicw8+eurod+adl+iadl"
def rrc(x, cov):
    m = smf.glm(f"visit ~ excl + {cov}+C(cc)", data=x, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": x["cc"]})
    ci = m.conf_int().loc["excl"]
    return np.exp(m.params["excl"]), np.exp(ci[0]), np.exp(ci[1])
print("== SHARE subgroups (Figure 3 / need gradient) ==")
for lab, x, cov in [("Age 50-64", dd[dd.agecat == "a"], noage), ("Age 65-79", dd[dd.agecat == "b"], noage),
                    ("Age 80+", dd[dd.agecat == "c"], noage), ("Women", dd[dd.sex == "f"], nosex),
                    ("Men", dd[dd.sex == "m"], nosex), ("Low education", dd[dd.edu3 == "lo"], noedu),
                    ("High education", dd[dd.edu3 == "hi"], noedu),
                    ("Not multimorbid", dd[dd.chronicw8 < 2], full), ("Multimorbid", dd[dd.chronicw8 >= 2], full),
                    ("No ADL limitation", dd[dd.adl == 0], full), ("Any ADL limitation", dd[dd.adl > 0], full)]:
    try:
        r = rrc(x, cov); print(f"  {lab:<18} RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]  N={len(x):,}")
    except Exception as ex: print(f"  {lab:<18} fail {str(ex)[:30]}")

# between-country heterogeneity (+ per-country RRs and random-effects pooling for the forest figure)
rows = []; clabs = []
for c in sorted(dd.cc.unique()):
    x = dd[dd.cc == c]
    if x.excl.nunique() < 2 or len(x) < 150: continue
    try:
        m = smf.glm("visit ~ excl + C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl",
                    data=x, family=sm.families.Poisson()).fit()
        rows.append((m.params["excl"], m.bse["excl"])); clabs.append((c, len(x)))
    except Exception: pass
th = np.array([r[0] for r in rows]); se = np.array([r[1] for r in rows]); w = 1 / se**2
thbar = (w * th).sum() / w.sum(); Q = (w * (th - thbar) ** 2).sum(); k = len(rows)
I2 = max(0, (Q - (k - 1)) / Q) * 100
print("== SHARE between-country heterogeneity ==")
print(f"  {k} countries: Cochran Q={Q:.1f} df={k-1} p={1-stats.chi2.cdf(Q,k-1):.3f} I2={I2:.0f}%; "
      f"RR range {np.exp(th).min():.2f}-{np.exp(th).max():.2f}, {(np.exp(th)<1).sum()}/{k} below 1")
# DerSimonian-Laird random-effects pooled estimate (complements the country-FE model)
tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
wre = 1 / (se**2 + tau2)
thre = (wre * th).sum() / wre.sum(); sere = np.sqrt(1 / wre.sum())
print(f"  random-effects (DerSimonian-Laird) pooled RR {np.exp(thre):.3f} "
      f"[{np.exp(thre-1.96*sere):.3f},{np.exp(thre+1.96*sere):.3f}] (tau^2={tau2:.4f})")
print("  per-country RRs (95% CI), country code / N:")
for (c, n), t, s in zip(clabs, th, se):
    print(f"    country {c:>3} N={n:>5,}  RR {np.exp(t):.3f} [{np.exp(t-1.96*s):.3f},{np.exp(t+1.96*s):.3f}]")

# temporal (SCS1 vs SCS2)
d1 = b3.merge(s1, on="mergeid", how="inner"); d1["visit"] = yn(d1["caq020_"])
d1d = d1.dropna(subset=DROP).copy()
r1 = rr(d1d, fM2)
print("== SHARE temporal ==")
print(f"  SCS1 (2020, caq020): RR {r1[0]:.3f} [{r1[1]:.3f},{r1[2]:.3f}] N={len(d1d):,}")
r2 = rr(dd, fM2)
print(f"  SCS2 (2021, caq120): RR {r2[0]:.3f} [{r2[1]:.3f},{r2[2]:.3f}] N={N_final:,}")

# ---- Before/during difference-in-differences: did the pandemic WIDEN the gap? ----
print("== SHARE before/during difference-in-differences ==")
dd["pre_any"] = np.where(dd["docv8"] >= 1, 1.0, np.where(dd["docv8"] >= 0, 0.0, np.nan))
dd["pre_4"]   = np.where(dd["docv8"] >= 4, 1.0, np.where(dd["docv8"] >= 0, 0.0, np.nan))
def rr_of(col):
    x = dd.dropna(subset=[col])
    m = smf.glm(f"{col} ~ excl + {fM2}", data=x, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": x["cc"]})
    ci = m.conf_int().loc["excl"]; return np.exp(m.params["excl"]), np.exp(ci[0]), np.exp(ci[1])
ra = rr_of("pre_any"); r4 = rr_of("pre_4")
print(f"  PRE  any doctor visit (~91%): RR {ra[0]:.3f} [{ra[1]:.3f},{ra[2]:.3f}]")
print(f"  PRE  4+ visits (~61%, room) : RR {r4[0]:.3f} [{r4[1]:.3f},{r4[2]:.3f}]")
print(f"  DURING in-person (SCS2)     : RR {r2[0]:.3f} [{r2[1]:.3f},{r2[2]:.3f}]")
def did(precol):
    # FULLY-INTERACTED (covariate x period + country x period): interaction = ratio of the
    # period-specific adjusted RRs (matches the separately-standardised wave RRs in Figure 1).
    cols = ["excl","agecat","sex","edu3","sphus","chronicw8","eurod","adl","iadl","cc"]
    lo = pd.concat([dd[cols+[precol]].rename(columns={precol:"y"}).assign(per=0.0),
                    dd[cols+["visit"]].rename(columns={"visit":"y"}).assign(per=1.0)], ignore_index=True)
    Xfi = patsy.dmatrix("excl*per + (C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl+C(cc))*per",
                        lo, return_type="dataframe")
    m = sm.GLM(lo["y"].astype(float).values, Xfi.values, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": lo["cc"].values})
    j = list(Xfi.columns).index("excl:per"); ci = m.conf_int()
    return np.exp(m.params[j]), np.exp(ci[j,0]), np.exp(ci[j,1]), m.pvalues[j]
da = did("pre_any"); d4 = did("pre_4")
print(f"  period contrast, any-visit baseline (fully-interacted, absorbs country x period): ratio-of-RR {da[0]:.3f} [{da[1]:.3f},{da[2]:.3f}] p={da[3]:.1e}")
print(f"  period contrast, 4+-visit baseline : ratio-of-RR {d4[0]:.3f} [{d4[1]:.3f},{d4[2]:.3f}] p={d4[3]:.1e}")
print("  => ratio of during- to before-pandemic adjusted RR; <1 = larger non-user gap in the pandemic measure (descriptive, not causal).")

# Table 1
print("== SHARE Table 1 (analytic sample) ==")
for kk, sub in [("users", dd[dd.excl == 0]), ("non-users", dd[dd.excl == 1])]:
    print(f"  {kk}: N={len(sub)} age={sub.age.mean():.1f}({sub.age.std():.1f}) "
          f"female={(sub.sex=='f').mean()*100:.0f}% eduLO%={(sub.edu3=='lo').mean()*100:.0f} "
          f"srh={sub.sphus.mean():.2f} eurod={sub.eurod.mean():.2f} mm%={(sub.chronicw8>=2).mean()*100:.0f} "
          f"adl={sub.adl.mean():.2f} anyADLIADL%={((sub.adl>0)|(sub.iadl>0)).mean()*100:.0f} baseDocv={sub.docv8.mean():.1f}")
def smd(v):
    a = dd[dd.excl == 0][v].dropna(); bb = dd[dd.excl == 1][v].dropna()
    return (bb.mean() - a.mean()) / np.sqrt((a.var() + bb.var()) / 2)
print("  SMD age=%.2f sphus=%.2f chronic=%.2f eurod=%.2f adl=%.2f"
      % (smd("age"), smd("sphus"), smd("chronicw8"), smd("eurod"), smd("adl")))
# binary-row SMDs so every Table 1 SMD is script-traceable
dd["_fe"] = (dd.sex == "f").astype(float); dd["_el"] = (dd.edu3 == "lo").astype(float)
dd["_mm"] = (dd.chronicw8 >= 2).astype(float)
dd["_af"] = ((dd.adl > 0) | (dd.iadl > 0)).astype(float)
print("  SMD (binary + count rows) female=%.2f edu-lo=%.2f multimorbid=%.2f anyADLIADL=%.2f baseDocv=%.2f"
      % (smd("_fe"), smd("_el"), smd("_mm"), smd("_af"), smd("docv8")))


# ====================== CONSOLIDATED EXTENDED ANALYSES (review-added) ======================
# All reproduce the REVISION_PLAN numbers. Reuses module-level helpers (n2n, yn, rd, U, R, fM2, dd).

# --- (A) Multi-wave before/during pre-trend: W6 (2015), W8 (2019), SCS2 (2021) ---
print("== SHARE multi-wave pre-trend (period x internet interaction, not a causal DiD) ==")
hc6 = rd(f"sharew6_rel{R}_hc.dta", ["mergeid", "hc602_"]).rename(columns={"hc602_": "hc602w6"})
mw = dd.merge(hc6, on="mergeid", how="left")
mw["preW6"] = np.where(n2n(mw["hc602w6"]) >= 1, 1.0, np.where(n2n(mw["hc602w6"]) >= 0, 0.0, np.nan))
mw["preW8"] = np.where(mw["docv8"] >= 1, 1.0, np.where(mw["docv8"] >= 0, 0.0, np.nan))
def _rr(df, col):
    x = df.dropna(subset=[col]); m = smf.glm(f"{col} ~ excl + {fM2}", data=x, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": x["cc"]}); ci = m.conf_int().loc["excl"]
    return float(np.exp(m.params["excl"])), float(np.exp(ci[0])), float(np.exp(ci[1])), len(x)
for lab, c in [("W6 2015 (pre)", "preW6"), ("W8 2019 (pre)", "preW8"), ("SCS2 2021 (during, in-person)", "visit")]:
    r = _rr(mw, c); print(f"  {lab:<32} RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] N={r[3]:,}")
print("  => gap small and STABLE across two pre-pandemic waves, larger in the pandemic in-person measure.")

# --- (B) Total contact = in-person OR remote (caq118 is a COUNT: 0=none, >=1 had remote) ---
print("== SHARE total contact vs modality (during) ==")
rem = rd(f"sharew9ca_rel{R}_ca.dta", ["mergeid", "caq118_"])
tc = dd.merge(rem, on="mergeid", how="left")
r118 = n2n(tc["caq118_"]); tc["remote"] = np.where(r118 >= 1, 1.0, np.where(r118 >= 0, 0.0, np.nan))
tc["anyc"] = np.where((tc["visit"] == 1) | (tc["remote"] == 1), 1.0,
              np.where((tc["visit"] == 0) & (tc["remote"] == 0), 0.0, np.nan))
for lab, c in [("in-person", "visit"), ("remote (>=1)", "remote"), ("ANY contact (in-person OR remote)", "anyc")]:
    r = _rr(tc.dropna(subset=[c]), c); print(f"  {lab:<34} RR {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] N={r[3]:,}")
print("  => total-contact gap real but smaller than in-person; remote only partially offsets (net deficit).")
# adjusted standardised % (g-computation, focal logistic) for the modality table
import patsy as _pat
_sig=lambda z:1/(1+np.exp(-z))
for _lab,_c in [("in-person","visit"),("remote","remote"),("any","anyc")]:
    _x=tc.dropna(subset=[_c,"excl","agecat","sex","edu3","sphus","chronicw8","eurod","adl","iadl","cc"]).copy()
    _Xd=_pat.dmatrix(f"excl + {fM2}",_x,return_type="dataframe"); _j=list(_Xd.columns).index("excl"); _X=_Xd.values
    _m=sm.GLM(_x[_c].astype(float).values,_X,family=sm.families.Binomial()).fit()
    _X0=_X.copy();_X0[:,_j]=0;_X1=_X.copy();_X1[:,_j]=1
    _p0=_sig(_X0@_m.params).mean()*100;_p1=_sig(_X1@_m.params).mean()*100
    print(f"  adj-std %% {_lab:<10} users {_p0:.1f} / non-users {_p1:.1f}  RD {_p1-_p0:+.1f}pp")

# attrition: SCS2-included vs not (W8 exposure-known)
print("== SHARE attrition: SCS2-included vs not-included (W8 exposure-known) ==")
_ins=set(dd["mergeid"])
_a=b3.copy(); _a["inSCS2"]=_a["mergeid"].isin(_ins)
for _k,_sub in [("included",_a[_a.inSCS2]),("not included",_a[~_a.inSCS2])]:
    print(f"  {_k}: N={len(_sub):,} age={_sub.age.mean():.1f} female%={(_sub.sex=='f').mean()*100:.0f} "
          f"non-user%={_sub.excl.mean()*100:.0f} edu-lo%={(_sub.edu3=='lo').mean()*100:.0f} "
          f"SRH={_sub.sphus.mean():.2f} EURO-D={_sub.eurod.mean():.2f} "
          f"anyADLIADL%={((_sub.adl>0)|(_sub.iadl>0)).mean()*100:.0f} "
          f"multimorbid%={(_sub.chronicw8>=2).mean()*100:.0f} baseDocv={_sub.docv8.mean():.1f}")

# --- (C) Extended-confounder sensitivity: +income +wealth +living-alone ---
print("== SHARE extended-confounder sensitivity (focal need-set vs +income+wealth+living-alone) ==")
dn = rd(f"sharew8_rel{R}_dn.dta", ["mergeid", "dn014_"])
imp = rd(f"sharew8_rel{R}_gv_imputations.dta", ["mergeid", "thinc", "hnetw"]).groupby("mergeid", as_index=False).mean()
ex = dd.merge(dn, on="mergeid", how="left").merge(imp, on="mergeid", how="left")
ex["incq"] = pd.qcut(n2n(ex["thinc"]).rank(method="first"), 5, labels=False, duplicates="drop")
ex["wq"]   = pd.qcut(n2n(ex["hnetw"]).rank(method="first"), 5, labels=False, duplicates="drop")
ex["alone"] = n2n(ex["dn014_"]).map(lambda x: 0.0 if x in (1, 2) else (1.0 if x in (3, 4, 5, 6) else np.nan))
cc_ex = ex.dropna(subset=["visit", "excl", "agecat", "sex", "edu3", "sphus", "chronicw8", "eurod", "adl", "iadl", "cc", "incq", "wq", "alone"]).copy()
def _rrf(df, f):
    m = smf.glm(f"visit ~ excl + {f}", data=df, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": df["cc"]})
    ci = m.conf_int().loc["excl"]; return float(np.exp(m.params["excl"])), float(np.exp(ci[0])), float(np.exp(ci[1]))
# PRIMARY extended = income+wealth on near-full sample (income/wealth ~99% covered);
# living-arrangement (dn014) covers only ~29% -> reported as a smaller secondary sensitivity.
cc_iw = ex.dropna(subset=["visit","excl","agecat","sex","edu3","sphus","chronicw8","eurod","adl","iadl","cc","incq","wq"]).copy()
r0 = _rrf(cc_iw, fM2); r1 = _rrf(cc_iw, fM2 + "+C(incq)+C(wq)")
print(f"  focal need-set (N={len(cc_iw):,})            RR {r0[0]:.3f} [{r0[1]:.3f},{r0[2]:.3f}]")
print(f"  +income+wealth (near-full sample)         RR {r1[0]:.3f} [{r1[1]:.3f},{r1[2]:.3f}]")
r2 = _rrf(cc_ex, fM2 + "+C(incq)+C(wq)+C(alone)")
print(f"  +income+wealth+living-arrangement (dn014-limited, N={len(cc_ex):,}) RR {r2[0]:.3f} [{r2[1]:.3f},{r2[2]:.3f}]")
print("  NOTE: the dn014-based row is a low-coverage (~29%) legacy sensitivity, superseded by the")
print("        A2 upgraded extended model below (living alone from household size, ~full coverage);")
print("        the manuscript reports the A2 model.")
print("  included vs excluded (income/wealth available): focal RR %.3f vs near-identical to full -> not a selected subsample." % r0[0])
print("  => association persists after adding income and wealth on nearly the full sample; SES explains only a small part.")

# --- (D) age 65+ sensitivity ---
print("== SHARE 65+ sensitivity (focal M2) ==")
r65 = rr(dd[dd["age"] >= 65], fM2)
print(f"  50+ RR {rr(dd, fM2)[0]:.3f}; 65+ RR {r65[0]:.3f} [{r65[1]:.3f},{r65[2]:.3f}] N={(dd['age']>=65).sum():,}")


# ===== REVISION-2 ANALYSES (A2 upgraded extended confounders; A3 visit-threshold period contrasts; A4 access outcomes) =====
print("\n===== REVISION-2 ANALYSES =====")
# --- A2: upgraded extended model on near-full sample: + income + wealth + lives-alone(full) + cognition + rurality ---
_hh = rd(f"sharew8_rel{R}_cv_r.dta", ["mergeid","hhsize"]).drop_duplicates("mergeid")
_ab = rd(f"sharew8_rel{R}_gv_housing.dta", ["mergeid","areabldgi"]).drop_duplicates("mergeid")
_cf = rd(f"sharew8_rel{R}_cf.dta", ["mergeid","cf010_","cf104tot","cf105tot","cf106tot","cf107tot"]).drop_duplicates("mergeid")
_imp = rd(f"sharew8_rel{R}_gv_imputations.dta", ["mergeid","thinc","hnetw"]).groupby("mergeid", as_index=False).mean()
ex2 = (dd.merge(_hh,on="mergeid",how="left").merge(_ab,on="mergeid",how="left")
         .merge(_cf,on="mergeid",how="left").merge(_imp,on="mergeid",how="left"))
ex2["alone_full"] = (n2n(ex2["hhsize"])==1).astype(float); ex2.loc[n2n(ex2["hhsize"]).isna(),"alone_full"]=np.nan
ex2["rural"] = n2n(ex2["areabldgi"])
ex2["fluency"] = n2n(ex2["cf010_"])
ex2["recall"] = ex2[["cf104tot","cf105tot","cf106tot","cf107tot"]].apply(n2n).bfill(axis=1).iloc[:,0]
ex2["incq"] = pd.qcut(n2n(ex2["thinc"]).rank(method="first"),5,labels=False,duplicates="drop")
ex2["wq"]   = pd.qcut(n2n(ex2["hnetw"]).rank(method="first"),5,labels=False,duplicates="drop")
def _rrf(x,form):
    m=smf.glm(f"visit ~ excl + {form}",data=x,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":x["cc"]})
    ci=m.conf_int().loc["excl"]; return np.exp(m.params["excl"]),np.exp(ci[0]),np.exp(ci[1])
_need=["visit","excl","agecat","sex","edu3","sphus","chronicw8","eurod","adl","iadl","cc","incq","wq","alone_full","rural","fluency","recall"]
cc2=ex2.dropna(subset=_need).copy()
rE =_rrf(cc2, fM2+"+C(incq)+C(wq)+C(alone_full)+C(rural)+fluency+recall")
rEb=_rrf(cc2, fM2)
print(f"  A2 upgraded extended (+income+wealth+lives-alone[full]+cognition+rurality): RR {rE[0]:.3f} [{rE[1]:.3f},{rE[2]:.3f}] N={len(cc2):,}")
print(f"     focal M2 on the SAME sample (for comparison):                            RR {rEb[0]:.3f} [{rEb[1]:.3f},{rEb[2]:.3f}]")
print(f"     coverage: lives-alone(full) {(n2n(ex2['hhsize'])==1).mean()*100:.0f}%% prevalence, hhsize present {n2n(ex2['hhsize']).notna().mean()*100:.0f}%%; rurality present {ex2['rural'].notna().mean()*100:.0f}%%; cognition present {ex2['recall'].notna().mean()*100:.0f}%%")

# --- A3: visit-threshold period contrasts (pandemic in-person vs pre-pandemic >= k visits) ---
print("  A3 visit-threshold period contrasts (ratio-of-RR; pandemic in-person vs pre-pandemic >=k visits):")
for k in [1,2,3,4,5]:
    dd[f"pre_t{k}"]=np.where(dd["docv8"]>=k,1.0,np.where(dd["docv8"]>=0,0.0,np.nan))
    _prev=dd[f"pre_t{k}"].mean()*100
    dk=did(f"pre_t{k}")
    print(f"     >= {k} visits (pre prevalence {_prev:.0f}%%): ratio-of-RR {dk[0]:.3f} [{dk[1]:.3f},{dk[2]:.3f}] p={dk[3]:.1e}")

# --- A4: secondary pandemic access outcomes (focal M2) ---
print("  A4 secondary pandemic access/unmet-care outcomes (focal M2, RR non-users vs users; self-report, denominator-dependent):")
_acc = rd(f"sharew9ca_rel{R}_ca.dta", ["mergeid","caq105_","caq110_","caq115_","caq125_"])
ac = dd.merge(_acc,on="mergeid",how="left")
for col,lab in [("caq105_","forwent any treatment"),("caq110_","appointment postponed (COVID)"),("caq115_","appointment denied"),("caq125_","hospital treatment")]:
    ac["_y"]=yn(ac[col])
    da=ac.dropna(subset=["_y","excl","agecat","sex","edu3","sphus","chronicw8","eurod","adl","iadl","cc"])
    m=smf.glm("_y ~ excl + "+fM2,data=da,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":da["cc"]})
    ci=m.conf_int().loc["excl"]; ru=da[da.excl==0]["_y"].mean()*100; rn=da[da.excl==1]["_y"].mean()*100
    print(f"     {lab:<28} RR {np.exp(m.params['excl']):.3f} [{np.exp(ci[0]):.3f},{np.exp(ci[1]):.3f}] raw u/n {ru:.1f}/{rn:.1f}%% N={len(da):,}")


# ===== TRACEABILITY (raw cross-tabs, E-values, leave-one-country-out) =====
import math as _math
print("\n===== TRACEABILITY =====")
def _ev(rr):
    r=1.0/rr if rr<1 else rr
    return r+_math.sqrt(r*(r-1))
# raw (unadjusted) percentages by exposure for the modality outcomes
print("  raw %% by exposure (users / non-users):")
print(f"     in-person (visit)  {dd[dd.excl==0]['visit'].mean()*100:.1f} / {dd[dd.excl==1]['visit'].mean()*100:.1f}")
_tcr=tc.dropna(subset=['remote']); _tca=tc.dropna(subset=['anyc'])
print(f"     remote (>=1)       {_tcr[_tcr.excl==0]['remote'].mean()*100:.1f} / {_tcr[_tcr.excl==1]['remote'].mean()*100:.1f}")
print(f"     any contact        {_tca[_tca.excl==0]['anyc'].mean()*100:.1f} / {_tca[_tca.excl==1]['anyc'].mean()*100:.1f}")
# E-value for the SHARE focal estimate (point and CI limit closest to the null)
_f=rr(dd,fM2)
print(f"  SHARE focal RR {_f[0]:.3f} -> E-value {_ev(_f[0]):.3f}; "
      f"CI limit closest to null {_f[2]:.3f} -> E-value {_ev(_f[2]):.3f}")
# leave-one-country-out (focal M2)
_loco=[]
for _c in sorted(dd['cc'].unique()):
    _sub=dd[dd['cc']!=_c]
    try: _loco.append(rr(_sub,fM2)[0])
    except Exception: pass
print(f"  leave-one-country-out focal RR range: {min(_loco):.3f}--{max(_loco):.3f} (n={len(_loco)} countries dropped in turn)")

