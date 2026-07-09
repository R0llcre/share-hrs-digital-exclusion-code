#!/usr/bin/env python
"""
PAPER 2 FEASIBILITY: did the digital-exclusion gap in doctor contact persist
after the acute pandemic, or close?

Exposure : SHARE Wave 8 it004_ (pre-pandemic internet non-use), as in paper 1.
Outcome  : Wave 9 CAPI (2021-22) hc602_ doctor visits, 12-month recall --
           THE SAME INSTRUMENT as the pre-pandemic W6/W8 outcomes (like-for-like).
Key numbers:
  (1) W9 any-visit and 4+-visit adjusted RRs (focal M2 covariates, country FE,
      country-clustered SE) -- compare with pre-pandemic 0.968/0.941 and the
      pandemic in-person 0.887.
  (2) Fully-interacted W8->W9 period contrasts on the identical item.
  (3) Digital catch-up: W9 it004 among W8 non-users (who came online?).
Run:  venv/bin/python paper2_feasibility.py
"""
import pyreadstat, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
import patsy
np.seterr(all="ignore")

U, R = "../unzipped", "9-0-0"
def n2n(s): return s.where(s >= 0, np.nan)
def yn(s):  return n2n(s).map({1: 1.0, 5: 0.0})
def rd(fn, c): return pyreadstat.read_dta(f"{U}/{fn}", usecols=c)[0]

# ---- Wave-8 base: identical to paper 1's construction ----
it8 = rd(f"sharew8_rel{R}_it.dta", ["mergeid", "it004_"])
it8["excl"] = np.where(it8["it004_"] == 5, 1.0, np.where(it8["it004_"] == 1, 0.0, np.nan))
cvr = rd(f"sharew8_rel{R}_cv_r.dta", ["mergeid", "country", "gender", "age2020"]).drop_duplicates("mergeid")
isc = rd(f"sharew8_rel{R}_gv_isced.dta", ["mergeid", "isced1997_r"])
gh  = rd(f"sharew8_rel{R}_gv_health.dta", ["mergeid", "sphus", "chronicw8", "eurod", "adl", "iadl"])
hc8 = rd(f"sharew8_rel{R}_hc.dta", ["mergeid", "hc602_", "hc012_"])
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

# ---- Wave-9 outcome (2021-22): SAME item hc602 + repeated internet item ----
hc9 = rd(f"sharew9_rel{R}_hc.dta", ["mergeid", "hc602_"]).rename(columns={"hc602_": "hc602w9"})
it9 = rd(f"sharew9_rel{R}_it.dta", ["mergeid", "it004_"]).rename(columns={"it004_": "it004w9"})
d = b.merge(hc9, on="mergeid", how="inner")
N_w9 = len(d)
d["docv9"] = n2n(d["hc602w9"])
d["v9_any"] = np.where(d["docv9"] >= 1, 1.0, np.where(d["docv9"] >= 0, 0.0, np.nan))
d["v9_4p"]  = np.where(d["docv9"] >= 4, 1.0, np.where(d["docv9"] >= 0, 0.0, np.nan))
DROP = ["v9_any", "excl", "agecat", "sex", "edu3", "sphus", "chronicw8",
        "eurod", "adl", "iadl", "docv8", "hosp8", "cc"]
dd = d.dropna(subset=DROP).copy()
dd = dd.merge(it9, on="mergeid", how="left")
print("== PAPER 2 FEASIBILITY: SHARE W9 (2021-22) doctor contact by W8 internet non-use ==")
print(f"  flow: W8 exposure-known {N_expo:,} -> in W9 CAPI with hc602 {N_w9:,} -> covariate-complete {len(dd):,}")
print(f"  W9 prevalence: any visit {dd['v9_any'].mean()*100:.1f}%; 4+ visits {dd['v9_4p'].mean()*100:.1f}%")

fM2 = "C(agecat)+C(sex)+C(edu3)+C(cc)+sphus+chronicw8+eurod+adl+iadl"
def rr(col, x):
    m = smf.glm(f"{col} ~ excl + {fM2}", data=x.dropna(subset=[col]),
                family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": x.dropna(subset=[col])["cc"]})
    ci = m.conf_int().loc["excl"]
    return np.exp(m.params["excl"]), np.exp(ci[0]), np.exp(ci[1])

r_any = rr("v9_any", dd); r_4p = rr("v9_4p", dd)
print("== W9 adjusted RRs (focal M2; compare: pre-2019 any 0.968 / 4+ 0.941; pandemic in-person 0.887) ==")
print(f"  W9 2021-22 any doctor visit : RR {r_any[0]:.3f} [{r_any[1]:.3f},{r_any[2]:.3f}]")
print(f"  W9 2021-22 4+ doctor visits: RR {r_4p[0]:.3f} [{r_4p[1]:.3f},{r_4p[2]:.3f}]")
print(f"  raw any-visit %: users {dd[dd.excl==0]['v9_any'].mean()*100:.1f} vs non-users {dd[dd.excl==1]['v9_any'].mean()*100:.1f}")

# ---- Like-for-like W8 -> W9 period contrast (identical item, same respondents) ----
dd["pre_any"] = np.where(dd["docv8"] >= 1, 1.0, 0.0)
dd["pre_4"]   = np.where(dd["docv8"] >= 4, 1.0, 0.0)
def did(precol, postcol):
    cols = ["excl", "agecat", "sex", "edu3", "sphus", "chronicw8", "eurod", "adl", "iadl", "cc"]
    sub = dd.dropna(subset=[precol, postcol])
    lo = pd.concat([sub[cols + [precol]].rename(columns={precol: "y"}).assign(per=0.0),
                    sub[cols + [postcol]].rename(columns={postcol: "y"}).assign(per=1.0)], ignore_index=True)
    X = patsy.dmatrix(f"excl*per + (C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl+C(cc))*per",
                      lo, return_type="dataframe")
    m = sm.GLM(lo["y"].astype(float).values, X.values, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": lo["cc"].values})
    j = list(X.columns).index("excl:per"); ci = m.conf_int()
    return np.exp(m.params[j]), np.exp(ci[j, 0]), np.exp(ci[j, 1]), m.pvalues[j], len(sub)
print("== Like-for-like W8(2019) -> W9(2021-22) period contrasts (SAME hc602 item, same respondents) ==")
for pre, post, lab in [("pre_any", "v9_any", "any-visit baseline"), ("pre_4", "v9_4p", "4+-visit baseline")]:
    c = did(pre, post)
    print(f"  {lab:<20} ratio-of-RR {c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}] p={c[3]:.1e} (n={c[4]:,})")
print("  => ratio < 1: gap larger in 2021-22 than 2019 (persistence/scar); ~1: gap back to gradient (healing).")

# ---- Digital catch-up: W9 internet use among W8 non-users ----
dd["excl9"] = np.where(dd["it004w9"] == 5, 1.0, np.where(dd["it004w9"] == 1, 0.0, np.nan))
sub = dd.dropna(subset=["excl9"])
print("== Digital transition W8 -> W9 (it004, among W9 analytic sample) ==")
for k, g in [("W8 users", sub[sub.excl == 0]), ("W8 non-users", sub[sub.excl == 1])]:
    print(f"  {k}: N={len(g):,}; still/now non-user in W9: {g['excl9'].mean()*100:.1f}%")
came = sub[(sub.excl == 1) & (sub.excl9 == 0)]
print(f"  'digital catch-up' group (W8 non-user -> W9 user): n={len(came):,} "
      f"({len(came)/max(len(sub[sub.excl==1]),1)*100:.1f}% of W8 non-users)")
