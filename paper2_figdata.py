#!/usr/bin/env python
"""
PAPER 2 FIGURE DATA: the 2015 (Wave 6) anchor points of Figure 1, estimated
inside the same W9 analytic frame as paper2_analysis.py (W8 exposure, M2
covariates measured at W8 -- descriptive trajectory, disclosed in caption).
All other figure values come from PAPER2_ANALYSIS.txt.

Run from this directory:  venv/bin/python paper2_figdata.py
Appends nothing; writes PAPER2_FIGDATA.txt.
"""
import pyreadstat, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
np.seterr(all="ignore")

U, R = "../unzipped", "9-0-0"
def n2n(s): return s.where(s >= 0, np.nan)
def yn(s):  return n2n(s).map({1: 1.0, 5: 0.0})
def rd(fn, c): return pyreadstat.read_dta(f"{U}/{fn}", usecols=c)[0]

# ---- analytic frame: mirrors paper2_analysis.py exactly ----
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

hc9 = rd(f"sharew9_rel{R}_hc.dta", ["mergeid", "hc602_"]).rename(columns={"hc602_": "hc602w9"})
d = b.merge(hc9, on="mergeid", how="inner")
d["docv9"] = n2n(d["hc602w9"])
d["v9_any"] = np.where(d["docv9"] >= 1, 1.0, np.where(d["docv9"] >= 0, 0.0, np.nan))
DROP = ["v9_any", "excl", "agecat", "sex", "edu3", "sphus", "chronicw8",
        "eurod", "adl", "iadl", "docv8", "hosp8", "cc"]
dd = d.dropna(subset=DROP).copy()

# ---- Wave-6 (2015) outcome, same hc602 instrument ----
hc6 = rd(f"sharew6_rel{R}_hc.dta", ["mergeid", "hc602_"]).rename(columns={"hc602_": "hc602w6"})
assert hc6["mergeid"].is_unique
m6 = dd.merge(hc6, on="mergeid", how="inner")
m6["docv6"] = n2n(m6["hc602w6"])
m6["v6_any"] = np.where(m6["docv6"] >= 1, 1.0, np.where(m6["docv6"] >= 0, 0.0, np.nan))
m6["v6_4p"]  = np.where(m6["docv6"] >= 4, 1.0, np.where(m6["docv6"] >= 0, 0.0, np.nan))

fM2 = "C(agecat)+C(sex)+C(edu3)+C(cc)+sphus+chronicw8+eurod+adl+iadl"
def rr(col, x):
    sub = x.dropna(subset=[col])
    m = smf.glm(f"{col} ~ excl + {fM2}", data=sub, family=sm.families.Poisson()
                ).fit(cov_type="cluster", cov_kwds={"groups": sub["cc"]})
    ci = m.conf_int().loc["excl"]
    return np.exp(m.params["excl"]), np.exp(ci[0]), np.exp(ci[1]), int(m.nobs)

out = ["PAPER 2 FIGURE DATA: all Figure-1/2 point estimates at 4dp (half-up 2dp labels",
       "are taken from THESE full-precision values, never re-rounded from 3dp prints)",
       f"W9 analytic n={len(dd):,}; with W6 outcome n={m6['v6_any'].notna().sum():,}"]
for col, lab in [("v6_any", "2015 any doctor visit"), ("v6_4p", "2015 4+ doctor visits")]:
    r = rr(col, m6)
    out.append(f"  {lab}: RR {r[0]:.4f} [{r[1]:.4f},{r[2]:.4f}] (n={r[3]:,}; prev {m6[col].mean()*100:.1f}%)")

# 2019 / 2021-22 points on the full analytic frame (4dp for label rounding)
dd["v9_4p"]  = np.where(dd["docv9"] >= 4, 1.0, np.where(dd["docv9"] >= 0, 0.0, np.nan))
dd["pre_any"] = np.where(dd["docv8"] >= 1, 1.0, 0.0)
dd["pre_4"]   = np.where(dd["docv8"] >= 4, 1.0, 0.0)
for col, lab in [("pre_any", "2019 any doctor visit"), ("pre_4", "2019 4+ doctor visits"),
                 ("v9_any", "2021-22 any doctor visit"), ("v9_4p", "2021-22 4+ doctor visits")]:
    r = rr(col, dd)
    out.append(f"  {lab}: RR {r[0]:.4f} [{r[1]:.4f},{r[2]:.4f}] (n={r[3]:,})")

# SCS2 in-person bridge point (hollow marker in Figure 1)
ca2 = rd(f"sharew9ca_rel{R}_ca.dta", ["mergeid", "caq120_"]).drop_duplicates("mergeid")
br = dd.merge(ca2, on="mergeid", how="inner")
br["visit_ip"] = yn(br["caq120_"])
r = rr("visit_ip", br)
out.append(f"  2021 in-person (SCS2 bridge): RR {r[0]:.4f} [{r[1]:.4f},{r[2]:.4f}] (n={r[3]:,})")
rep = "\n".join(out)
open("PAPER2_FIGDATA.txt", "w").write(rep + "\n")
print(rep)
