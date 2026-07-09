#!/usr/bin/env python
"""
PAPER 2 TABLE 1: baseline (Wave 8) characteristics of the W9 analytic sample
(n=37,378) by pre-pandemic internet use, with standardised mean differences.
Row set mirrors Paper 1's Table 1 (SHARE panel).

Run from this directory:  venv/bin/python paper2_table1.py
Writes PAPER2_TABLE1.txt.
"""
import pyreadstat, numpy as np, pandas as pd
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

# ---- table rows ----
dd["age5064"] = (dd["agecat"] == "a").astype(float)
dd["age6579"] = (dd["agecat"] == "b").astype(float)
dd["age80"]   = (dd["agecat"] == "c").astype(float)
dd["female"]  = (dd["sex"] == "f").astype(float)
dd["edulo"]   = (dd["edu3"] == "lo").astype(float)
dd["edumi"]   = (dd["edu3"] == "mi").astype(float)
dd["eduhi"]   = (dd["edu3"] == "hi").astype(float)
dd["srhfp"]   = (dd["sphus"] >= 4).astype(float)          # fair/poor self-rated health
dd["mm2"]     = (dd["chronicw8"] >= 2).astype(float)      # multimorbidity
dd["eurod4"]  = (dd["eurod"] >= 4).astype(float)          # depressive symptoms
dd["anyadl"]  = (dd["adl"] >= 1).astype(float)
dd["anyiadl"] = (dd["iadl"] >= 1).astype(float)
dd["anyai"]   = ((dd["adl"] >= 1) | (dd["iadl"] >= 1)).astype(float)
dd["docv8any"] = (dd["docv8"] >= 1).astype(float)
dd["docv84p"]  = (dd["docv8"] >= 4).astype(float)

CONT = {"age": "Age, mean (SD)", "chronicw8": "Chronic conditions, mean (SD)",
        "eurod": "EURO-D score, mean (SD)", "docv8": "Doctor visits 2019, mean (SD)"}
BIN = {"age5064": "Age 50-64, %", "age6579": "Age 65-79, %", "age80": "Age 80+, %",
       "female": "Female, %", "edulo": "Education low (ISCED 0-2), %",
       "edumi": "Education medium (ISCED 3-4), %", "eduhi": "Education high (ISCED 5-6), %",
       "srhfp": "Fair/poor self-rated health, %", "mm2": "2+ chronic conditions, %",
       "eurod4": "EURO-D >= 4, %", "anyadl": "Any ADL limitation, %",
       "anyiadl": "Any IADL limitation, %", "anyai": "Any ADL or IADL limitation, %",
       "docv8any": "Any doctor visit 2019, %", "docv84p": "4+ doctor visits 2019, %",
       "hosp8": "Hospital stay 2019, %"}

u, n = dd[dd.excl == 0], dd[dd.excl == 1]
def smd_bin(p1, p0):
    den = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return abs(p1 - p0) / den if den > 0 else np.nan
def smd_cont(x1, x0):
    den = np.sqrt((x1.std()**2 + x0.std()**2) / 2)
    return abs(x1.mean() - x0.mean()) / den if den > 0 else np.nan

out = ["PAPER 2 TABLE 1: Wave-8 baseline characteristics by pre-pandemic internet use",
       f"  users n={len(u):,} | non-users n={len(n):,}  (analytic sample {len(dd):,})",
       f"  {'row':<36}{'users':>12}{'non-users':>12}{'SMD':>8}"]
for k, lab in CONT.items():
    out.append(f"  {lab:<36}{u[k].mean():>7.1f} ({u[k].std():.1f})"
               f"{n[k].mean():>7.1f} ({n[k].std():.1f}){smd_cont(u[k], n[k]):>8.2f}")
for k, lab in BIN.items():
    p1, p0 = u[k].mean(), n[k].mean()
    out.append(f"  {lab:<36}{p1*100:>11.1f}%{p0*100:>11.1f}%{smd_bin(p1, p0):>8.2f}")
rep = "\n".join(out)
open("PAPER2_TABLE1.txt", "w").write(rep + "\n")
print(rep)
