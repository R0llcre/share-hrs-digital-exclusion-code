#!/usr/bin/env python
"""
PAPER 2 WILD CLUSTER BOOTSTRAP: restricted (null-imposed) wild cluster
bootstrap-t, Rademacher weights, B=9999, 26 SHARE country clusters, on
LPM (risk-difference) analogues of the Paper 2 estimands:

  (1) W9 any-visit level RD (excl), covariate-complete sample n=37,378
  (2) W9 4+-visit level RD
  (3) excl x period interaction, stacked fully-interacted LPM,
      W8(2019) -> W9(2021-22), any-visit baseline
  (4) same, 4+-visit baseline

Run from this directory:  venv/bin/python paper2_bootstrap.py
Writes PAPER2_BOOTSTRAP.txt to the current directory.
"""
import pyreadstat, os, numpy as np, pandas as pd
from patsy import dmatrices, dmatrix

U, R = "../unzipped", "9-0-0"
rng = np.random.default_rng(20260702)

def rd(fn, cols):
    df, _ = pyreadstat.read_dta(os.path.join(U, fn), usecols=cols)
    return df
def n2n(s): return s.where(s >= 0, np.nan)
def yn(s):  return n2n(s).map({1: 1.0, 5: 0.0})

# ---- analytic sample: mirrors paper2_analysis.py exactly ----
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
d["v9_4p"]  = np.where(d["docv9"] >= 4, 1.0, np.where(d["docv9"] >= 0, 0.0, np.nan))
DROP = ["v9_any", "excl", "agecat", "sex", "edu3", "sphus", "chronicw8",
        "eurod", "adl", "iadl", "docv8", "hosp8", "cc"]
dd = d.dropna(subset=DROP).copy()
dd["pre_any"] = np.where(dd["docv8"] >= 1, 1.0, 0.0)
dd["pre_4"]   = np.where(dd["docv8"] >= 4, 1.0, 0.0)
print(f"analytic sample n={len(dd):,}", flush=True)

FM2 = "C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl+C(cc)"

def wcb_core(X, y, clu, j, B=9999):
    clusters = np.unique(clu)
    G = len(clusters); n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    Xg = [X[clu == g] for g in clusters]
    XtXinv = np.linalg.inv(X.T @ X)
    beta = XtXinv @ (X.T @ y); u = y - X @ beta
    meat = np.zeros((k, k))
    for g_i, g in enumerate(clusters):
        sc = Xg[g_i].T @ u[clu == g]; meat += np.outer(sc, sc)
    V = adj * XtXinv @ meat @ XtXinv
    b_hat = beta[j]; se = np.sqrt(V[j, j]); t_hat = b_hat / se
    Xr = np.delete(X, j, axis=1)
    betar = np.linalg.solve(Xr.T @ Xr, Xr.T @ y)
    fitr_g = [Xr[clu == g] @ betar for g in clusters]
    ur_g   = [y[clu == g] - f for f, g in zip(fitr_g, clusters)]
    cnt = 0
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=G)
        Xty = np.zeros(k)
        ystar_g = [f + w[g_i] * u_ for g_i, (f, u_) in enumerate(zip(fitr_g, ur_g))]
        for g_i in range(G):
            Xty += Xg[g_i].T @ ystar_g[g_i]
        beta_s = XtXinv @ Xty
        meat = np.zeros((k, k))
        for g_i in range(G):
            sc = Xg[g_i].T @ (ystar_g[g_i] - Xg[g_i] @ beta_s)
            meat += np.outer(sc, sc)
        Vs = adj * XtXinv @ meat @ XtXinv
        if abs(beta_s[j] / np.sqrt(Vs[j, j])) >= abs(t_hat):
            cnt += 1
    return b_hat, se, t_hat, (cnt + 1) / (B + 1), n

def wcb_level(var):
    sub = dd.dropna(subset=[var]).copy()
    yv, X = dmatrices(f"{var} ~ excl + {FM2}", sub, return_type="dataframe")
    j = list(X.columns).index("excl")
    return wcb_core(X.values, yv.values.ravel(), sub["cc"].values, j)

def wcb_interaction(precol, postcol):
    cols = ["excl", "agecat", "sex", "edu3", "sphus", "chronicw8", "eurod", "adl", "iadl", "cc"]
    sub = dd.dropna(subset=[precol, postcol])
    lo = pd.concat([sub[cols + [precol]].rename(columns={precol: "y"}).assign(per=0.0),
                    sub[cols + [postcol]].rename(columns={postcol: "y"}).assign(per=1.0)],
                   ignore_index=True)
    X = dmatrix(f"excl*per + ({FM2})*per", lo, return_type="dataframe")
    j = list(X.columns).index("excl:per")
    return wcb_core(X.values, lo["y"].astype(float).values, lo["cc"].values, j)

out = ["PAPER 2 WILD CLUSTER BOOTSTRAP-t (restricted, Rademacher, B=9999, 26 clusters)",
       f"analytic sample n={len(dd):,} (mirrors paper2_analysis.py)", ""]
for var, lab in [("v9_any", "W9 any doctor visit, level RD (M2)"),
                 ("v9_4p",  "W9 4+ doctor visits, level RD (M2)")]:
    bh, se, t, p, n = wcb_level(var)
    line = f"  {lab:<46} RD={bh*100:+.2f}pp  CRVE-t={t:.2f}  wild-cluster p={p:.4f}  N={n:,}"
    print(line, flush=True); out.append(line)
out.append("")
out.append("  excl x period interaction, stacked fully-interacted LPM, W8(2019)->W9(2021-22):")
print(out[-1], flush=True)
for precol, postcol, lab in [("pre_any", "v9_any", "any-visit baseline"),
                             ("pre_4",   "v9_4p",  ">=4-visit baseline")]:
    bh, se, t, p, n = wcb_interaction(precol, postcol)
    line = f"  {lab:<46} coef={bh*100:+.2f}pp  CRVE-t={t:.2f}  wild-cluster p={p:.4f}  N(stacked)={n:,}"
    print(line, flush=True); out.append(line)
out.append("")
out.append("  smallest reportable p = 1/(B+1) = 0.0001; a LARGE p here supports the null contrast.")
open("PAPER2_BOOTSTRAP.txt", "w").write("\n".join(out) + "\n")
print("written: PAPER2_BOOTSTRAP.txt", flush=True)
