#!/usr/bin/env python
"""
Restricted (null-imposed) wild cluster bootstrap-t, Rademacher weights, B=9999,
over the 26 SHARE country clusters, on linear-probability (risk-difference)
specifications of the focal models.

Run from this directory:  venv/bin/python wild_bootstrap.py
Writes WILD_BOOTSTRAP_RESULTS.txt to the current directory.

(1) During-pandemic in-person visit (caq120, SCS2 2021), focal M2 covariates,
    on the SAME covariate-complete analytic sample as share_analysis.py
    (complete-case including baseline utilisation docv8/hosp8) -> N=30,244.
(2) Postponed care (caq110) on the same sample definition.
(3) Internet-use x period interaction in the stacked fully-interacted model
    (covariate x period + country x period; LPM analogue of the modified-Poisson
    period contrast), pre-pandemic baseline = any doctor visit and >=4 visits.
    The bootstrapped coefficient is the additive (risk-difference-scale)
    interaction; the Poisson ratio-of-RR estimates are reported in
    share_analysis.py / SHARE_output.txt.
"""
import pyreadstat, os, numpy as np, pandas as pd
from patsy import dmatrices, dmatrix

U = "../unzipped"; R = "9-0-0"
rng = np.random.default_rng(20260701)

def rd(fn, cols):
    df, _ = pyreadstat.read_dta(os.path.join(U, fn), usecols=cols)
    return df
def n2n(s): return s.where(s >= 0, np.nan)
def yn(s):  return n2n(s).map({1: 1.0, 5: 0.0})

# ---- analytic sample: mirrors share_analysis.py exactly ----
it8 = rd(f"sharew8_rel{R}_it.dta", ["mergeid", "it004_"])
it8["excl"] = np.where(it8["it004_"] == 5, 1.0, np.where(it8["it004_"] == 1, 0.0, np.nan))
cvr = rd(f"sharew8_rel{R}_cv_r.dta", ["mergeid", "country", "gender", "age2020"]).drop_duplicates("mergeid")
isc = rd(f"sharew8_rel{R}_gv_isced.dta", ["mergeid", "isced1997_r"])
gh  = rd(f"sharew8_rel{R}_gv_health.dta", ["mergeid", "sphus", "chronicw8", "eurod", "adl", "iadl"])
hc  = rd(f"sharew8_rel{R}_hc.dta", ["mergeid", "hc602_", "hc012_"])
s2  = rd(f"sharew9ca_rel{R}_ca.dta", ["mergeid", "caq110_", "caq120_"])

b = (cvr.merge(it8[["mergeid", "excl"]], on="mergeid")
        .merge(isc, on="mergeid", how="left")
        .merge(gh, on="mergeid", how="left")
        .merge(hc, on="mergeid", how="left"))
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

d = b.merge(s2, on="mergeid", how="inner")
d["visit"] = yn(d["caq120_"]); d["postp"] = yn(d["caq110_"])
DROP = ["visit", "excl", "agecat", "sex", "edu3", "sphus", "chronicw8",
        "eurod", "adl", "iadl", "docv8", "hosp8", "cc"]
dd = d.dropna(subset=DROP).copy()          # the analytic sample (N=30,244)
dd["pre_any"] = np.where(dd["docv8"] >= 1, 1.0, 0.0)
dd["pre_4"]   = np.where(dd["docv8"] >= 4, 1.0, 0.0)

FM2 = "C(agecat)+C(sex)+C(edu3)+sphus+chronicw8+eurod+adl+iadl+C(cc)"

def wcb_core(X, y, clu, j, B=9999):
    """Restricted wild cluster bootstrap-t (Rademacher) for coefficient j of an LPM."""
    clusters = np.unique(clu)
    G = len(clusters); n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    Xg = [X[clu == g] for g in clusters]
    XtXinv = np.linalg.inv(X.T @ X)
    # unrestricted fit + CRVE t
    beta = XtXinv @ (X.T @ y); u = y - X @ beta
    meat = np.zeros((k, k))
    for g_i, g in enumerate(clusters):
        sc = Xg[g_i].T @ u[clu == g]; meat += np.outer(sc, sc)
    V = adj * XtXinv @ meat @ XtXinv
    b_hat = beta[j]; se = np.sqrt(V[j, j]); t_hat = b_hat / se
    # restricted fit under H0: beta_j = 0
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

def wcb_interaction(precol):
    cols = ["excl", "agecat", "sex", "edu3", "sphus", "chronicw8", "eurod", "adl", "iadl", "cc"]
    lo = pd.concat([dd[cols + [precol]].rename(columns={precol: "y"}).assign(per=0.0),
                    dd[cols + ["visit"]].rename(columns={"visit": "y"}).assign(per=1.0)],
                   ignore_index=True)
    X = dmatrix(f"excl*per + ({FM2})*per", lo, return_type="dataframe")
    j = list(X.columns).index("excl:per")
    return wcb_core(X.values, lo["y"].astype(float).values, lo["cc"].values, j)

out = ["WILD CLUSTER BOOTSTRAP-t (restricted, Rademacher, B=9999, 26 country clusters)",
       "LPM (risk-difference) specifications on the covariate-complete analytic sample",
       "(same complete-case definition as share_analysis.py, incl. baseline utilisation)", ""]
for var, lab in [("visit", "caq120 in-person visit (during, focal M2)"),
                 ("postp", "caq110 appointment postponed (COVID)")]:
    bh, se, t, p, n = wcb_level(var)
    out.append(f"  {lab:<44} RD={bh*100:+.2f}pp  CRVE-t={t:.2f}  wild-cluster p={p:.4f}  N={n:,}")
out.append("")
out.append("  internet-use x period interaction, stacked fully-interacted LPM")
out.append("  (additive-scale analogue of the Poisson ratio-of-RR period contrast):")
for precol, lab in [("pre_any", "pre-pandemic baseline: any doctor visit"),
                    ("pre_4",   "pre-pandemic baseline: >=4 doctor visits")]:
    bh, se, t, p, n = wcb_interaction(precol)
    out.append(f"  {lab:<44} coef={bh*100:+.2f}pp  CRVE-t={t:.2f}  wild-cluster p={p:.4f}  N(stacked)={n:,}")
out.append("")
out.append("  smallest reportable p = 1/(B+1) = 0.0001")
rep = "\n".join(out)
open("WILD_BOOTSTRAP_RESULTS.txt", "w").write(rep + "\n")
print(rep)
