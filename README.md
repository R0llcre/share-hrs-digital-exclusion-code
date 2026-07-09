# Analysis code — digital exclusion and doctor contact in ageing cohorts (SHARE/HRS)

Authors: Zihan Zhu (Columbia University); Kairan Shen (University of California, Berkeley; corresponding, kairanshen@berkeley.edu)

This repository contains the analysis code that reproduces every estimate in two related manuscripts and their supplements (saved outputs included):

1. **Paper 1** — *Digital Exclusion and Doctor Contact During COVID-19 Among Adults Aged 50 and Older: Pandemic-Specific Evidence From SHARE With HRS Context* (`share_analysis.py`, `hrs_analysis.py`, `wild_bootstrap.py`, `make_main_figs.py`, `make_forest_figs.py`).
2. **Paper 2** — *Internet Non-Use and Doctor Contact After the Acute Phase of COVID-19 Among Adults Aged 50 and Older: Like-for-Like Evidence From SHARE* (`paper2_*.py`; see the Paper 2 section below).

Both are observational secondary analyses of ageing cohorts (SHARE and HRS).

## ⚠️ Data are NOT included
The raw SHARE and HRS data **cannot be redistributed** under their data-use agreements. This repository contains **code and saved output logs only**. To reproduce the results, obtain the data yourself (free, on registration) from the official providers and point the scripts at your local copies:

- **SHARE** (Release 9.0.0): SHARE Research Data Centre — https://share-eric.eu/data
  Modules used: Wave 6 `hc` (2015 doctor visits); Wave 8 `hc` (doctor visits, hospital, health literacy `hc889_`), `it` (internet use `it004_`), `cv_r` (demographics, interview date, `hhsize`), `dn`, `cf` (cognition), `gv_isced`, `gv_health`, `gv_imputations` (income `thinc`, wealth `hnetw`), `gv_housing` (`areabldgi` rurality); Corona Survey 1 (`sharew8ca…ca`, `caq020_`) and Corona Survey 2 (`sharew9ca…ca`, items `caq105_`, `caq110_`, `caq115_`, `caq118_`, `caq120_`, `caq125_`) with the calibrated weight (`gv_weights_ca`, `cciw_w9ca_ca`).
  Dataset DOIs: 10.6103/SHARE.w6.900, 10.6103/SHARE.w8.900, 10.6103/SHARE.w8ca.900, 10.6103/SHARE.w9ca.900.
- **HRS** (2014–2020 Core incl. 2020 COVID module) + **RAND HRS Longitudinal File 2022**: HRS Data Portal — https://hrsdata.isr.umich.edu
  Variables used: 2018 Core internet item `QW303`; RAND `r12doctor`–`r15doctor`, wave-14 covariates (incl. `r14adl5a`/`r14iadl5a`), income/wealth, `raracem`/`rahispan`, insurance, cognition, weights/strata/PSU (`r15wtresp`, `raestrat`, `raehsamp`); 2020 COVID module `RCOVW579` (`H20COV_R`).

## Files
| File | What it does |
|------|--------------|
| `share_analysis.py` | SHARE arm: STROBE flow, need-adjustment ladder M0–M3, adjusted risks/RD, weighting + IPW + exposure-timing sensitivity, multimorbidity-stratified RDs, effect modification, subgroups, between-country heterogeneity, temporal (Corona Survey 1 vs 2), before/during period contrasts, multi-wave pre-trend (2015/2019/2021), total-contact/remote modality (raw + adjusted-standardised %), attrition comparison, extended-confounder models (income/wealth; + living alone, cognition, rurality), visit-threshold contrasts, secondary unmet-care/hospital outcomes, 65+ restriction, E-values (point + CI-limit), leave-one-country-out. Writes `SHARE_output.txt`. |
| `hrs_analysis.py` | HRS arm: flow, ladder M0–M3, focal + design-weighted estimate, adjusted risks/RD, Table 1, delayed-care secondary outcome, before/during contrast, multi-wave pre-trend (2014–2020), extended-confounder model, 65+ restriction, raw percentages and E-values (point + CI-limit). Writes `HRS_output.txt`. |
| `wild_bootstrap.py` | Restricted wild cluster bootstrap-t (Rademacher, B=9,999) over the 26 SHARE country clusters, on linear-probability specifications and the exact analytic sample (n=30,244): the focal during-pandemic association, postponed care, and the internet-use × period interaction for the any-visit and ≥4-visits baselines. Writes `WILD_BOOTSTRAP_RESULTS.txt`. |
| `make_forest_figs.py` | Recomputes the subgroup, per-country (incl. the DerSimonian–Laird random-effects pooled estimate) and leave-one-country-out forest figures and writes the supplement figure PDFs/PNGs. |
| `make_main_figs.py` | Draws the main-text figures (wave trajectories; need gradient) and the ladder/temporal supplement figures from the published estimates in the saved outputs. |
| `SHARE_output.txt`, `HRS_output.txt`, `WILD_BOOTSTRAP_RESULTS.txt` | Saved outputs (the reproduced numbers, regenerated 2026-07-01). |

## Paper 2 files (SHARE Wave 9 like-for-like persistence analysis)

Additional SHARE modules used: Wave 9 `hc` (doctor visits `hc602_`), `it` (internet use `it004_`), `cv_r` (interview date, vital status), `xt` (end-of-life interviews, year of death), `gv_weights` (`cciw_w9`). Dataset DOI added: 10.6103/SHARE.w9.900.

| File | What it does |
|------|--------------|
| `paper2_feasibility.py` | Initial feasibility run: Wave-9 adjusted RRs, like-for-like W8→W9 fully-interacted period contrasts, digital-transition crosstab. |
| `paper2_analysis.py` | Main analysis: flow with complete-case decomposition; period-specific RRs and contrasts from the fully-interacted model (incl. the within-survivor and full-pool 2019 baselines); one-sided equivalence tests; fieldwork-timing dose-response; retention/mortality decomposition, mortality RR, IPW-for-retention, extreme-case bounds; digital-transition groups and stable-exposure sensitivity; country heterogeneity (DerSimonian–Laird, leave-one-country-out); Corona-Survey in-person and total-contact bridges inside the analytic sample; threshold ladder; calibrated-weight sensitivity. Writes `PAPER2_ANALYSIS.txt`. |
| `paper2_bootstrap.py` | Restricted wild cluster bootstrap-t (Rademacher, B=9,999, 26 clusters) on the Wave-9 level risk differences and the two period-interaction contrasts. Writes `PAPER2_BOOTSTRAP.txt`. |
| `paper2_figdata.py` | Full-precision (4 dp) estimates behind every figure value, incl. the 2015 (Wave 6) anchors. Writes `PAPER2_FIGDATA.txt`. |
| `paper2_table1.py` | Table 1 baseline characteristics by exposure with standardised mean differences. Writes `PAPER2_TABLE1.txt`. |
| `paper2_make_figs.py` | Draws the Paper 2 figures (same-instrument trajectory with the flagged in-person anchor; fieldwork-timing dose-response) from the committed estimates. |
| `PAPER2_*.txt` | Saved outputs (regenerated 2026-07-02). |

## Environment
Python 3.x with: `pandas`, `numpy`, `pyreadstat`, `statsmodels`, `scipy`, `patsy`.

```
pip install pandas numpy pyreadstat statsmodels scipy patsy
```

## Reproduce
1. Download the SHARE and HRS data (above) and edit the `U` (SHARE) / `H` (HRS) path variables at the top of each script to your local data directories.
2. Run:
   ```
   python share_analysis.py  > SHARE_output.txt
   python hrs_analysis.py    > HRS_output.txt
   python wild_bootstrap.py            # writes WILD_BOOTSTRAP_RESULTS.txt itself
   ```
3. The printed estimates match the manuscript tables and figures. Note: `SHARE_output.txt` also contains one clearly-labelled legacy sensitivity row (the low-coverage dn014-based living-arrangement model), superseded by the full-coverage extended model reported in the manuscript.

## Notes
- Exposure (internet non-use) is measured **before** the pandemic; the before/during comparison is a descriptive period-by-internet-use interaction, **not** a causal difference-in-differences.
- Wild-cluster p-values are computed on linear-probability (risk-difference) specifications; the modified-Poisson RR estimates and CIs are produced by the analysis scripts.
- This is an association study; it does not establish causality or unmet need.

## License
Code released under the MIT License.
