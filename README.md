# Analysis code — Digital Exclusion and Doctor Contact During COVID-19 Among Adults Aged 50 and Older: Pandemic-Specific Evidence From SHARE With HRS Context

Authors: Zihan Zhu (Columbia University); Kairan Shen (University of California, Berkeley; corresponding, kairanshen@berkeley.edu)

This repository contains the analysis code that reproduces every estimate in the manuscript and supplement (saved outputs included). It is an observational secondary analysis of two ageing cohorts (SHARE and HRS).

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
