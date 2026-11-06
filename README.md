# SABER–MLS figure reproduction v2

Rebuild the four redesigned main figures and all twelve Supporting Information figures from frozen scientific data. Each figure is saved as a 600 dpi PNG and vector PDF/SVG. This archive contains plotting and numerical verification only: it does not train models, generate predictions, rerun bootstrap, or construct p-values.

## Quick start

Use Python 3.12 (tested with 3.12.14). Extract `GRL_Figure_Data_v2.zip` next to this repository so that both directories share a parent. No GPU, online map service, Cartopy download, or external font is needed.

```bash
cd GRL_Figure_Repository_v2
python -m pip install -r requirements.txt
python run_figures.py --figures all --dpi 600
```

An explicit data location is also supported:

```bash
python run_figures.py --data /path/to/GRL_Figure_Data_v2 --out build --figures all
```

For only main figures use `--figures main`; for SI use `--figures si`; for numerical verification without rendering use `--figures none`. Verification always runs and aborts on a mismatch. Expected outputs are `build/figures/Figure_1.*`–`Figure_4.*` and `Figure_S1.*`–`Figure_S12.*`, three derived CSVs under `build/tables/`, and `build/BUILD_REPORT.json`. A full build takes approximately one minute in the supplied environment; hardware and filesystem speed affect runtime.

## Scientific rules

- Pool SSE and valid counts within each seed and selection before taking RMSE, then average three seed scores. Never average daily RMSEs or predictions across seeds.
- 60 pressure-reference nodes (z* = 20–79) are available; 59 (21–79) are scored. z* is not measured geometric altitude. Main Figure 2 uses the original fixed reference-node bands; Figure 4c–d uses physical hPa bands.
- Main Figure 3 contains 8,496 monthly node/budget cells on a true log-pressure axis, with no smoothing. Its shared color limits are ±1 K with end triangles; 537 cells lie beyond these limits. The full range is −3.487643 to +2.476347 K, and every value is saved in `monthly_pressure_all_budgets.csv`.
- Annual intervals and Hunga intervals belong to separate families of nine. Frozen 14-day intervals support 4/9 annual comparisons and 8/9 Hunga windows. Ordinary, familywise, and descriptive quantities retain their original meanings.
- Figure 4 observed temperature contrasts use equally weighted valid daily means and equal reference-year weights. Model RMSE contrasts use pooled model errors. The two quantities have separate axes and imply no causal attribution.
- All evaluation years are retrospective; the 2025 boundary indicates chronology, not a prospective lockbox claim.

## Verified results

The supplied build checked 30 input payload hashes and 30,608 numerical values against frozen tables, with maximum absolute discrepancy 5.6843e−14 K. At 1% training, pooled 2022–2024 Transfer RMSE is 4.3587923143 K and Scratch RMSE is 4.7094562242 K (7.4459532743% reduction). The archived bootstrap intervals are verified as inputs but are not regenerated.

## Files

| Location | Content |
|---|---|
| `run_figures.py` | Complete offline command-line entry point |
| `src/grlfig/data.py` | Daily-statistics aggregation and derived tables |
| `src/grlfig/main.py` | Four redesigned main figures |
| `src/grlfig/si.py` | SI S1–S12 from archived numerical data |
| `src/grlfig/verify.py` | Numerical checks and frozen-interval checks |
| `src/grlfig/style.py` | Print-sized styles and log-pressure cells |
| `assets/` | Regional land vectors and exact provenance |
| `FIGURE_INVENTORY.csv` | Per-figure input inventory |
| `docs/` | Captions, replacement prose, change notes, upload instructions |
| `verification/` | Recorded numerical build report |
| `publication_metadata.json` | Real author, license, DOI and repository fields to fill |
| `CITATION.cff.in` | Citation template; finalize as CITATION.cff after filling metadata |

The land vector is derived from the supplied v1 Figure_1.svg, originally rendered from Natural Earth 1:50m land v4.1.0. Longitude/latitude were recovered by reversing the SVG axes' affine transform; the original shapefile is not asserted to be included. Scientific statistics were never digitized from plots. See `assets/MAP_PROVENANCE.json`.

## Publication

This is the plotting release, not a substitute for any existing full research software archive. Put expanded source files on GitHub, companion data in a Dataset record, and the fixed GitHub Release source ZIP in a Software record. If these research records already exist, create the appropriate new version instead of a duplicate. See `docs/UPLOAD_GUIDE_CN.txt` and the supplied Word guide.

Authors, affiliations, licenses, real repository URL and two version DOIs have not been supplied. Fill `publication_metadata.json`, finalize `CITATION.cff.in`, add the selected full license text, and update the publication links before public release. No new license is applied by this package. Do not cite invented DOIs or claim that this plotting package includes training or resampling code.
