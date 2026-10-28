# Figure QA and reproduction

## Compact layout revision, 2026-10-28

The figure contract remains a quantitative grid showing validation-error sensitivity to four hyperparameters, for the joint metric and its inflow/outflow components. This is a layout-only adaptation of the supplied Python figures. Every panel letter and parameter title now appears below its plot. Each flow panel also identifies Chengdu inflow or outflow below the plot, replacing the previous top column headings. The shared metric legend remains above the figure.

The joint figure is 160 × 106 mm, reduced from 160 × 122 mm (13.11% less height). The flow figure is 160 × 184 mm, reduced from 160 × 202 mm (8.91% less height). Top margins, column gaps and inter-row spacing are reduced while retaining the existing type sizes. Plot rectangles remain equal within each grid. The alignment helper measures the centered bottom-caption anchors explicitly, using the same 1.5 pt alignment tolerance without exemptions.

All 60 source rows and all 120 metric values are preserved: joint uses 20 rows / 40 values and flow uses 40 rows / 80 values. Full-precision values determine bar heights and minima; text shows four decimals. The source-file hashes are unchanged. PDF numeric text was compared with the entire source-value multiset. Green stars still mark each metric's minimum, bold ticks still mark reference configurations, and the MAE, RMSE and best-marker colors are unchanged.

Both final PDFs pass plot-area alignment and bottom-caption alignment with zero failures or warnings. Both final PDFs pass rendered collision checks with zero failures or warnings. The minimum rendered glyph size remains 6.3 pt, and ordinary numeric labels remain 7.3 pt. PDF/SVG text is editable; PNG exports use 600 dpi. Fresh PDF-derived 150 dpi previews were inspected panel by panel and as whole figures. No clipped values, overlapping stars, mixed flow identities or caption collisions were observed.

The source validator reports 19 PASS, 0 FAIL and 2 reviewed WARN. The PNG warning is accepted because PNG is the requested preview format alongside vector PDF/SVG. The width warning is a static-parser limitation: it interprets `160/25.4` as 160 inches; the final PDFs independently measure 160 mm wide.

No uncertainty intervals are added: each supplied configuration represents one run. All panels use the same zero-based 0–1.95 y-axis and unchanged validation-error definitions. This revision introduces no statistical comparisons or significance claims.

| Figure/panel | Sensitivity represented | Source records | Bottom caption | Alignment | Collision | Visual inspection |
|---|---|---|---|---|---|---|
| Joint a | SERR gate hidden dimension | Five joint configurations | Pass | Pass | Pass | Pass |
| Joint b | BCMS representation dimension | Five joint configurations | Pass | Pass | Pass | Pass |
| Joint c | SERR residual initialization | Five joint configurations | Pass | Pass | Pass | Pass |
| Joint d | BCMS residual initialization | Five joint configurations | Pass | Pass | Pass | Pass |
| Flow a | SERR gate hidden dimension, inflow | Five inflow configurations | Pass | Pass | Pass | Pass |
| Flow b | SERR gate hidden dimension, outflow | Five outflow configurations | Pass | Pass | Pass | Pass |
| Flow c | BCMS representation dimension, inflow | Five inflow configurations | Pass | Pass | Pass | Pass |
| Flow d | BCMS representation dimension, outflow | Five outflow configurations | Pass | Pass | Pass | Pass |
| Flow e | SERR residual initialization, inflow | Five inflow configurations | Pass | Pass | Pass | Pass |
| Flow f | SERR residual initialization, outflow | Five outflow configurations | Pass | Pass | Pass | Pass |
| Flow g | BCMS residual initialization, inflow | Five inflow configurations | Pass | Pass | Pass | Pass |
| Flow h | BCMS residual initialization, outflow | Five outflow configurations | Pass | Pass | Pass | Pass |

## Reproduction

From the paper package root, with Python, matplotlib and numpy:

    python source/hyperparameters/figure-code/plot_hyperparameters.py --csv source/hyperparameters/original/chengdu_hyperparameter_values.csv --out figures --audit-out source/hyperparameters/figure-audit

Keep the accompanying `audit_panel_alignment.py` beside the plotting script. After rendering, run the nature-figure collision audit on each final PDF and write the reports into `source/hyperparameters/figure-audit`. Then run `qa_hyperparameters.py --skill-scripts <nature-figure-scripts-directory>` with PyMuPDF available. It refreshes the source preflight, PDF glyph audits, numeric verification, previews and final manifest. QA overlay PDFs and preview images are diagnostics and do not replace the scientific figures.
