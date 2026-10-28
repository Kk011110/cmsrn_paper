# Scenario figure layout review — 2026-10-28

All 20 panel captions are centered immediately below their own panel content. The caption top is 4 pt below the rendered outer panel content, including the two-line SERR tick labels. Caption placement is measured and asserted during export; plot-area alignment uses a 1.5 pt tolerance. Colors, model order, axes, numeric formatting, observations, and sample counts are retained.

| Figure | Previous size (mm) | Final size (mm) | Height reduction | Minimum PDF glyph | Alignment | Collision audit |
|---|---:|---:|---:|---:|---|---|
| performance_h6_paper | 160 × 173 | 160 × 146 | 15.6% | 6.7 pt | PASS | PASS: 0 fail, 0 warn |
| performance_h6_absolute_mae_paper | 160 × 173 | 160 × 146 | 15.6% | 7.0 pt | PASS | PASS: 0 fail, 0 warn |
| serr_sensitivity_h6_paper | 160 × 95 | 160 × 88 | 7.4% | 7.0 pt | PASS | PASS: 0 fail, 0 warn |
| bcms_sensitivity_horizons_paper | 160 × 161 | 160 × 128 | 20.5% | 6.8 pt | PASS | PASS: 0 fail, 0 warn |
| bcms_sensitivity_horizons_all_test_paper | 160 × 161 | 160 × 128 | 20.5% | 6.8 pt | PASS | PASS: 0 fail, 0 warn |

Each of the five full figures and every a–d panel was visually inspected. Heatmap values intentionally sit within their cells; SERR value labels remain above the corresponding bars. Panel captions, shared legends, colorbars, sample counts, and axis labels remain legible and unclipped. The compact figures retain every panel and data row. The figures represent supplied locked-checkpoint statistics, so no seed uncertainty is invented.

Preservation verification compares all SVG text against the preceding scenario package: all 1,269 text elements are retained exactly, including 20 captions, all values, all sample counts, ticks, and legends. The three CSV source files are byte-identical by SHA-256. Details are in `preservation-check.json`.

The plotting script resolves the bundled alignment helper using a package-relative path. PDF/SVG/PNG names are unchanged. Vector PDF is the manuscript asset; PNG is a 300 dpi preview.

Source preflight is not a clean pass: it reports the inherited Times New Roman choice as a FAIL against its generic sans-serif preference. The manuscript's established serif typography is intentionally preserved. Its three warnings concern PNG instead of TIFF, 300 dpi preview instead of 600 dpi, and the manuscript-specific 160 mm width. These are documented output-contract differences, not rendered clipping or numerical failures. The independent final-PDF glyph, alignment, and collision gates all pass.
