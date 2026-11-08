# RBSRN English Paper — Slim Package

Title: RBSRN: Role-Aware and Background-Conditioned Residual Refinement for Multi-Step Regional Traffic Flow Forecasting

## Files

- `main.tex`: main LaTeX entry point.
- `main.pdf`: compiled manuscript supplied with the original package.
- `preamble.tex`, `frontmatter-metadata.tex`, `references.tex`: shared settings, metadata, and bibliography.
- `sections/`: all 13 active manuscript sections.
- `tables/`: all 9 active table files.
- `figures/`: all 11 active graphics, including those in `figures/scenarios/`.
- `build.cmd`: Windows build script for the clean manuscript.

## Build

Use XeLaTeX with a TeX Live installation containing ctexart, the Fandol fonts, and TeX Gyre Termes. On Windows, add the TeX Live executable directory to PATH and run `build.cmd` in this folder. Alternatively, run the following twice from this folder:

```
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

For an online LaTeX editor, select `main.tex` as the main document and XeLaTeX as the compiler. The project retains the original single-column working-manuscript layout.

## 精简范围

本包基于 RBSRN-LaTeX-English-Units-and-Wording-Revised.zip 整理。正文、公式、参考文献及所用图表文件均逐字节保留，未改动内容；成品 PDF 沿用原包版本。

仅保留当前清稿的编译依赖，移出历史审校目录、实验与绘图来源归档、未调用的备用图表、标记稿和编译辅助文件，并合并重复的外层目录。原始完整压缩包未修改，实验来源和可编辑绘图材料仍可在原包中查找。

本包共40个文件，可单独编译，不依赖原包中的 source/、protocol/ 或审校目录。
