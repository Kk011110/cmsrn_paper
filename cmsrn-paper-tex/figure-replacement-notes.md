# 用户模型图替换记录

日期：2026-10-26。

## 本轮修改

- 以此前交付的 `CMSRN-LaTeX-methodology-revised.zip` 为底稿。
- 图2改为用户上传的 `2026-10-11-模型图.vsdx`，使用 Visio 只读导出 PDF/SVG，并由导出 PDF 生成 300 dpi PNG。没有调整图中的文字、形状、连线、配色或布局。
- 原始 Visio 文件逐字节复制为 `source/architecture/user-model-architecture.vsdx`。上一版重绘图及对应生成文件不再随当前工具包提供。
- 仅修改方法学中的图2图注，逐项对应用户图的 CMSRN、SERR、BCMS 三分图，并将基础预测符号映射改为图中的 YBase。
- 图2引用标签、插入位置和浮动边界保持不变，仍在4.1总体框架之后、4.2之前。
- README及历史修订记录同步注明本轮图源。历史修订说明保留，当前模型图以本说明为准。

## 内容与排版验证

- 方法学文件扣除图2图注后，与上一版逐字节一致；所有其他 LaTeX 源文件逐字节一致。论文正文及公式未作修改。
- 用户原始 VSDX 与包内副本 SHA-256 一致，原文件在导出过程中未改变。
- XeLaTeX 编译通过，最终 PDF 为11页，图2在第7页。页面减少来自原图纵横比改变后的重新排版，没有删减正文。
- 35个引用标签唯一且可解析，无未定义引用、缺字或超宽内容。仅保留原相关工作中的一项 Underfull hbox 提示。
- 已目视核对导出图、最终图2页面与全篇排版，图注没有遮挡或裁切。

## 来源标识

用户模型图 SHA-256：`68bcae7a483a57e65aafe34b51a37671b2c4c30db245545d87347e4cead006fd`。

## 后续使用

直接解压工具包，用 XeLaTeX 编译 `main.tex`。如需再次导出用户模型图，在已安装 Microsoft Visio 的 Windows 环境运行 `source/architecture/export_user_figure.ps1`，随后重新编译论文。图注位于 `sections/04-methodology.tex`。

## 后续字体适配

本文件记录上一轮原图替换。本轮在保留其模型结构的基础上去除导出外框并调整字体与换行；当前图源及导出步骤以 `figure-formatting-notes.md` 为准。用户上传的原始VSDX仍保持不变。
