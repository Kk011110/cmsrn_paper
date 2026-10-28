# 图(b)左侧文字越界修复

日期：2026-10-26。

本次仅将SERR浅橙色分区框的左边界向外扩展约1.77 mm，右边界保持不变，以容纳左侧变量与 Frozen directed lag graph 说明。没有移动或缩小文字，没有调整节点和连线。

验证结果：

- 实际PDF字形边界核对通过，最左文字与新边框之间约2.73 pt；左上、左下圆角附近文字也位于边框内。
- Visio中仅分区框Shape 1258发生变化；其余形状完全一致。
- 模型图中全部文字内容及字形位置与上一版一致，字号未缩小。
- 所有LaTeX源文件与上一版逐字节一致，正文、公式、图注未改动。
- PDF/SVG/PNG及可编辑Visio已同步更新，XeLaTeX编译成功；论文仍为11页，图2在第7页。

当前编辑源为 source/architecture/formatted-model-architecture.vsdx，原始用户Visio副本保留。导出方法沿用 source/architecture/export_formatted_figure.ps1。
