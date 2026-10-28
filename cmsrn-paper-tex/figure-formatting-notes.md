# 模型图边框与字体适配记录

日期：2026-10-26。

## 修改内容

- 去除 Visio PDF 导出产生的整页黑色矩形描边，仅去掉该外框，保留彩色虚线分区和模型内部边框。PDF 保持矢量格式。
- 以 Times New Roman 为主，分图标题加粗。按论文160 mm显示宽度反算字号：主要模块文字约8 pt，密集标签适当降至约7.3–7.7 pt，辅助符号约6.3–7 pt；数学上下标保留原生缩排。
- 调整文字框宽度、内边距和换行，避免单词断裂及文字与连线重叠。左上方 Frozen directed lag graph 改为箭头上方的横排三行标注。
- 将图中文字的 Casual Convolution / Casual Conv、LayNorm、Score Compution 分别订正为 Causal Convolution / Causal Conv、LayerNorm、Score Computation，并统一分图编号为(a)、(b)、(c)。未改变模型结构。
- 节点、连线、分区、配色保留用户图。原始VSDX逐字节保留，调整版另存为 formatted-model-architecture.vsdx。

## 文件

- `figures/model-architecture.pdf`：论文当前使用的无外框矢量图。
- `figures/model-architecture.svg`、`figures/model-architecture.png`：矢量交换文件与300 dpi预览。
- `source/architecture/user-model-architecture.vsdx`：用户上传原图。
- `source/architecture/formatted-model-architecture.vsdx`：当前字号与换行适配后的可编辑图。
- `source/architecture/export_formatted_figure.ps1`：从调整版Visio导出PDF/SVG，再移除PDF导出外框并生成PNG。
- `source/architecture/finish_figure_export.py`：精确移除PDF全页描边，不删除正常剪裁路径或模块边框。

## 验证

- 所有LaTeX文件与上一版逐字节一致，正文、公式及图注均未改动。
- 原图与适配图的形状编号集合一致，非文字图形的坐标、尺寸及连线端点保持一致。
- 导出PDF中不存在整页矩形描边；PDF保留可选中的矢量文字。
- XeLaTeX编译通过，全文11页，图2仍在第7页、4.1之后和4.2之前。无未定义引用、缺字或超宽排版。
- 已检查最终图和论文内的实际尺寸，并独立复核标签遮挡；未发现必须修复的文字重叠或裁切。

## 再次导出

在安装Microsoft Visio的Windows环境中编辑 formatted-model-architecture.vsdx，再运行 export_formatted_figure.ps1。该脚本默认调用python，可用-PythonExecutable指定解释器；Python需安装pypdf、pypdfium2及Pillow。旧的export_user_figure.ps1仅用于归档原图，不用于本版正式出图。
