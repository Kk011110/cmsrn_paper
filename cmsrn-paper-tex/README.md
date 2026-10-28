# 在 TeXstudio 中继续编辑论文

本工程将《论文（转tex版）.docx》转换为 UTF-8 编码的中文 LaTeX 文稿。正文和公式均可编辑，使用 XeLaTeX 编译。

当前版本基于用户上传的 `CMSRN-LaTeX-metrics-and-placement-updated.zip` 更新5.3消融实验：正文采用已确认的模块贡献论述；表6集中呈现联合MAE、联合RMSE及相对变化；补充材料S4（表S3）保留四组完整分通道结果。A1、A2、A3均为独立重训练变体。主文25页，补充材料4页，已重新编译并检查图表位置。详见 `ablation-joint-revision-notes-20261028.md`。此前的数值更新和图表章节边界约束继续保留，历史说明文件作为存档。

第五章仍为5.1—5.4，情景分析保留在5.2.3—5.2.5，第6章保持连续讨论、不设子节。目录中的早期修改说明作为历史记录保留，其旧页码与版式描述不代表当前版本。

## 开始使用

1. 将整个 `cmsrn-paper-tex` 文件夹放到希望长期保存论文的位置。解压 ZIP 后再打开文件。
2. 在 TeXstudio 中打开 `main.tex`。不要只复制这一个文件，章节和参考文献文件需要一并保留。
3. 选择 **XeLaTeX** 编译。`main.tex` 首行已写入编译器提示 `% !TeX program = xelatex`。
4. 连续编译两次，使参考文献编号和交叉引用更新。也可以在工程目录运行 `build.cmd`。
5. 查看生成的 `main.pdf`。补充材料请打开 `supplementary-scenarios.tex`，同样使用XeLaTeX连续编译两次，输出 `supplementary-scenarios.pdf`。首次解压时附带的 PDF 是交付时已经编译的预览；修改正文后请重新编译。

本次已使用本机 `D:\texlive\2026\bin\windows\xelatex.exe` 验证。若 TeXstudio 找不到编译器，在其命令设置中将 XeLaTeX 可执行文件指向该路径。工程使用 TeX Live 的 Fandol 中文字体与 TeX Gyre Termes 英文字体，无需依赖宋体、黑体等 Windows 字体。

## 修改哪一个文件

| 文件 | 用途 |
|---|---|
| `main.tex` | 主文件、章节顺序；含暂未启用的标题、作者、摘要填写位置 |
| `preamble.tex` | 页面尺寸、字体、行距、标题与引文格式 |
| `sections/01-introduction.tex` | 引言与两条贡献 |
| `sections/02-related-work.tex` | 相关工作及三个小节 |
| `sections/03-problem-definition.tex` | 问题定义 |
| `sections/04-methodology.tex` | 方法学，已完善为 4.1–4.5 |
| `sections/05-experiments.tex` | 第五章入口，依次组织5.1—5.4 |
| `sections/05-main-results.tex` | 5.2主要结果入口，含总体/空间结果及情景分析加载位置 |
| `sections/05-ablation.tex` | 5.3消融实验正文 |
| `tables/serr-expert-ablation.tex` | 表6：可编辑专家消融结果 |
| `sections/05-hyperparameters.tex` | 5.4超参数敏感性分析 |
| `sections/05-scenarios.tex` | 5.2.3—5.2.5分组性能与残差作用分析，由05-main-results.tex加载 |
| `supplementary-scenarios.tex` | 独立补充材料主文件，用XeLaTeX编译两次 |
| `scenario-revision-notes.md` | 本次选图、比较范围及证据归档说明 |
| `source/scenarios/` | 原始图件、三份CSV、分组源码、阈值/来源与核查记录 |
| `sections/06-discussion.tex` | 第6章讨论，五个连续自然段，不设下级标题 |
| `results-integration-revision-notes.md` | 本次章节整合、讨论重写及编译核验说明 |
| `discussion-revision-notes.md` | PRISM写作方法借鉴及讨论章节的证据说明 |
| `section-5-1-2-revision-notes.md` | 基线对应、官方文献核验与指标口径说明 |
| `section-5-1-1-revision-notes.md` | 写作依据、作者确认的协议及核对记录 |
| `figures/chengdu-t5-overview.png` | 成都T5原始数据集可视化，图3 |
| `figures/shanghai-t5-overview.png` | 上海T5原始数据集可视化，图4 |
| `references.tex` | 24条参考文献；新增情景基线STNorm文献 |
| `figures/introduction-case.pdf` | 引言图1的矢量图文件，编译时需要保留 |
| `conversion-notes.md` | 转换记录与原稿待核对项 |
| `source/original-paper.docx` | Word 原稿的未修改副本 |
| `source/introduction-case/model-palette-case-figure.pptx` | 当前图1的可编辑PPTX源文件 |
| `figures/introduction-case.png` | 当前图1的600dpi预览 |
| `source/original-case-figure.pptx` | 历史案例图源文件，保留用于追溯 |
| `source/original-assets/` | 从 Word 中原样提取的公式图片、WMF 和 OLE 对象，供追溯 |

各章节首行均设置了 `% !TeX root = ../main.tex`，便于编辑子文件时仍编译整篇论文。

## 修改引言案例图

图1的正文引出段、图名和图注位于 `sections/01-introduction.tex`，插入点为原引言第二段之后、两个挑战之前。排版采用顶端浮动，当前图像及图注出现在第2页顶部。

修改图中曲线、标注或布局时，可编辑 `source/introduction-case/model-palette-case-figure.pptx`，再将整页导出为PDF并替换 `figures/introduction-case.pdf`。图文引用使用 `图~\ref{fig:intro-case}`，编号自动更新。

## 继续写作

方法学4.1–4.5、实验章节5.1–5.4及第6章讨论已完成。可在 `sections/05-experiments.tex` 中继续加入后续实验内容；该文件已在主文件加载。新建结论文件后，在 `main.tex` 的 `\input{references}` 之前添加相应的 `\input{sections/文件名}`。

公式已经加入 `\label{eq:...}`，正文可用 `式~\eqref{eq:overall-prediction}` 引用。独立公式使用 `equation` 环境自动编号；行内公式使用 `\(...\)`。

参考文献采用 LaTeX 原生 `thebibliography` / `\bibitem`，直接编辑 `references.tex` 即可，无需运行 BibTeX 或 Biber。正文中 `\cite{ref01}` 指向第 1 条文献；多篇文献写作 `\cite{ref01,ref02,ref03}`。新增文献时使用唯一键，并在正文通过 `\cite{键名}` 引用。

正式标题、作者、摘要、关键词和结论仍待填写。实验章节目前已包含实验设置、主要结果、专家消融、超参数敏感性分析和常规模式偏离情景分析。当前采用通用 A4 单栏中文论文排版；如后续有学校或期刊模板，可在保留正文文件的基础上迁移。


以下为各次交付的历史修订记录，历史章节编号、图号和页码保留用于追溯；当前结构以本文开头及最新版PDF为准。

## 问题定义修订 2026-10-26

已依据最新 `new_full_architecture.py` 将问题定义改为流入与流出联合预测，补全双通道维度、固定常规基线条件与历史时间范围，并同步方法学中的相关维度和双专家描述。该历史修订版的预览为8页；当前版本页数见本文件开头。完整修改依据与检查结果见 [问题定义修订记录](problem-definition-revision-notes.md)。本次修订记录更新了上文历史记录中有关单通道和四专家的对应关系说明。


## 方法学与架构图修订 2026-10-26

本版已依据最新架构补齐 SERR、BCMS 和联合预测输出，并将模型图作为图2放在4.1总体框架之后、4.2之前。

- `sections/04-methodology.tex`：方法学正文、可编辑公式、图2引用及中文图注。
- `figures/model-architecture.pdf`：正式插入论文的矢量模型图。
- `figures/model-architecture.svg`、`.png`：矢量交换文件与预览。
- `source/architecture/user-model-architecture.vsdx`：本轮用户上传的原始 Visio 图，逐字节保留。
- `source/architecture/export_user_figure.ps1`：使用已安装的 Visio 只读导出该图的 PDF/SVG。
- `source/architecture/new_full_architecture.py`：本次核对依据的架构快照。
- `methodology-revision-notes.md`：修订范围、公式与代码对应及训练目标的证据说明。

当前图2以用户提供的 `2026-10-11-模型图.vsdx` 为基础，保留原有结构与连线，去除导出外框并完成字体和换行适配。当前可编辑文件为 `source/architecture/formatted-model-architecture.vsdx`；用户原图另存为 `user-model-architecture.vsdx`。正式导出使用 `export_formatted_figure.ps1`，正文、公式和图注均保持定稿。

工程增加 `placeins` 的浮动边界控制，以保持模型图出现在4.2之前；其余字体、页边距和编译方式沿用原设置。使用 XeLaTeX 连续编译两次，或运行 `build.cmd`。


## 用户模型图替换 2026-10-26

按用户要求以其原始 Visio 图替换上一版重绘图，保留图2标签及其位于4.1之后、4.2之前的位置。中文图注已同步为原图的(a)(b)(c)结构，并将基础预测符号对应关系改为原图使用的 YBase。详细记录见 `figure-replacement-notes.md`。


## 模型图字体与边框适配 2026-10-26

本版已去除模型图最外层黑框，并按论文实际显示宽度统一字号层级、调整密集标签及换行。正式图片已同步更新为PDF/SVG/PNG，包内保留原图与适配后的可编辑Visio。具体修改及再导出方式见 `figure-formatting-notes.md`。

## 图(b)左侧边界修复 2026-10-26

已扩展SERR分区左边界，为左侧文字保留内边距。当前PDF/SVG/PNG和可编辑Visio均已更新，文字字号与正文公式不变。详见 `panel-b-bounds-fix-notes.md`。


## 5.1.1 数据集、预处理和预测协议 2026-10-26

本节已根据本轮上传的模型、两张数据图、对应预处理记录及作者补充完成。正文明确说明窗口级时间顺序划分、相邻边界11个目标时间点重叠、上海采集异常区间的不同处理口径，以及正目标值掩码。数据表列出两城市的属性和训练/验证/测试窗口数。

详见 [5.1.1修订说明](section-5-1-1-revision-notes.md)。图3和图4分别位于当前PDF的第11、12页。作者补充和来源清单位于 `source/section511/`。


## 引言案例图替换 2026-10-26

图1采用本轮上传的模型配色版PDF，直接替换正式图文件，配套PPTX和600dpi PNG均已收入工程。原有图注、编号和引出文字与新图内容对应，继续沿用。编译预览为14页，图1仍位于第2页。详见 `introduction-figure-replacement-notes.md`。


## 5.1.2 基线和评估指标 2026-10-26

已完成13种基线的分类与引用，补充MAE、RMSE和AVG12定义。各单步指标按流量通道，在全部测试窗口及节点的正目标位置上合并计算，排除批次补齐样本；AVG12按12个逐步指标取算术平均。HyperD、RAST使用作者指定的官方名称，不附加标记。

本节位于当前PDF第13页，新增参考文献位于第15页。完整对应关系见 [5.1.2修订说明](section-5-1-2-revision-notes.md)，后续可继续添加5.1.3实现细节。


## 5.2 主要实验结果 2026-10-27

已完成总体排名、AVG12误差、预测时距变化和空间预测案例分析，加入四张可编辑对比表（表2—5）及双城市H6矢量热力图（图5）。正文位于 `sections/05-main-results.tex`，由实验章节自动引入。详细修改及图表口径见 [5.2修订说明](section-5-2-revision-notes.md)。原始结果素材及完整精度核对记录保存在 `source/results/`。此前日志的页码属于历史版本，以本版PDF为准。


## 5.3 超参数敏感性分析 2026-10-27

新增成都数据集的四组单变量验证实验分析，正文位于 `sections/05-hyperparameters.tex`，由实验章节自动接入。图6为联合验证指标，图7为流入与流出分解；两图已按论文宽度重排为可编辑PDF/SVG，并提供600dpi PNG和重绘脚本。执行与指标口径按作者本轮补充记录，具体说明见 [超参数章节修订说明](hyperparameter-revision-notes.md)。

源数据、原图、作者补充和审计记录位于 `source/hyperparameters/`。如需重绘，在工程根目录运行：

```text
python source/hyperparameters/figure-code/plot_hyperparameters.py --csv source/hyperparameters/original/chengdu_hyperparameter_values.csv --out regenerated-hyperparameters
```

重绘需要Python、matplotlib、numpy；正常编译论文直接使用现成PDF即可。


## 5.3消融实验与章节顺延 2026-10-27

在主要结果之后加入成都T5的两类SERR专家消融，原超参数分析顺延为5.4。新增表6保留作者提供的全部数值，正文说明独立重训练、剩余专家权重归一化、BCMS保留以及验证选模与测试AVG12的不同聚合方式。表注注明本轮消融Full与5.2对比Full对应不同运行和检查点；消融变化始终以本轮锁定的Full为参照。

正文及表6位于本版PDF第18页，5.4与图6、图7位于第19—21页，整篇共24页。上述“5.3超参数”标题为历史加入记录，本版章节编号以PDF为准。修改说明见 [消融章节修订记录](ablation-revision-notes.md)，结果原表、作者确认和数值核对位于 `source/ablation/`。


## 第6章讨论 2026-10-27

新增第6章“讨论”，位于本版PDF第22页、参考文献之前；整篇25页。正文分为6.1角色化证据与条件化消息选择、6.2预测表现的差异与启示、6.3局限性与后续研究。写作借鉴PRISM深入分析与结尾部分的论证方式，内容以CMSRN代码、作者创新点材料及现有实验为依据。未新增实验结果或引用条目。

编辑入口为 `sections/06-discussion.tex`；详细修改说明见 [讨论章节修订说明](discussion-revision-notes.md)。来源记录在 `source/discussion/`。此前修订记录中的页数属于历史版本，以本版PDF及本文件开头为准。
