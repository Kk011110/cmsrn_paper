# 情景分析来源

- originals：用户上传的原始图件和说明，未修改。
- data：从本次用户补充消息直接提取的CSV及阈值来源JSON。
- scripts/prepare_groups.py：用户提供的分组与统计源码，静态核查，不在本机执行训练/推理。
- author-supplied-evidence.txt：作者本次补充的完整原文。
- 数值核查和版面处理记录由本轮审计另行保存。

正式比较与情景图模型均为 end_to_end_full_batch64_v2__seed42；成都checkpoint SHA256为5d6efd31d5935e793c3f30b190e36d3e6d9bbd30f600ebc4d88afbfde01827b8，上海为396fa24fd02055eba3b3ed38e10398a7d028f179768bd7cb4668a108bda78276。5.3的joint_full_v3消融对照是另一权重版本，保持原有记录。

## 复用图件

正文使用figures/scenarios中带_paper后缀的PDF；同名SVG和PNG供编辑交换与预览。编译LaTeX不需要运行绘图或分组脚本。layout_scenario_figures.py根据CSV重排五图，使用Python、NumPy、Matplotlib和Times New Roman；其对齐核验调用脚本中列出的本机nature-figure检查组件。分组源码保留作者原始项目依赖，不能将此资料包当作完整训练工程。

numeric-audit.md记录数值核验；figure-qa保存对齐、字体与碰撞检查。字体按原图保留Times New Roman，源码检查中Nature通用模板的无衬线要求不用于替换本稿既定字体。
