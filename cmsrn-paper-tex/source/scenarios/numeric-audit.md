# 困难情景统计核查

总体检查：存在待检查项；100/101项。

核查范围：基于用户补充CSV重新计算汇总恒等式、排名和展示精度，并静态阅读分组脚本；未运行训练/推断脚本，也未以原始预测数组或检查点文件重新推断。

## H6重构与主表

|城市|通道|有效n|重构MAE|重构RMSE|
|---|---|---:|---:|---:|
|chengdu|inflow|499545|0.709952489|1.297768171|
|chengdu|outflow|526904|0.709943702|1.226077209|
|shanghai|inflow|653592|0.726774085|1.225502683|
|shanghai|outflow|639653|0.773783775|1.344938436|

现有主表中的56个方法—城市—通道组合，全部H6 MAE通过三位小数展示精度核对；RMSE仅上海流入Full存在以下舍入边界差异。原表没有VAR、STNorm、STWave，三者仅属新增情景扩展比较。

上海流入Full：原工作簿H6 MAE为0.726768800793293，RMSE为1.22547011631793；新CSV重构为0.726774085249245和1.225502683017405。MAE显示均为0.727，RMSE分别四舍五入为1.225和1.226。差异位于原脚本的rtol=1e-4、atol=1e-5容差内，具体数值入口/精度原因尚未单独核定。本次保留原主表，新增章节只论述CSV支持的分组MAE。

## 极高偏离组

|城市|通道|CMSRN排名|最近/最佳基线|相对最佳基线改善（%）|
|---|---|---:|---|---:|
|chengdu|inflow|1|STWave|0.356555|
|chengdu|outflow|3|STWave|-0.598080|
|shanghai|inflow|1|STWave|0.237088|
|shanghai|outflow|1|STWave|0.235342|

## 残差与分组一致性

- SERR与BCMS的delta_mae均核对为removed_mae−full_mae。
- SERR快/慢组与BCMS背景组在H6的有效n均重构相应总体/高偏离子集；加权完整模型MAE亦一致。
- Self与Delay共16项移除差值均正；快速/平缓自身修正比值为2.818108, 1.613247, 2.124105, 1.819461。
- 高偏离BCMS：143/144为正；完整测试BCMS：144/144为正。两范围的所有城市—通道—预测步均呈低背景差异>中>高；完整测试的144项均大于对应高偏离值。
- 不将单分支MAE移除效应求和视为整个SERR效应；不将冻结输出诊断解释为重新训练消融或结构机制的独立因果贡献。

## SVG显示核对

- bcms_sensitivity_horizons: 通过，SVG 144 个显示数值，CSV 144 个对应数值。
- bcms_sensitivity_horizons_all_test: 通过，SVG 144 个显示数值，CSV 144 个对应数值。
- performance_h6: 通过，SVG 256 个显示数值，CSV 256 个对应数值。
- performance_h6_chengdu: 通过，SVG 128 个显示数值，CSV 128 个对应数值。
- performance_h6_shanghai: 通过，SVG 128 个显示数值，CSV 128 个对应数值。
- serr_sensitivity_h6: 通过，SVG 16 个显示数值，CSV 16 个对应数值。
- performance_h6_absolute_mae: 通过，SVG 272 个显示数值，CSV 272 个对应数值。

## 分组实现要点

- 常规中位数按节点、通道、日类型、时槽拟合；原组合支持不足4点时合并同槽日类型，合并后为空才回退到节点/通道总体中位数。
- MAD尺度是每节点/通道固定量，来自可用的唯一训练历史索引；不是时槽特定尺度。
- 训练阈值只使用12个历史槽全部有效的训练窗口。上海为4180个，成都为5170个；这不改变原预测实验的训练样本。
- 检查点身份由用户提供的provenance记录；本次没有检查点二进制，未独立验证文件SHA256。

## 待检查项

- paper_table_h6_display_matches: {'comparisons': 56, 'unlisted_methods': ['STNorm', 'STWave', 'VAR'], 'exceptions': [{'city': 'shanghai', 'flow': 'inflow', 'model': 'Full', 'reconstructed': [0.7267740852492449, 1.2255026830174054], 'displayed': [0.727, 1.225], 'passed': False}]}
