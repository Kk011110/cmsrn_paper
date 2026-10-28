# 最终图件人工复核

所有面板均为单个锁定检查点的完整分组统计；无多种子置信区间，不绘制虚构误差条。热图单元格文字位于色块内，为有意包含叠加；所有描边碰撞已清除。字体Times New Roman沿用原图。160mm为当前正文宽度，PNG为300dpi预览，实际入稿使用PDF。源码预检保留1项字体规则差异与预览格式/宽度提示，具体结果见source-preflight.stdout.txt。

| 图 | a—d面板 | 数值/单位 | 对齐 | 最小字号 | 碰撞 | 目视 |
|---|---|---|---|---|---|---|
| performance_h6_paper | 成都流入、流出；上海流入、流出 | MAE改善率%，两位小数，16基线全保留 | 通过 | 6.7pt | 0/0 | 四面板及整图通过 |
| performance_h6_absolute_mae_paper | 同上 | 17方法原始流量单位MAE | 通过 | 7.0pt | 0/0 | 四面板及整图通过 |
| serr_sensitivity_h6_paper | 同上 | 原始单位ΔMAE，Slow/Rapid各2角色 | 通过 | 7.0pt | 0/0 | 四面板及整图通过 |
| bcms_sensitivity_horizons_paper | 同上 | 1000×ΔMAE，高偏离子集 | 通过 | 6.8pt | 0/0 | 四面板及整图通过 |
| bcms_sensitivity_horizons_all_test_paper | 同上 | 1000×ΔMAE，完整测试集 | 通过 | 6.8pt | 0/0 | 四面板及整图通过 |

浮点指标计算未重跑训练或推理。色标覆盖全部有限数值；两个BCMS图各自覆盖其全范围，跨图应读取数值。最终主文30页、补充4页；新增图注、单位和正文引用一致。
