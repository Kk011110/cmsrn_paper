# 消融实验作者确认记录

日期：2026-10-27。来源：本会话中作者对实验协议问题的直接回复。以下Linux运行路径、检查点散列前缀和实现说明由作者核验并提供；本地未访问对应远程文件。

## 评估与结果版本

- 本次三组结果均来自成都T5测试集，保留 `target > 0` 掩码，排除补齐重复样本。
- 每个预测步跨全部真实测试窗口和节点累计有效位置误差，计算单步MAE，再对12步等权平均得到AVG12。
- 联合MAE为流入、流出AVG12 MAE的算术平均。不是测试batch等权平均，也不是将12步有效位置直接合并。
- 三组共用1,723个真实测试窗口。
- 消融Full运行：`ablation_experiments/joint_full_v3/results/full_seed42`；测试文件 `test_metrics.json`；检查点SHA256前缀 `b42bbc1802f2`；最佳epoch118。
- 第5.2节对比实验Full运行：`baselines/Full/outputs/chengdu/end_to_end_full_batch64_v2__seed42`；指标文件 `metrics.json`；检查点SHA256前缀 `5d6efd31d593`；最佳epoch118。
- 两个Full均为seed42，采用相同测试聚合方式，运行和权重不同。原消融Full为0.708421/0.708654/0.708537，后续对比Full为0.711292/0.710679/0.710986（流入/流出/联合）。训练轨迹差异的具体原因尚未逐项确定，不作额外推断。
- 本次消融变化必须以原始joint_full_v3对照为分母，不能换用第5.2节的Full。
- 当前超参数敏感性图使用验证集，保留补齐，按batch等权平均；本次不修改其聚合口径。困难情景图的指定步长结果也不等同于AVG12。

## 功能移除与训练

- softmax之前将被屏蔽专家的logit设为负无穷，屏蔽权重为0，剩余专家权重归一化为1。
- 完整模型：`B + alpha * [w_s * (S - B) + w_d * D] + R_BCMS`。
- 去自身专家：`B + alpha * D + R_BCMS`。
- 去时滞专家：`B + alpha * (S - B) + R_BCMS`。
- `R_BCMS`包含BCMS自身可学习强度。
- 先构造完整模型，保证与Full相同初始化顺序；施加消融前的参数一致。随后冻结被屏蔽专家和退化为单专家选择的门控。
- 被屏蔽分支仍执行前向并保留在state_dict中，但预测贡献为零、参数不更新。因此不从本实验推断物理删除模块后的参数量或部署加速。
- 两消融组均从随机初始化独立重训练，不加载Full训练权重微调。
- 各组联合训练流入与流出，每组一套权重。BCMS、共享主干、剩余专家和alpha均参与训练。
- 数据划分、归一化、历史证据、延迟图和BCMS候选图与本轮Full对齐。
- seed42，每组一次；120epochs全部完成；batch64；Ranger，学习率0.001。
- 按最低联合验证MAE选检查点，相同时保留更早epoch；测试不参与选模。
- 选模验证指标：每batch内分别计算两通道正目标MAE，通道等权平均，再对补齐后的验证batch等权平均。
- 最佳epoch：Full118，去自身120，去时滞120。

作者提供的实现依据：

- `/home/changan/wangkai/VsCodeProjects/traffic_volume_model_experiment/ablation_experiments/joint_expert_ablation_v3/scripts/run.py:231`：最终测试统计。
- `/home/changan/wangkai/VsCodeProjects/traffic_volume_model_experiment/ablation_experiments/joint_expert_ablation_v3/scripts/model.py:12`：功能移除。

## 数字展示

正文和表格沿用作者提供的全部六位小数MAE与四位小数变化率。作者尚未提供未舍入原始结果文件；核对使用展示值，详见 `numeric-audit.json`。去时滞相对变化按展示值计算为约0.044458%，与原表0.0444%的末位不同，保留原表，正文采用约0.04%；未声称已从原始精度复算。三行通道展示值均值与联合展示值存在0.0000005的差值，同样不以展示值重写作者联合结果。

## 工具包基线

作者随后重新提供 `C:/Users/13594/Desktop/CMSRN-LaTeX-hyperparameter-analysis.zip`。本版以该新上传ZIP为唯一工程基线，SHA256为 `bcf8773e84eef3334204ba0cb7edac6c7eec0e90b3dfc9810e1ecfcf5fe0f733`，保留其中已修改的超参数正文。
