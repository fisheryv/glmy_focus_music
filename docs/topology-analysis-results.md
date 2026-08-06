# Path Homology 拓扑建模与统计分析结果

生成日期：2026-08-02。主验证集为 validation / 180 秒；discovery 仅用于探索和分类器拟合，300 秒为敏感性分析，holdout 未参与本轮统计。

## 数据与方法

共建模 1,200 个片段、4,800 个片段-视图。对 modulation、pitch、rhythm、structure 共 4 类冻结状态序列构建 top-k 出向概率有向图；在 0.50–0.95 的固定阈值上计算实系数 GLMY H0/H1，并以链空间包含映射计算有限过滤的持久秩不变量和条形码。

单变量检验采用 Kruskal–Wallis，分析集内 Benjamini–Hochberg FDR q=0.10；多变量检验采用 Mahalanobis PERMANOVA（999 次置换）。分类基线仅在 discovery 拟合与选参，在 validation 报告。

## 主结果（validation / 180 秒）

多变量 PERMANOVA：pseudo-F=4.180，p=0.001，n=139。

300 秒敏感性分析的 PERMANOVA 为 pseudo-F=3.571，p=0.001，n=139。

80 个预设视图-指标检验中，FDR q≤0.10 的结果有 47 个。其中 46/47 个在 validation / 300 秒仍通过 FDR 且方向一致。效应最大的显著结果如下：

| 视图 | 指标 | Classical 中位数 | Focus 中位数 | ε² | FDR p |
|---|---|---:|---:|---:|---:|
| pitch | vertex_count | 13.000 | 8.000 | 0.693 | 5.69e-21 |
| pitch | path_entropy | 1.792 | 1.143 | 0.690 | 5.69e-21 |
| pitch | edge_count | 63.000 | 32.500 | 0.676 | 1.02e-20 |
| pitch | h0_betti_mean | 11.667 | 6.500 | 0.669 | 1.02e-20 |
| pitch | h0_betti_auc | 5.300 | 2.975 | 0.669 | 1.02e-20 |
| pitch | h0_observed_persistence | 5.400 | 3.175 | 0.660 | 1.56e-20 |
| pitch | h0_betti_max | 12.000 | 8.000 | 0.652 | 2.05e-20 |
| pitch | h0_interval_count | 12.000 | 8.000 | 0.652 | 2.05e-20 |
| pitch | h0_censored_count | 9.000 | 3.000 | 0.613 | 2.66e-19 |
| pitch | directed_recurrence | 0.028 | 0.102 | 0.606 | 4.00e-19 |
| pitch | self_transition_ratio | 0.306 | 0.569 | 0.567 | 5.29e-18 |
| pitch | transition_entropy | 0.895 | 0.773 | 0.427 | 8.41e-14 |

### H1 专项结果

24 个 H1 视图-指标检验中有 0 个通过 FDR。本轮差异主要由 H0 连通结构和图转移描述子驱动；在当前 ≥0.50 的稀疏过滤下，多数曲目的 H1 为 0，因此不能将多变量显著性解释为有向一维洞的组间差异。

## 原 H2 专项假设的适用性

原 H2 专项检验预先固定为 Focus vs Pop。由于新的规范数据集已移除 Pop，本轮不执行该专项检验，也不将比较组事后改为 Classical；其状态记为**不适用（comparator absent）**。通用 H0/H1 两组统计仍按冻结设置执行。

## 分类基线

| 特征 | Macro-F1 | 95% CI | 平衡准确率 | Macro-AUROC |
|---|---:|---:|---:|---:|
| all | 0.970 | [0.940, 0.993] | 0.967 | 1.000 |
| acoustic_modulation | 0.948 | [0.910, 0.978] | 0.944 | 0.995 |
| acoustic | 0.941 | [0.902, 0.978] | 0.935 | 0.995 |
| topology | 0.917 | [0.869, 0.955] | 0.908 | 0.989 |
| modulation | 0.868 | [0.808, 0.920] | 0.868 | 0.904 |

## 解释边界

这些结果描述 Focus 与 Classical 两组音频状态转移拓扑的分布差异，不构成注意力提升或因果效果证据。300 秒与扩展过滤结果用于敏感性分析。holdout 仅含 Focus 曲目、没有对照组，因此不用于本轮组间假设检验。

## 图形输出

- `runs/topology_statistics/topology_pca_validation_180.png`
- `runs/topology_statistics/h1_filtration_validation_180.png`
- `runs/topology_statistics/classification_macro_f1_validation_180.png`
