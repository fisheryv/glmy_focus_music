# Path Homology `pitch_v2`：Focus–Classical 音高视角完整分析

生成日期：2026-08-06。本文使用当前规范数据集 Jamendo Open Focus 300 与 Classical 300。两组均分为 discovery 195、validation 60、holdout 45；每首有 180 s 与 300 s 两个片段，共 1,200 个片段。主推断固定为 validation/180 s（n=120：每组 60）；validation/300 s 仅作时长敏感性。holdout 是既有哈希门控后的单次操作性确认，不用于重新选择参数或指标。本版将确认性端点统一为 BH-FDR $q\le0.05$；原 holdout 的 $q\le0.10$ 门控作为历史记录保留，不倒写为新预注册标准。

## 1. 结论摘要

- 仅用 discovery/180 s 的 Focus 与 Classical 各 50,000 个有效节拍重新拟合 Tonnetz 码本，并完成 1,200/1,200 个片段的有向图和持续 Path Homology，失败 0。码本 SHA-256：`172e7094b7a170b50058cc2cfb72f89054f951dc2bf8506ae0e3381d3977a7c8`。
- 20 个预设指标中，validation/180 s 有 13 个通过 BH-FDR $q\le0.05$；validation/300 s 有 18 个通过，其中 13 个在两种时长均显著且方向一致。原 $q\le0.10$ 下额外入选的 `edge_density`（$q=0.0654$）现降为提示性结果。
- 既有单次 holdout/180 s 确认中，pitch 表示的 permutation pseudo-$F=7.402$、$p=0.001$、次级家族 FDR $q=0.00167$；原门控的 14 个方向性指标中 14 个方向一致、按历史 $q\le0.10$ 有 14 个复现，按统一严格口径 $q\le0.05$ 有 14 个复现。重跑后的拓扑清单哈希与开盲门控记录一致。
- $H_1$ 主阈值非零为 Classical 2/60、Open Focus 6/60。预设 $H_1$ 指标在 180 s 有 0 个、300 s 有 5 个通过 FDR；必须与零膨胀和效应量一起解释。
- 结论属于观察性声学结构比较；不支持疗效、认知提升、生成质量或任何因果结论。

## 2. 表示与冻结设计

对每个节拍的 12 维 chroma $\mathbf c_b$ 作 $L_1$ 归一化，并映射到 Harte Tonnetz 的五度、小三度和大三度三个圆周：

$$
\widetilde{\mathbf c}_b=\frac{\mathbf c_b}{\sum_p c_b(p)+\varepsilon},\qquad
\mathbf z_b=\Phi\widetilde{\mathbf c}_b\in\mathbb R^6.
$$

令 $q=(7/6,3/2,2/3)$、$r=(1,1,1/2)$，则固定基矩阵可写为

$$
\Phi_{2k,p}=r_k\sin(\pi q_kp),\qquad
\Phi_{2k+1,p}=r_k\cos(\pi q_kp),
\quad k=0,1,2,\ p=0,\ldots,11.
$$

这一步把八度等价的音级质量映到五度/三度圆周；随后直接在固定 Tonnetz 欧氏度量中聚类，不对 validation 或 holdout 重新标准化或拟合。

仅在 discovery/180 s 上按组等量抽样，每组 50,000 个有效节拍，拟合固定 $V_{pitch}=16$ 的 MiniBatch K-means；validation 与 holdout 均不参与码本拟合：

$$
s_b=\arg\min_{v\in\{0,\ldots,15\}}\|\mathbf z_b-\boldsymbol\mu_v\|_2^2.
$$

低置信节拍沿用既有 1.15 主峰比规则并记为缺失，不建立第 17 个状态，也不跨缺失位置连接转移。validation/180 s 有效节拍比例中位数为 Classical 81.0%、Open Focus 81.0%。

码本状态数在分析前固定为 16；下表是 discovery-only 诊断，不用于事后更换主结果：

| $V_{pitch}$ | Silhouette | 种子稳定 ARI | 最小簇占比 | 最大簇占比 | 每步 inertia |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.193 | 0.370 | 0.063 | 0.221 | 0.0863 |
| 12 | 0.216 | 0.542 | 0.047 | 0.114 | 0.0672 |
| 16 | 0.222 | 0.528 | 0.042 | 0.128 | 0.0552 |
| 24 | 0.195 | 0.460 | 0.022 | 0.065 | 0.0453 |

## 3. 有向图与持续 Path Homology

相邻有效状态定义转移计数与条件概率：

$$
C_{uv}=|\{b:s_b=u,s_{b+1}=v\}|,\qquad
p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}}.
$$

每个源状态最多保留 top-6 非自环边。主阈值冻结为 $\tau\in\{0.50,0.60,0.70,0.80,0.90,0.95\}$；扩展至 0.05 的网格仅用于敏感性和机制图。过滤图为

$$
G_\tau=(V,\{(u,v):u\ne v,\ p_{uv}\ge\tau\}).
$$

对允许路径空间 $\Omega_p$，使用 GLMY 边界与路径同调：

$$
\partial e_{v_0\ldots v_p}=\sum_i(-1)^i e_{v_0\ldots\widehat{v_i}\ldots v_p},\qquad
\Omega_p=A_p\cap\partial^{-1}(A_{p-1}),\qquad
H_p^{path}(G)=\frac{\ker(\partial_p|_{\Omega_p})}{\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}})}.
$$

其中 $A_p$ 由图中允许的有向 $p$-路径张成，$\beta_p=\dim H_p^{path}$。阈值下降时边逐步加入，得到包含映射及秩不变量

$$
\rho_p(\tau_i,\tau_j)=\operatorname{rank}\operatorname{im}\left[H_p(G_{\tau_i})\to H_p(G_{\tau_j})\right],\qquad \tau_i\ge\tau_j.
$$

条形码和持续图使用单调坐标 $a=1-\tau$；主报告给出 $H_0/H_1$ 的 Betti 曲线、区间数量、观测持续性与右删失数量。

## 4. 可视化

示例 `focus_jamendo_1700327__180s` 按冻结的说明性规则选出：优先选择当前 Open Focus validation/180 s 中恰有一个有限 $H_1$ 区间的片段，再按区间 lifetime 最大及 segment ID 确定性排序。其敏感阈值区间在 $\tau=0.5$ 出生、$\tau=0.1$ 死亡；它不代表组中心，也不参与假设检验。SSM 仅作 Tonnetz 诊断，主图直接由相邻状态转移构造。

![pitch_v2_codebook](../runs/pitch_v2_path_homology_open/pitch_v2_codebook.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_codebook.svg)

![pitch_v2_tonnetz_ssm](../runs/pitch_v2_path_homology_open/pitch_v2_tonnetz_ssm.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_tonnetz_ssm.svg)

![pitch_v2_directed_state_graph](../runs/pitch_v2_path_homology_open/pitch_v2_directed_state_graph.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_directed_state_graph.svg)

![pitch_v2_filtration_process](../runs/pitch_v2_path_homology_open/pitch_v2_filtration_process.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_filtration_process.svg)

![pitch_v2_persistence_diagram](../runs/pitch_v2_path_homology_open/pitch_v2_persistence_diagram.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_persistence_diagram.svg)

![pitch_v2_barcode](../runs/pitch_v2_path_homology_open/pitch_v2_barcode.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_barcode.svg)

![pitch_v2_group_summary](../runs/pitch_v2_path_homology_open/pitch_v2_group_summary.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_group_summary.svg)

![pitch_v2_betti_curves](../runs/pitch_v2_path_homology_open/pitch_v2_betti_curves.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_betti_curves.svg)

![pitch_v2_effect_sizes](../runs/pitch_v2_path_homology_open/pitch_v2_effect_sizes.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_effect_sizes.svg)

![pitch_v2_duration_stability](../runs/pitch_v2_path_homology_open/pitch_v2_duration_stability.png)

[SVG](../runs/pitch_v2_path_homology_open/pitch_v2_duration_stability.svg)

## 5. 组间结果

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s BH-FDR $q$ | 300 s BH-FDR $q$ |
|---|---:|---:|---:|---:|---:|
| edge_count | 70.000 | 31.000 | 0.683 | 8.2e-19 | 1.8e-19 |
| h0_betti_auc | 6.200 | 3.125 | 0.677 | 8.2e-19 | 1.8e-19 |
| h0_betti_max | 14.500 | 8.000 | 0.672 | 8.2e-19 | 1.8e-19 |
| h0_betti_mean | 13.667 | 6.833 | 0.676 | 8.2e-19 | 1.8e-19 |
| h0_interval_count | 14.500 | 8.000 | 0.672 | 8.2e-19 | 1.8e-19 |
| h0_observed_persistence | 6.375 | 3.325 | 0.677 | 8.2e-19 | 1.8e-19 |
| path_entropy | 1.696 | 0.965 | 0.673 | 8.2e-19 | 1.8e-19 |
| vertex_count | 16.000 | 10.000 | 0.703 | 8.2e-19 | 2.8e-19 |
| h0_censored_count | 11.000 | 3.500 | 0.596 | 6.53e-17 | 7.59e-18 |
| directed_recurrence | 0.025 | 0.114 | 0.585 | 1.14e-16 | 4.43e-17 |
| self_transition_ratio | 0.322 | 0.631 | 0.563 | 3.89e-16 | 1.92e-16 |
| transition_entropy | 0.915 | 0.771 | 0.489 | 3.03e-14 | 8.01e-14 |
| reciprocity | 0.522 | 0.667 | 0.361 | 6.25e-11 | 2.31e-09 |
| edge_density | 0.311 | 0.332 | 0.025 | 0.0654 | 1 |
| h1_betti_auc | 0.000 | 0.000 | 0.010 | 0.153 | 0.0255 |
| h1_betti_max | 0.000 | 0.000 | 0.010 | 0.153 | 0.0255 |
| h1_betti_mean | 0.000 | 0.000 | 0.010 | 0.153 | 0.0255 |
| h1_censored_count | 0.000 | 0.000 | 0.010 | 0.153 | 0.0255 |
| h1_interval_count | 0.000 | 0.000 | 0.010 | 0.153 | 0.0255 |
| h1_observed_persistence | 0.000 | 0.000 | 0.000 | 1 | 1 |

Open Focus 与 Classical 在主尺度通过独立两两 FDR 的指标：

| 指标 | rank-biserial（Open Focus−Classical） | bootstrap 95% CI | BH-FDR $q$ |
|---|---:|---:|---:|
| edge_count | -0.956 | [-0.989, -0.903] | 8.4e-19 |
| h0_betti_auc | -0.952 | [-0.994, -0.891] | 8.4e-19 |
| h0_betti_max | -0.944 | [-0.986, -0.886] | 8.4e-19 |
| h0_betti_mean | -0.951 | [-0.994, -0.892] | 8.4e-19 |
| h0_interval_count | -0.944 | [-0.987, -0.889] | 8.4e-19 |
| h0_observed_persistence | -0.952 | [-0.993, -0.892] | 8.4e-19 |
| path_entropy | -0.949 | [-0.987, -0.897] | 8.4e-19 |
| vertex_count | -0.954 | [-0.987, -0.909] | 8.4e-19 |
| h0_censored_count | -0.892 | [-0.965, -0.801] | 6.68e-17 |
| directed_recurrence | 0.886 | [0.806, 0.947] | 1.16e-16 |
| self_transition_ratio | 0.869 | [0.762, 0.956] | 3.97e-16 |
| transition_entropy | -0.811 | [-0.906, -0.692] | 3.09e-14 |
| reciprocity | 0.699 | [0.543, 0.834] | 6.37e-11 |

### 5.1 解读

1. 状态数、边数、熵和 $H_0$ 指标描述两组在同一 16 状态码本中的覆盖与连通过程，不等于音乐质量高低。
2. 只有同时通过 validation/180 s FDR、在 300 s 方向一致并再次显著的指标，才视为跨时长稳定差异；本轮共有 13 项。
3. 主尺度的 $H_1$ 非零率分别为 Classical 3.3%、Open Focus 10.0%。即使秩检验显著，也不能在中位数为零或低发生率时改写为“普遍存在稳定音高环”。

### 5.2 统计原理

每个指标在 validation/180 s 上做两组 Kruskal–Wallis omnibus 检验，并以 $\epsilon^2=(H-k+1)/(N-k)$ 报告秩效应量；独立两两表使用 Mann–Whitney $U$ 与 rank-biserial 效应。20 个指标各自在预先定义的家族内做 Benjamini–Hochberg 校正，确认性判定统一要求 $q\le0.05$。300 s 重复同一套检验，但只解释为同曲目的时长敏感性，不称为独立复制。

## 6. 已冻结 holdout 的兼容性核验

- 本次重跑产出的 `pitch_v2_topology_segments.csv` SHA-256 为 `c3926081ca9708bafb940b5d51ccca309199b3253c1f70ab239b826dbe66fb56`；开盲门控记录为 `c3926081ca9708bafb940b5d51ccca309199b3253c1f70ab239b826dbe66fb56`；核验结果：**一致**。
- 因哈希一致，本次统计口径更新没有改变既有单次 holdout 的输入，故可引用原先冻结后的结果：pitch/180 s pseudo-$F=7.402$，$p=0.001$，$q=0.00167$；14/14 方向一致，历史 $q\le0.10$ 与严格 $q\le0.05$ 均为 14/14 复现。
- 该 holdout 是回顾性对称重切分下的操作性最终确认；Classical holdout 在旧切分中曾属于 discovery，因此不是 pristine 外部确认集。不得将它提升为外部独立复制或因果证据。

## 7. 证据层级与局限

- **确认性：** validation/180 s、固定 16 状态、top-6、主阈值 0.50–0.95、20 指标 omnibus FDR 与独立 pairwise FDR，统一要求 $q\le0.05$。
- **指纹入选：** $q\le0.05$ 是必要条件；还必须满足预定方向、300 s 方向一致性、去冗余和融合增量审计。$0.05<q\le0.10$ 的端点只作提示性结果，不进入核心冻结指纹。
- **敏感性：** validation/300 s；报告跨时长显著性和方向一致性，不以敏感性结果替代主检验。
- **探索/说明性：** discovery 诊断、扩展至 0.05 的阈值、码本 $K$ 诊断和单曲 birth/death 图。
- **操作性最终确认：** 既有哈希门控 holdout/180 s；本轮只核验输入哈希仍相同，不重新开盲或调参。
- **不支持：** 将两组声学差异解释为注意力、治疗、认知或生成效果；将旧 Pop 对照结论转移到当前两组数据。
- `pitch_v2` 的顶点具有 Tonnetz 原型语义，但 Path Homology 本身只使用状态 ID、边方向和转移概率，没有直接计算 Tonnetz 群作用。

## 8. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
python scripts/run_pitch_v2_analysis.py
python scripts/render_pitch_v2_report.py
```

主要数值文件为 `metadata/pitch_v2_features.csv`、`metadata/pitch_v2_topology_segments.csv`、`metadata/pitch_v2_topology_filtration.csv`、`metadata/pitch_v2_topology_filtration_sensitivity.csv`、`metadata/pitch_v2_statistical_tests.csv`、`metadata/pitch_v2_pairwise_tests.csv` 和 `metadata/pitch_v2_summary.json`。
