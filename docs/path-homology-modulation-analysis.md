# 调制视角 Path Homology：Focus–Classical 完整重跑报告

生成日期：2026-08-04。切分版本：`symmetric_holdout_v2`。数据为 Jamendo Open Focus 300 首与 Classical 300 首；每组 discovery/validation/holdout 分别为 195/60/45 首。分析共重算 1,200/1,200 个 180 s/300 s 片段视图，覆盖 600 首曲目，失败 0。本版将确认性端点统一为 BH-FDR $q\le0.05$；原 holdout 的 $q\le0.10$ 门控作为历史记录保留。

> 证据边界：validation/180 s 是本报告的主单视角检验；validation/300 s 是同曲目的时长敏感性。holdout 是冻结哈希门控后的单次操作性最终确认。由于 Classical holdout 在旧切分中曾属于 discovery，它不是 pristine 外部复制集。

## 1. 结论摘要

- 冻结的三状态调制表示检测到稳定的组间组织差异：20 个预设指标中，validation/180 s 有 **12 个**通过 BH-FDR $q\le0.05$；300 s 有 **13 个**通过。共有 **12 个**在两个时长均显著，其中 **10 个**具有一致的非零中位数方向，另有 **2 个**两组中位数均相等、显著性来自分布而非中位数位移。
- Open Focus 在冻结量化下呈现更高的自转移率与有向复现度，以及更低的转移熵、路径熵、边数和多项 $H_0$ 汇总量。这表示三状态转移更集中，不表示音乐质量、注意力效果或因果机制。
- **$H_1$ 明确不支持。** 主阈值下 Classical 为 0/60、Focus 为 0/60；阈值扩展至 0.05 后仍分别为 0/60 与 0/60。holdout/180 s 也为 Classical 0/45、Focus 0/45。
- 冻结 holdout 中，调制块整体表示 pseudo-$F=2.597$，$p=0.005$，跨次级块 BH $q=0.005$。原门控的 10 个调制方向指标中，10/10 方向一致；历史 $q\le0.10$ 与统一严格 $q\le0.05$ 均为 10/10 复现。
- 当前重跑的拓扑输入 SHA-256 为 `b99a97d769a5999fef8a87b4dcce8e7b64e88c5fe76fae08e52f7e4e90e9f186`，与 holdout gate **一致**；模型 SHA-256 为 `0b4705b1bffe9c58774570835bfeefca496e4f17309cb3a416bbafcd2d86787c`。

## 2. 方法思想：把调制动力学变成有向路径

该方法不直接对音频波形做同调，而是先把短时调制能量压缩成 Low/Medium/High 状态序列，再把相邻状态的条件转移概率构造成有向图。普通图拓扑只关心“是否连通”，Path Homology 还保留路径的方向和可连接次序，因此适合描述“调制状态如何演化”。过滤阈值从高到低加入边，观察有向连通分量和有向环是否持续存在。

这一路径回答的是：**调制强度级别之间的转移组织是否不同**。它不回答三个频带各自的独立作用，也不等价于完整声学表征。原有 27 个组合状态的 `modulation` 分支保持独立；本报告只分析新重跑的 `modulation_tertile` 三状态分支。

## 3. Spectral Modulation Profile 与冻结三分位表示

对 mel 子带能量包络 $x_b[n]$，在第 $t$ 个 4 s 窗内计算调制频率谱：

$$
P_t(f_m)=\sum_b\left|\sum_n w[n]x_b[n+tH]e^{-i2\pi f_m n/f_s}\right|^2,
\qquad
\widetilde P_t(f_m)=\frac{P_t(f_m)}{\sum_{0.5\le f\le45}P_t(f)}.
$$

步长 $H=2$ s，只保留 0.5–45 Hz，并将谱归一化。三个预先指定频带的相对能量为

$$
E_{B,t}=\sum_{f_m\in B}\widetilde P_t(f_m),
\qquad
m_t=E_{8:12,t}+E_{18:20,t}+E_{28:32,t}.
$$

因此 $m_t$ 是“重点频带相对调制能量占比”，不是绝对调制功率。仅在 discovery/180 s 中从 Classical 与 Focus 各平衡抽取 14,715 个有效窗口，拟合冻结边界 $q_1=0.024829621$、$q_2=0.044771713$：

$$
s_t=\begin{cases}
0\ (\mathrm{Low}),&m_t<q_1,\\
1\ (\mathrm{Medium}),&q_1\le m_t<q_2,\\
2\ (\mathrm{High}),&m_t\ge q_2.
\end{cases}
$$

validation/180 s 的实际状态占用为：

| 组别 | Low | Medium | High |
|---|---:|---:|---:|
| Classical | 41.5% | 43.0% | 15.5% |
| Open Focus | 17.2% | 23.2% | 59.6% |

无效窗记作缺失状态 $-1$，缺失区间两侧不跨越连接。validation、300 s 与 holdout 均不参与边界拟合。

## 4. 有向转移图、Path Homology 与持久性

相邻有效状态定义转移计数与条件概率：

$$
C_{uv}=\left|\{t:s_t=u,\ s_{t+1}=v\}\right|,
\qquad
p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}}.
$$

自转移只用于描述统计，不进入 Path Homology 图。每个源状态保留 top-6 非自环边；三状态下每源最多只有两条，所以该规则不会额外截边。超水平过滤为

$$
G_\tau=\left(V,\{(u,v):u\ne v,\ p_{uv}\ge\tau\}\right).
$$

主阈值冻结为 $\tau\in\{0.50,0.60,0.70,0.80,0.90,0.95\}$；0.05–0.40 只用于敏感性和机制图。令 $A_p$ 为允许的有向 $p$-路径张成空间，路径边界为

$$
\partial e_{v_0\ldots v_p}=\sum_{i=0}^p(-1)^i
e_{v_0\ldots\widehat{v_i}\ldots v_p}.
$$

并非每条允许路径的边界仍然允许，因此使用 $\partial$-不变路径空间

$$
\Omega_p=A_p\cap\partial^{-1}(A_{p-1}),
\qquad
H_p^{\mathrm{path}}(G)=
\frac{\ker(\partial_p:\Omega_p\to\Omega_{p-1})}
{\operatorname{im}(\partial_{p+1}:\Omega_{p+1}\to\Omega_p)}.
$$

$\beta_p=\dim H_p^{\mathrm{path}}$。用 $a=1-\tau$ 把降阈值过滤改写为递增参数；持久秩不变量为

$$
\rho_p(a_i,a_j)=\operatorname{rank}\operatorname{im}
\left[H_p(G_{a_i})\longrightarrow H_p(G_{a_j})\right],
\qquad a_i\le a_j.
$$

据此得到 barcode、persistence diagram、Betti 曲线、区间数、观测持久量与 AUC。生产流程只报告 $H_0/H_1$；本轮没有计算 $H_2$，因此不能作 $H_2$ 发现声明。

## 5. 统计检验与冻结确认

主检验对 20 个预设指标分别做两组 Kruskal–Wallis 检验，并在单一 modulation_tertile family 内作 BH-FDR，确认性判定统一要求 $q\le0.05$。若秩和统计量为 $H$、组数为 $k$、总样本为 $N$，效应量为

$$
\epsilon^2=\frac{H-k+1}{N-k}.
$$

两组情况下另报告 Mann–Whitney rank-biserial，方向统一为 Focus $-$ Classical。300 s 是同曲目时长敏感性，不是独立复制。holdout 的整体表示使用发现集拟合的秩正态 Mahalanobis 距离和 999 次标签置换；pseudo-$F$ 可写为

$$
F^*=\frac{SS_{between}/(g-1)}{SS_{within}/(N-g)}.
$$

冻结 holdout 只验证 gate 中按原 $q\le0.10$ 方案预先锁定的 10 个调制指标；它不重新选择当前表中的 12 个 validation 发现，也不重开盲或调参。完整 holdout 的 44 个四视角方向指标中，43/44 方向一致，历史 $q\le0.10$ 为 42/44、严格 $q\le0.05$ 为 39/44 联合 FDR 复现；本报告只把其中调制视角的 10/10 作为严格口径证据。

## 6. 完整数值结果

### 6.1 Omnibus 检验

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s FDR | 300 s FDR |
|---|---:|---:|---:|---:|---:|
| path_entropy | 0.855 | 0.546 | 0.266 | 2.55e-07 | 7.29e-08 |
| directed_recurrence | 0.216 | 0.430 | 0.253 | 2.8e-07 | 1.37e-07 |
| self_transition_ratio | 0.540 | 0.706 | 0.227 | 8.77e-07 | 2.07e-07 |
| edge_count | 6.000 | 4.500 | 0.200 | 2.03e-06 | 5.75e-08 |
| h0_betti_max | 3.000 | 1.500 | 0.200 | 2.03e-06 | 1.37e-07 |
| h0_interval_count | 3.000 | 1.500 | 0.200 | 2.03e-06 | 1.37e-07 |
| transition_entropy | 0.818 | 0.642 | 0.201 | 2.03e-06 | 5.1e-06 |
| h0_betti_mean | 1.833 | 1.083 | 0.135 | 9.59e-05 | 0.000163 |
| h0_betti_auc | 0.775 | 0.462 | 0.130 | 9.63e-05 | 0.000712 |
| h0_observed_persistence | 0.850 | 0.475 | 0.131 | 9.63e-05 | 0.00019 |
| vertex_count | 3.000 | 3.000 | 0.132 | 9.63e-05 | 0.00111 |
| edge_density | 1.000 | 1.000 | 0.032 | 0.047 | 1.68e-05 |
| reciprocity | 1.000 | 1.000 | 0.012 | 0.191 | 0.00125 |
| h0_censored_count | 1.000 | 1.000 | 0.000 | 0.453 | 0.801 |
| h1_betti_auc | 0.000 | 0.000 | 0.000 | 1 | 1 |
| h1_betti_max | 0.000 | 0.000 | 0.000 | 1 | 1 |
| h1_betti_mean | 0.000 | 0.000 | 0.000 | 1 | 1 |
| h1_censored_count | 0.000 | 0.000 | 0.000 | 1 | 1 |
| h1_interval_count | 0.000 | 0.000 | 0.000 | 1 | 1 |
| h1_observed_persistence | 0.000 | 0.000 | 0.000 | 1 | 1 |

### 6.2 两组方向效应

| 指标 | rank-biserial（Open Focus−Classical） | FDR |
|---|---:|---:|
| path_entropy | -0.602 | 2.59e-07 |
| directed_recurrence | 0.588 | 2.85e-07 |
| self_transition_ratio | 0.558 | 8.89e-07 |
| edge_count | -0.477 | 2.06e-06 |
| h0_betti_max | -0.471 | 2.06e-06 |
| h0_interval_count | -0.471 | 2.06e-06 |
| transition_entropy | -0.526 | 2.06e-06 |
| h0_betti_mean | -0.428 | 9.7e-05 |
| h0_betti_auc | -0.421 | 9.74e-05 |
| h0_observed_persistence | -0.423 | 9.74e-05 |
| vertex_count | -0.267 | 9.74e-05 |
| edge_density | -0.191 | 0.0473 |
| reciprocity | -0.117 | 0.192 |
| h0_censored_count | -0.017 | 0.465 |
| h1_betti_auc | -0.000 | 1 |
| h1_betti_max | -0.000 | 1 |
| h1_betti_mean | -0.000 | 1 |
| h1_censored_count | -0.000 | 1 |
| h1_interval_count | -0.000 | 1 |
| h1_observed_persistence | -0.000 | 1 |

### 6.3 结果解释

1. **路径集中度是最稳定的差异。** Focus 的 directed recurrence 与 self-transition ratio 更高，path entropy 和 transition entropy 更低，说明状态路径更集中于少数状态/转移。
2. **$H_0$ 描述高阈值连通过程。** Classical 保留更多高阈值边，并有更高的 $\beta_0$ 最大值、均值、AUC 与观测持久量；这表示条件转移概率更分散，图要到更低阈值才合并，不表示“拓扑更好”。
3. **普通图指标显著不等于 $H_1$ 显著。** 六个 $H_1$ 指标均恒为零、FDR 为 1。证据来自状态占用、边组织、路径熵和 $H_0$，不是环。
4. **三状态压缩限制环表示能力。** 它可解释、稳定，但远低于 27 个组合状态的表达容量；不能据此否定更细状态空间中可能存在的环，只能说当前冻结表示没有观察到。

## 7. 可视化

示例 `focus_jamendo_1858750__180s` 按预设回退规则选自 Focus validation/180 s：由于没有有限 $H_1$ 区间，选择边数最多、随后路径熵最高的片段。该示例不参与检验，也不代表组中心。

![modulation_tertile_diagnostics](../runs/modulation_tertile_path_homology_open/modulation_tertile_diagnostics.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_tertile_diagnostics.svg)

![modulation_smp_profile](../runs/modulation_tertile_path_homology_open/modulation_smp_profile.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_smp_profile.svg)

![modulation_directed_state_graph](../runs/modulation_tertile_path_homology_open/modulation_directed_state_graph.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_directed_state_graph.svg)

![modulation_filtration_process](../runs/modulation_tertile_path_homology_open/modulation_filtration_process.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_filtration_process.svg)

![modulation_persistence_diagram](../runs/modulation_tertile_path_homology_open/modulation_persistence_diagram.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_persistence_diagram.svg)

![modulation_barcode](../runs/modulation_tertile_path_homology_open/modulation_barcode.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_barcode.svg)

![modulation_group_summary](../runs/modulation_tertile_path_homology_open/modulation_group_summary.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_group_summary.svg)

![modulation_betti_curves](../runs/modulation_tertile_path_homology_open/modulation_betti_curves.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_betti_curves.svg)

![modulation_scale_sensitivity](../runs/modulation_tertile_path_homology_open/modulation_scale_sensitivity.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_scale_sensitivity.svg)

![modulation_effect_sizes](../runs/modulation_tertile_path_homology_open/modulation_effect_sizes.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_effect_sizes.svg)

![modulation_duration_stability](../runs/modulation_tertile_path_homology_open/modulation_duration_stability.png)

[下载 SVG](../runs/modulation_tertile_path_homology_open/modulation_duration_stability.svg)

## 8. 证据层级与局限

- **确认性：** validation/180 s、冻结三分位、top-6、主阈值 0.50–0.95、20 指标 family 与 BH-FDR，统一要求 $q\le0.05$。
- **指纹入选：** $q\le0.05$ 是必要条件；还必须满足预定方向、300 s 方向一致性、去冗余和融合增量审计。$0.05<q\le0.10$ 的端点只作提示性结果，不进入核心冻结指纹。
- **操作性最终确认：** 哈希门控的 holdout/180 s；但 Classical holdout 不是 pristine 外部样本，且其配器构成没有 piano solo，泛化解释需保守。
- **敏感性：** validation/300 s 和阈值下探至 0.05；不能替代主检验。
- **探索/说明性：** discovery 分布、单片段 SMP 热图与机制图。
- **不支持：** 当前三状态表示下的 $H_1$；任何 $H_2$ 发现；把相对 SMP 占比解释为绝对功率；注意力、治疗、认知、生成质量或因果结论。
- 三个重点频带求和会丢失频带间方向；归一化比例也会受谱内其他频带能量变化影响。4 s/2 s 网格只刻画局部调制轨迹，三分位边界是当前 discovery 数据的经验量化器，不是普适生理阈值。
- 两组来自不同曲库，录音、母带、配器、作曲家和元数据选择差异仍可能混入观察结果。

## 9. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
.\.venv\Scripts\python.exe scripts\run_modulation_tertile_analysis.py
.\.venv\Scripts\python.exe scripts\render_modulation_tertile_report.py
```

主要数值产物：`metadata/modulation_tertile_features.csv`、`metadata/modulation_tertile_topology_segments.csv`、`metadata/modulation_tertile_topology_filtration.csv`、`metadata/modulation_tertile_topology_filtration_sensitivity.csv`、`metadata/modulation_tertile_statistical_tests.csv`、`metadata/modulation_tertile_pairwise_tests.csv` 与 `metadata/modulation_tertile_summary.json`。图和持久结果位于 `runs/modulation_tertile_path_homology_open/`、`graphs/modulation_tertile/` 与 `homology/*/modulation_tertile/`。
