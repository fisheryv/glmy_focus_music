# Path Homology modulation_smp_k10：Focus–Classical 调制视角完整分析

生成日期：2026-08-06。本文使用当前规范数据集Jamendo Open Focus 300首与Classical 300首。两组均分为discovery 195、validation 60、holdout 45；每首有180 s与300 s两个片段，共1,200个片段。状态数固定为$K=10$。主推断为validation/180 s（n=120，每组60），validation/300 s仅作时长敏感性。该模型在既有holdout打开后提出，因此本报告是探索性验证，不把旧holdout倒写为当前模型的确认结果。统计阈值统一为BH-FDR $q\le0.05$。

## 1. 结论摘要

- 仅用discovery/180 s的Classical与Focus各14,715个有效SMP窗口拟合共享变换与固定$K=10$码本；完成1,200/1,200个片段的状态转换及1,200个$K=10$有向图与持续Path Homology，失败0。码本SHA-256为5a7611641f6196213208a89cf3977f1ce9509015c48c40a21502473de13457f2。
- 20个预设指标中，validation/180 s有5个通过BH-FDR，validation/300 s有11个通过；其中4个在两种时长均显著且方向一致。
- 稳定差异为：Open Focus观察状态更多、边数更多，但在已观察节点上的边密度更低、互惠性更低。这描述调制谱形路径覆盖与连接组织，不表示音乐质量高低。
- $H_1$主阈值非零率为Classical 2/60、Open Focus 3/60；预设$H_1$指标在180 s有0个、300 s有0个通过FDR。因此当前不支持稳定的组间$H_1$差异。
- 结论属于观察性声学结构比较；不支持疗效、认知提升、生成质量或因果结论。

## 2. 表示与冻结设计

对mel子带能量包络$x_b[n]$，在4 s窗、2 s步长上计算归一化调制频谱：

$$
P_t(f_m)=\sum_b\left|\sum_n w[n]x_b[n+tH]e^{-i2\pi f_mn/f_s}\right|^2,\qquad
\widetilde P_t(f_m)=\frac{P_t(f_m)}{\sum_{0.5\le f\le45}P_t(f)}.
$$

保留0.5–45 Hz的178维相对SMP。先作Hellinger映射与discovery拟合的中位数/IQR标准化：

$$
h_{tj}=\sqrt{\widetilde P_t(f_j)},\qquad
z_{tj}=
\frac{h_{tj}-\operatorname{median}_D(h_{\cdot j})}
{Q_{0.75,D}(h_{\cdot j})-Q_{0.25,D}(h_{\cdot j})}.
$$

共享PCA-32为$y_t=W_{32}(z_t-\mu_D)$，累计解释方差为0.821。固定MiniBatch K-means状态数$K=10$：

$$
s_t=\arg\min_{v\in\{0,\ldots,9\}}\|y_t-\boldsymbol\mu_v\|_2^2.
$$

原型按原始SMP频谱质心从低到高编号。validation/180 s有效窗口比例中位数为Classical 100.0%、Open Focus 100.0%。无效窗口记为缺失，不建立额外状态，也不跨缺失位置连接转移。

状态数在本报告前固定为10；下表仅描述discovery平衡训练样本，不用于事后增减$K$：

| 状态 | SMP频谱质心（Hz） | 训练窗口 | 训练占比 |
|---|---:|---:|---:|
| P00 | 1.654 | 5,056 | 17.180% |
| P01 | 2.124 | 5,980 | 20.319% |
| P02 | 2.182 | 8,150 | 27.693% |
| P03 | 2.408 | 925 | 3.143% |
| P04 | 2.857 | 4,697 | 15.960% |
| P05 | 2.940 | 217 | 0.737% |
| P06 | 3.189 | 2,735 | 9.293% |
| P07 | 3.569 | 1,282 | 4.356% |
| P08 | 3.675 | 84 | 0.285% |
| P09 | 4.520 | 304 | 1.033% |

训练占比范围为0.285%–27.693%，说明码本存在明显占用不均衡；这属于表示局限，不能通过删除低占用状态来重新优化当前结果。

## 3. 有向图与持续 Path Homology

相邻有效状态定义转移计数与条件概率：

$$
C_{uv}=|\{t:s_t=u,s_{t+1}=v\}|,\qquad
p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}}.
$$

每个源状态最多保留top-6非自环边。主阈值固定为$\tau\in\{0.50,0.60,0.70,0.80,0.90,0.95\}$；扩展至0.05的网格只用于敏感性和机制图。过滤图为

$$
G_\tau=(V,\{(u,v):u\ne v,\ p_{uv}\ge\tau\}).
$$

对允许路径空间使用GLMY边界与路径同调：

$$
\partial e_{v_0\ldots v_p}=\sum_i(-1)^i e_{v_0\ldots\widehat{v_i}\ldots v_p},\qquad
\Omega_p=A_p\cap\partial^{-1}(A_{p-1}),
$$

$$
H_p^{path}(G)=
\frac{\ker(\partial_p|_{\Omega_p})}
{\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}})}.
$$

其中$A_p$由图中允许的有向$p$-路径张成，$\beta_p=\dim H_p^{path}$。阈值下降时边逐步加入，得到秩不变量

$$
\rho_p(\tau_i,\tau_j)=
\operatorname{rank}\operatorname{im}
\left[H_p(G_{\tau_i})\to H_p(G_{\tau_j})\right],
\qquad \tau_i\ge\tau_j.
$$

条形码和持续图使用$a=1-\tau$；生产流程只报告$H_0/H_1$，不作$H_2$声明。

## 4. 可视化

示例focus_jamendo_1571054__180s按固定说明性规则选出：优先选择Open Focus validation/180 s中仅含一个有限$H_1$区间的片段，再按lifetime与segment ID确定性排序。该区间在$\tau=0.50$出生、$\tau=0.20$死亡；它不代表组中心，也不参与假设检验。SSM仅作SMP谱形诊断，主图直接由相邻状态转移构造。

![modulation_smp_k10_codebook](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_codebook.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_codebook.svg)

![modulation_smp_k10_ssm](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_ssm.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_ssm.svg)

![modulation_smp_directed_graph](../runs/modulation_smp_k10_path_homology_open/modulation_smp_directed_graph.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_directed_graph.svg)

![modulation_smp_filtration](../runs/modulation_smp_k10_path_homology_open/modulation_smp_filtration.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_filtration.svg)

![modulation_smp_k10_persistence_diagram](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_persistence_diagram.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_persistence_diagram.svg)

![modulation_smp_k10_barcode](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_barcode.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_barcode.svg)

![modulation_smp_k10_group_summary](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_group_summary.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_k10_group_summary.svg)

![modulation_smp_betti_curves](../runs/modulation_smp_k10_path_homology_open/modulation_smp_betti_curves.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_betti_curves.svg)

![modulation_smp_effect_sizes](../runs/modulation_smp_k10_path_homology_open/modulation_smp_effect_sizes.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_effect_sizes.svg)

![modulation_smp_duration_stability](../runs/modulation_smp_k10_path_homology_open/modulation_smp_duration_stability.png)

[SVG](../runs/modulation_smp_k10_path_homology_open/modulation_smp_duration_stability.svg)

## 5. 组间结果

| 指标 | Classical中位数 | Open Focus中位数 | $\epsilon^2$ | 180 s BH-FDR $q$ | 300 s BH-FDR $q$ |
|---|---:|---:|---:|---:|---:|
| edge_density | 0.667 | 0.524 | 0.105 | 0.00509 | 0.0423 |
| h0_censored_count | 1.000 | 1.000 | 0.087 | 0.00533 | 0.159 |
| vertex_count | 5.000 | 6.000 | 0.092 | 0.00533 | 0.0396 |
| reciprocity | 0.873 | 0.776 | 0.078 | 0.0071 | 0.0464 |
| edge_count | 11.500 | 14.000 | 0.051 | 0.0328 | 0.0424 |
| h0_betti_max | 4.000 | 5.000 | 0.035 | 0.0633 | 0.0424 |
| h0_betti_mean | 2.917 | 3.583 | 0.034 | 0.0633 | 0.0424 |
| h0_interval_count | 4.000 | 5.000 | 0.035 | 0.0633 | 0.0424 |
| self_transition_ratio | 0.448 | 0.401 | 0.031 | 0.0699 | 0.0928 |
| h0_betti_auc | 1.300 | 1.625 | 0.028 | 0.0747 | 0.0424 |
| h0_observed_persistence | 1.450 | 1.775 | 0.026 | 0.0809 | 0.0424 |
| directed_recurrence | 0.130 | 0.109 | 0.021 | 0.1 | 0.0424 |
| transition_entropy | 0.843 | 0.860 | 0.010 | 0.216 | 0.0424 |
| path_entropy | 1.087 | 1.119 | 0.000 | 0.643 | 0.962 |
| h1_betti_auc | 0.000 | 0.000 | 0.000 | 0.683 | 1 |
| h1_betti_max | 0.000 | 0.000 | 0.000 | 0.683 | 1 |
| h1_betti_mean | 0.000 | 0.000 | 0.000 | 0.683 | 1 |
| h1_censored_count | 0.000 | 0.000 | 0.000 | 0.683 | 1 |
| h1_interval_count | 0.000 | 0.000 | 0.000 | 0.683 | 1 |
| h1_observed_persistence | 0.000 | 0.000 | 0.000 | 1 | 0.453 |

Open Focus与Classical在主尺度通过独立两两FDR的指标：

| 指标 | rank-biserial（Open Focus−Classical） | bootstrap 95% CI | BH-FDR $q$ |
|---|---:|---:|---:|
| edge_density | -0.386 | [-0.560, -0.191] | 0.00514 |
| h0_censored_count | 0.299 | [0.133, 0.458] | 0.00539 |
| vertex_count | 0.355 | [0.154, 0.534] | 0.00539 |
| reciprocity | -0.336 | [-0.530, -0.144] | 0.00716 |
| edge_count | 0.279 | [0.077, 0.477] | 0.0331 |

### 5.1 解读

1. Open Focus在同一10状态码本中访问更多状态并形成更多绝对边，但边密度和互惠性更低；这意味着覆盖更广而连接更选择性，不等于“拓扑更好”。
2. 只有同时通过validation/180 s FDR，并在300 s同方向且再次显著的指标，才视为跨时长稳定差异；本轮共4项。
3. h0_censored_count在180 s显著但300 s不显著，且两组180 s中位数同为1；它是分布形状差异，不应写成中位数位移。
4. 主尺度$H_1$发生率低且六个$H_1$指标均未通过FDR，不能改写为“普遍存在稳定调制环”。

### 5.2 统计原理

每个指标在validation/180 s做两组Kruskal–Wallis omnibus检验，并以$\epsilon^2=(H-k+1)/(N-k)$报告秩效应量；独立两两表使用Mann–Whitney $U$与rank-biserial。20个指标在预先定义的单一$K=10$ family内分别做Benjamini–Hochberg校正，判定要求$q\le0.05$。300 s重复同一套检验，但只解释为同曲目的时长敏感性。

## 6. holdout兼容性与不可确认边界

- 当前$K=10$共享SMP模型是在旧holdout打开后提出；旧holdout gate锁定的是modulation_tertile三状态分支，不包含modulation_smp_k10模型、码本哈希或指标方向。
- 虽然全部holdout片段已按同一码本转换以保证产物完整，本报告不对其做组间检验，也不引用旧modulation holdout数值作为$K=10$确认。
- 当前拓扑总表SHA-256为aca9512abaa272007529691df135507275f5d8674c2efe77f31811cd34e44f12。它证明本次结果可审计，不证明与旧门控兼容。
- 若要获得确认性证据，应冻结当前共享变换、$K=10$码本、top-6、阈值、20指标family与哈希，并在未参与本次设计的新数据上验证。

## 7. 证据层级与局限

- **探索性主分析：** validation/180 s、固定$K=10$、top-6、主阈值0.50–0.95、20指标omnibus及pairwise FDR，统一要求$q\le0.05$。
- **时长敏感性：** validation/300 s；只报告跨时长显著性和方向，不称为独立复制。
- **说明性：** discovery码本占用、扩展至0.05的过滤、SMP SSM与单曲birth/death图。
- **不具备：** 当前模型的冻结holdout确认或外部独立复制。
- **不支持：** 稳定组间$H_1$差异、$H_2$发现、注意力/治疗/认知/生成质量或因果结论。
- **表示局限：** PCA-32只保留约82.1%的稳健标准化方差，码本占用不均衡；每片段约88或148个SMP窗口，低占用原型及$H_1$均可能零膨胀。
- Path Homology只使用状态ID、边方向和转移概率；原型频谱质心用于解释，不直接进入链复形。

## 8. 复现入口与产物

PowerShell：

    $env:PYTHONPATH = "packages/pathhom_tda/src;src"
    .\.venv\Scripts\python.exe scripts\run_modulation_smp_prototype_analysis.py
    .\.venv\Scripts\python.exe scripts\render_modulation_smp_k10_report.py

主要数值文件为metadata/modulation_smp_prototype_features.csv、metadata/modulation_smp_prototype_topology_segments.csv、metadata/modulation_smp_prototype_topology_filtration.csv、metadata/modulation_smp_prototype_topology_filtration_sensitivity.csv、metadata/modulation_smp_prototype_statistical_tests.csv、metadata/modulation_smp_prototype_pairwise_tests.csv与metadata/modulation_smp_prototype_summary.json。$K=10$模型为features/models/modulation_smp_proto_k10.npz/json；图和哈希清单位于runs/modulation_smp_k10_path_homology_open。
