# 音高视角 Path Homology：Open Focus 与 Classical 重新分析

生成日期：2026-08-02。分析对象为当前 Open Focus 300 首与 Classical 300 首的两组规范数据。

## 摘要

本次按冻结配置重新计算全部 1,200 个 pitch 片段视图，覆盖 600 首曲目的 180s/300s 版本，成功 1,200、失败 0。validation/180s 的 20 个预设指标中有 14 个通过 BH-FDR q≤0.10；其中 14 个在同曲目的 validation/300s 中方向一致并再次通过 FDR。20 指标联合 Mahalanobis PERMANOVA 得到 pseudo-F=10.693、p=0.001。主要差异来自状态字母表、边数、H0 连通过程、路径熵和转移集中度，而不是稳定 H1。

主阈值 0.50–0.95 下，validation/180s 非零 H1 仅为 Open Focus 1/60、Classical 0/60；扩展阈值至 0.05 后变为 Open Focus 16/60、Classical 25/60。故不能声称存在稳健或 Focus 特异的音高 H1。当前 validation 已在其他项目分析中被查看，本报告是冻结方法在迁移后数据上的观察性重新分析，而非新的 pristine 确认性实验。

## 1. 方法思想

音高视角不直接比较曲调名称或调性标签，而是把每个节拍区间编码成主导音级状态，再研究状态之间的有向转移。强边反映某个音级状态后经常出现的下一状态；随着转移概率阈值降低，弱边逐步进入图，连通分量合并，并可能形成或填充有向一维路径同调类。

```mermaid
flowchart LR
    A["音频"] --> B["谐波 chroma"]
    B --> C["节拍同步池化"]
    C --> D["12 音级 + 不确定态 U"]
    D --> E["相邻状态转移计数"]
    E --> F["按源状态归一化 + top-6"]
    F --> G["超水平过滤 G_tau"]
    G --> H["GLMY H0/H1 与持久区间"]
```

## 2. 音高状态表示

### 2.1 节拍同步 chroma

把 STFT 谐波功率谱按八度折叠到 12 个音级。对第 b 个相邻节拍区间的帧集合 I_b，池化表示为

$$\bar{\mathbf c}_b=\frac{1}{|I_b|}\sum_{t\in I_b}\mathbf c_t,\qquad \bar{\mathbf c}_b\in\mathbb R_{\ge0}^{12}.$$

节拍同步减少逐帧颤音、起音偏移和局部时间伸缩造成的状态抖动，但其质量仍依赖节拍估计。

### 2.2 主导音级与不确定态

令 c_b^(1)、c_b^(2) 为最大和次大 chroma 分量，冻结的不确定比为 1.15：

$$s_b=\begin{cases}\arg\max_{p\in\{0,\ldots,11\}}\bar c_b(p),&c_b^{(1)}/\max(c_b^{(2)},10^{-8})\ge1.15\ \text{且}\ c_b^{(1)}>10^{-8},\\U,&\text{其他情况}.\end{cases}$$

U 编码为 12。虽然特征文件还保存 `valid = states != 12`，当前研究批处理将所有非负整数状态都作为图顶点，因此 U 是第 13 个可观测状态，而不是缺失值。

| 组别 | n | U 比例中位数 | U 比例均值 |
|---|---:|---:|---:|
| Classical | 60 | 0.190 | 0.199 |
| Open Focus | 60 | 0.190 | 0.220 |

### 2.3 音高自相似矩阵

对解释性图形，先令 $\widehat{\mathbf c}_i=\bar{\mathbf c}_i/(\|\bar{\mathbf c}_i\|_2+\varepsilon)$，再计算 $S_{ij}^{\mathrm{pitch}}=\widehat{\mathbf c}_i^{\mathsf T}\widehat{\mathbf c}_j$。远离对角线的亮块表示不同时段具有相近的音级能量配置；SSM 不参与状态图边权或组间检验。

![Pitch chromagram and SSM](../runs/pitch_path_homology_open/pitch_chromagram_ssm.png)

代表片段为 `focus_jamendo_1068406__180s`（Open Focus，validation/180s）；U 比例为 0.089。

## 3. 有向状态图

相邻状态转移计数与按源状态归一化概率为

$$C_{uv}=|\{t:s_t=u,\ s_{t+1}=v\}|,\qquad p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}}.$$

每个源状态最多保留概率最大的 6 条非自环边。自转移不进入 GLMY 正则路径图，但仍以描述量保留：

$$r_{\mathrm{self}}=\frac{\sum_u C_{uu}}{\sum_{u,v}C_{uv}}.$$

路径熵和有向复现度分别为

$$H_{\mathrm{path}}=-\sum_{u,v}\frac{C_{uv}}{N}\log\frac{C_{uv}}{\sum_w C_{uw}},\qquad R_{\mathrm{dir}}=\sum_{u,v}\left(\frac{C_{uv}}N\right)^2.$$

前者衡量条件转移的多样性，后者衡量概率质量是否集中在少数转移。

![Directed pitch graph](../runs/pitch_path_homology_open/pitch_directed_state_graph.png)

## 4. GLMY 路径同调与持续过滤

对阈值 τ 定义超水平过滤

$$G_\tau=(V,E_\tau),\qquad E_\tau=\{(u,v):u\ne v,\ p_{uv}\ge\tau\}.$$

主分析阈值冻结为 {0.50,0.60,0.70,0.80,0.90,0.95}；敏感性阈值扩展到 {0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95}。实现按超水平方向从高阈值向低阈值加入边。

允许的正则 p-路径 e_(v0...vp) 要求相邻顶点均由有向边连接。边界算子为

$$\partial e_{v_0\ldots v_p}=\sum_{i=0}^{p}(-1)^i e_{v_0\ldots\widehat{v_i}\ldots v_p}.$$

删除中间顶点后所得路径可能不再允许，因此链空间限制为

$$\Omega_p=\{a\in A_p:\partial a\in A_{p-1}\}.$$

路径同调群和 Betti 数为

$$H_p^{\mathrm{path}}(G)=\frac{\ker(\partial_p|_{\Omega_p})}{\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}})},\qquad \beta_p=\dim H_p^{\mathrm{path}}(G).$$

β0 描述弱连通分量；β1 描述未被允许 2-路径边界填充的独立有向一维类，它不等于简单环计数。过滤包含映射诱导同调映射，其秩不变量为

$$\rho_p(i,j)=\operatorname{rank}\bigl(H_p(G_i)\to H_p(G_j)\bigr),\qquad i\le j.$$

持久区间由完整秩不变量分解；在观测末端仍存活的类标为右删失。绘图采用递增坐标 a=1−τ。

![Pitch filtration](../runs/pitch_path_homology_open/pitch_filtration_process.png)

![Pitch persistence diagram](../runs/pitch_path_homology_open/pitch_persistence_diagram.png)

![Pitch barcode](../runs/pitch_path_homology_open/pitch_barcode.png)

代表片段的敏感性过滤含 1 个 H1 区间；最长观测跨度为 0.100。该样本用于展示边的加入如何形成或填充路径类，不代表总体典型性。

## 5. 数据与统计协议

- 数据：Open Focus 300、Classical 300；每首各有 180s 与 300s。
- 对称切分：每组 discovery 195、validation 60、holdout 45。
- pitch 状态无需拟合码本；1.15 不确定比、top-6、非自环和过滤阈值均来自冻结配置。
- 主要分析：validation/180s；20 指标 Kruskal–Wallis（两组时与双侧 Mann–Whitney 秩检验等价），在 20 指标家族内 BH-FDR q≤0.10。
- 多变量检查：20 指标 Mahalanobis PERMANOVA，协方差只由 discovery/180s 冻结参考估计，999 次置换。
- 时长敏感性：validation/300s；同曲目而非独立复制。
- holdout：只报告描述统计，不根据其结果调参或再次开启显著性家族。

## 6. validation/180s 主要结果

20 个预设指标中 14 个通过 q≤0.10。正效应表示 Open Focus 倾向更高，负效应表示 Classical 倾向更高。

| 指标 | Classical 180s | Focus 180s | 180s 效应 | 180s FDR | 300s 效应 | 300s FDR |
|---|---:|---:|---:|---:|---:|---:|
| 有向边数 | 64.000 | 32.500 | -0.967 | 7.36e-19 | -0.972 | 1.11e-19 |
| H0 Betti AUC | 5.325 | 2.975 | -0.956 | 7.36e-19 | -0.979 | 1.11e-19 |
| 平均 beta0 | 11.750 | 6.500 | -0.955 | 7.36e-19 | -0.980 | 1.11e-19 |
| 路径熵 | 1.830 | 1.143 | -0.955 | 7.36e-19 | -0.975 | 1.11e-19 |
| 观察状态数 | 13.000 | 8.000 | -0.921 | 7.36e-19 | -0.907 | 4.11e-19 |
| H0 观察持久量 | 5.475 | 3.175 | -0.949 | 9.09e-19 | -0.973 | 1.11e-19 |
| 最大 beta0 | 12.500 | 8.000 | -0.925 | 1.95e-18 | -0.952 | 1.11e-19 |
| H0 右删失数 | 10.000 | 3.000 | -0.936 | 1.95e-18 | -0.947 | 5.92e-19 |
| H0 区间数 | 12.500 | 8.000 | -0.925 | 1.95e-18 | -0.952 | 1.11e-19 |
| 有向复现度 | 0.026 | 0.102 | 0.905 | 2.46e-17 | 0.928 | 3.56e-18 |
| 自转移比 | 0.303 | 0.569 | 0.863 | 6.70e-16 | 0.876 | 2.40e-16 |
| 互惠性 | 0.596 | 0.780 | 0.795 | 9.72e-14 | 0.838 | 4.18e-15 |
| 转移熵 | 0.897 | 0.773 | -0.778 | 3.09e-13 | -0.764 | 7.87e-13 |
| 边密度 | 0.434 | 0.484 | 0.421 | 9.99e-05 | 0.243 | 0.031 |
| H1 Betti AUC | 0.000 | 0.000 | 0.017 | 0.334 | 0.017 | 0.334 |
| 最大 beta1 | 0.000 | 0.000 | 0.017 | 0.334 | 0.017 | 0.334 |
| 平均 beta1 | 0.000 | 0.000 | 0.017 | 0.334 | 0.017 | 0.334 |
| H1 右删失数 | 0.000 | 0.000 | 0.017 | 0.334 | 0.017 | 0.334 |
| H1 区间数 | 0.000 | 0.000 | 0.017 | 0.334 | 0.017 | 0.334 |
| H1 观察持久量 | 0.000 | 0.000 | -0.000 | 1.000 | -0.000 | 1.000 |

![Pitch group summary](../runs/pitch_path_homology_open/pitch_group_summary.png)

![Pitch effect sizes](../runs/pitch_path_homology_open/pitch_effect_sizes.png)

方向最稳定的模式是：Classical 具有更多状态、更多边、更高路径熵和更大的 H0 连通持久过程；Open Focus 具有更高自转移、互惠性、边密度和有向复现度。这是音高状态组织差异，不是价值排序。H0 较高也不能单独解释为“更复杂”，因为它强烈受状态字母表大小影响。

## 7. 多变量结果

| 角色 | 时长 | n | pseudo-F | 置换 p | 有效维数 |
|---|---:|---:|---:|---:|---:|
| primary_validation_180 | 180.000 | 120 | 10.693 | 0.001 | 13 |
| duration_sensitivity_300 | 300.000 | 120 | 19.828 | 0.001 | 13 |

PERMANOVA 支持两组在联合 pitch 拓扑描述空间中可分，但不能判断哪一单项指标是原因，也不能排除配器、古典子类型、制作方式和状态数等混杂。

## 8. H0/H1 过滤行为

![Pitch Betti curves](../runs/pitch_path_homology_open/pitch_betti_curves.png)

主阈值下 H1 几乎完全零膨胀（Focus 1/60；Classical 0/60）。扩展到低概率边后，H1 出现率反而是 Classical 25/60、Focus 16/60。这个方向变化和阈值依赖说明：H1 适合做单曲解释与敏感性诊断，不适合作为稳健组间核心结论。

## 9. 180s/300s 稳定性

| 组别 | 显著指标 ρ 中位数 | 最小 ρ | 最大 ρ |
|---|---:|---:|---:|
| Classical | 0.794 | 0.463 | 0.895 |
| Open Focus | 0.920 | 0.823 | 0.984 |

主要 14 个指标中，14 个在 300s 中同方向且再次通过 FDR。由于 300s 包含同曲目的前 180s，它只是时长敏感性，不是独立样本复制。逐指标相关见 `metadata/pitch_scale_stability.csv`。

## 10. holdout 描述

| 指标 | Classical 中位数 | Focus 中位数 |
|---|---:|---:|
| 观察状态数 | 13.000 | 8.000 |
| 有向边数 | 70.000 | 32.000 |
| 自转移比 | 0.314 | 0.549 |
| 路径熵 | 1.851 | 1.175 |
| 有向复现度 | 0.024 | 0.088 |
| H0 观察持久量 | 5.850 | 3.050 |
| 最大 beta1 | 0.000 | 0.000 |

holdout 未用于本报告的新显著性检验。Classical holdout 不含钢琴独奏，且其中曲目在旧切分中曾属于 discovery，不能称为 pristine 外部确认集。

## 11. 结论与证据边界

### 支持

- 当前 Open Focus 与 Classical 在节拍级主导音高状态的有向组织上存在强而稳定的观察性差异。
- 差异主要由状态覆盖、转移网络规模、路径熵、转移集中度以及 H0 连通过程贡献。
- validation/300s 保持绝大多数主要指标的方向和 FDR 结果。

### 不支持

- 不支持稳健、普遍或 Focus 特异的 pitch H1；主阈值 H1 极稀疏，低阈值结果明显依赖弱边。
- 不支持由 H0 较低或转移更集中直接推出‘更有序’、‘更优’或‘更适合专注’。
- 不支持注意力提升、治疗作用、生成质量或其他因果结论。

### 局限

- Chroma 折叠八度并混合旋律、和声、伴奏与泛音，不能替代音符级转录。
- 节拍跟踪误差会改变池化边界与状态序列长度。
- U 同时反映复音、低能量和主峰不明确，不是和弦或休止标签；将 U 改作缺失值会构成另一套分析。
- H0 与可观察状态数高度耦合，必须与状态数、边数和归一化指标联合解释。
- 两组在来源、配器和体裁上不同，组间差异不等于 Focus 功能机制。

## 12. 可复现产物

- `scripts/rerun_pitch_path_homology.py`
- `scripts/render_pitch_path_report_current.py`
- `metadata/pitch_topology_segments.csv`
- `metadata/pitch_topology_filtration.csv`
- `metadata/pitch_topology_filtration_sensitivity.csv`
- `metadata/pitch_statistical_tests.csv`
- `metadata/pitch_pairwise_tests.csv`
- `metadata/pitch_permanova.csv`
- `metadata/pitch_scale_stability.csv`
- `metadata/pitch_path_homology_analysis_summary.json`
- `runs/pitch_path_homology_open/`（全部图均有 PNG 与 SVG）

复现命令：

```powershell
$env:PYTHONPATH='packages/pathhom_tda/src;src'
.\.venv\Scripts\python.exe scripts\rerun_pitch_path_homology.py
.\.venv\Scripts\python.exe scripts\render_pitch_path_report_current.py
```

## 参考文献

1. Müller, M. (2015). *Fundamentals of Music Processing*. Springer.
2. Ellis, D. P. W., & Poliner, G. E. (2007). Identifying cover songs with chroma features and dynamic programming beat tracking. *ICASSP*.
3. Grigor'yan, A., Lin, Y., Muranov, Y., & Yau, S.-T. (2012). Homologies of path complexes and digraphs. arXiv:1207.2834.
4. Chowdhury, S., & Mémoli, F. (2018). Persistent path homology of directed networks. *SODA*.
