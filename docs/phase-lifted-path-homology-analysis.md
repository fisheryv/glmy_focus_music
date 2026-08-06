# 相位提升路径同调：Open Focus 与 Classical 的重新分析

生成日期：2026-08-02。本报告对应 2026-08-02 两组数据迁移后的独立重跑。

## 摘要

本次在当前 600 首曲目的 Open Focus/Classical 数据上重新执行相位提升路径同调。方法参数沿用原冻结配置，不根据本次结果调节：6 个相位节点、4 帧块聚合、至少 96 个原始时间步、至少 3 个周期、候选周期不超过 32 个块、超水平阈值 0.05–0.95。三个预定义视角 Acoustic、Rhythm、Chroma 全部纳入，未再次依据组间 p 值筛选。共分析 1,198 个合格片段、3,594 个片段-视角；质量门槛排除 2 个片段。

在 discovery-only 构造校准中，通过人工循环优于时间打乱门槛的视角为：Rhythm phase、Acoustic phase、Chroma phase。validation/180s 的三视角双侧 Mann–Whitney 检验在三视角检验家族内进行 BH-FDR 后，q≤0.05 的视角为：Acoustic phase、Chroma phase。这是一项迁移后的重新分析，不是对原 Focus>Pop 假设的确认性复制；Classical 比较对象发生改变，而且当前 validation 已在其他分析中被查看，因此结果应解释为当前数据上的观察性组间差异。

## 1. 方法思想

普通状态转移图回答“哪些声学状态彼此转换”；相位提升则回答“一个候选重复周期内部，各相位是否按稳定顺序闭合”。它先从声学轨迹估计主导重复周期，再把周期位置压缩到 6 个相位节点。若所有相邻相位都能跨周期稳定复现，就形成有向环，其一维路径同调非零。最弱的一条相位边决定该环能够承受多高的边权过滤阈值。

```mermaid
flowchart LR
    A["声学时序 x_t"] --> B["4 帧块聚合"]
    B --> C["距离矩阵 D_ij"]
    C --> D["估计主导周期 P*"]
    D --> E["映射为 6 个相位"]
    E --> F["相位一致性 c_k"]
    F --> G["有向环边权 w_k"]
    G --> H["超水平过滤 G_tau"]
    H --> I["Path H1 与 loop score"]
```

## 2. 数学原理与公式

### 2.1 三种输入表示

- Acoustic：对 discovery 拟合的声学标准化与 PCA 表示取前 8 维，再按 4 帧求均值；块间隔为 2 秒。
- Rhythm：对节奏向量按 discovery 模型插补、标准化，再按 4 帧求均值；块间隔为 2 秒。
- Chroma：逐帧 L2 归一化后按 4 帧求均值；距离对 12 种循环移调取最优匹配。

对 Acoustic 与 Rhythm，令 z_i 为块级向量，距离为

$$D_{ij}=\frac{\lVert z_i-z_j\rVert_2}{\sqrt d},\qquad z_{ij}=\frac{x_{ij}-\operatorname{median}_i x_{ij}}{\max(\operatorname{sd}_i x_{ij},10^{-8})}.$$

对 Chroma，先单位化 $u_i=x_i/\lVert x_i\rVert_2$，再定义移调不变距离

$$D_{ij}=\sqrt{\max\left(0,2-2\max_{s\in\{0,\ldots,11\}}u_i^\top R_su_j\right)}.$$

其中 $R_s$ 表示循环移动 $s$ 个半音。

### 2.2 主导周期与相位提升

在候选集合 $\mathcal P=\{K,\ldots,\min(P_{\max},\lfloor N/C\rfloor)\}$ 中选择

$$P^*=\arg\min_{P\in\mathcal P}\operatorname{median}_i D_{i,i+P}.$$

其中相位数 $K=6$、最小周期数 $C=3$、$P_{\max}=32$。以非邻近距离的正值中位数 $s$ 为尺度，跨周期复现强度为

$$r_i=\exp\left(-D_{i,i+P^*}/s\right).$$

将周期位置映射到离散相位

$$q_i=\left\lfloor\frac{(i\bmod P^*)K}{P^*}\right\rfloor,\qquad c_k=\operatorname{mean}\{r_i:q_i=k\}.$$

并构造有向边 $k\to(k+1)\bmod K$，边权为

$$w_k=\min(c_k,c_{k+1}).$$

取相邻相位一致性的较小值，是为了让每条边同时受其起点和终点的稳定性约束。

### 2.3 GLMY 路径同调

在有向图中，允许的 $p$-路径是顶点序列 $e_{i_0\ldots i_p}$，相邻顶点之间均存在有向边。边界算子为

$$\partial e_{i_0\ldots i_p}=\sum_{q=0}^{p}(-1)^q e_{i_0\ldots\widehat{i_q}\ldots i_p}.$$

并取保持允许性的链空间

$$\Omega_p=\{v\in A_p:\partial v\in A_{p-1}\},\qquad H_p=\ker(\partial_p|_{\Omega_p})/\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}}).$$

Betti 数为

$$\beta_p=\dim\ker(\partial_p|_{\Omega_p})-\operatorname{rank}(\partial_{p+1}|_{\Omega_{p+1}}).$$

对边权采用超水平过滤

$$G_\tau=(V,\{e:w(e)\ge\tau\}),\qquad \tau\in\{0.05,0.10,\ldots,0.95\}.$$

因为本方法构造的是单一 6 节点有向环，所以当且仅当所有 6 条边均保留时 $\beta_1=1$。因此连续临界值

$$\lambda=\min_k w_k$$

就是 `loop_score`；在离散阈值上，$\beta_1(\tau)=\mathbf 1[\tau\le\lambda]$。该指标衡量最弱相位连接，而不是一般有向图中所有可能 H1 类的总复杂度。

## 3. 数据与分析协议

- 当前规范数据：Open Focus 300 首、Classical 300 首；每首均有 180s 与 300s 片段。
- 对称切分：每组 discovery 195、validation 60、holdout 45。
- 质量排除：`classical_musicnet_2305` 的 180s/300s chroma 时间步均为 94，低于冻结门槛 96；排除发生在 discovery，validation 仍为完整的 60+60。逐行记录见 `metadata/phase_lifted_path_homology_exclusions.csv`。
- 状态模型只来自 discovery/180s；模型 SHA-256：`a0dc5f846e8f7121293a9ff7e9386aad9d5c9377b5e76ae48bd67da2d5322b0d`。
- 构造校准：从 discovery 中按固定随机种子每组抽取 24 首，比较人工循环与时间打乱。校准不用于从三视角中删选当前组间结果。
- 主要比较：validation/180s；双侧 Mann–Whitney U，三视角 BH-FDR。原先的 Focus>Pop 单侧方向仅作为历史背景，不转移为新的确认性主检验。
- 时长敏感性：同一 validation 曲目的 300s 结果，以及 180/300s Spearman 相关。
- holdout：仅报告描述统计，不进行新的显著性开启或调参。
- 分类：三项 loop score 的逻辑回归，discovery 训练、validation 测试；仅作辅助预测检查。

## 4. 构造校准

| 视角 | n | 人工循环中位数 | 打乱中位数 | 中位差 | 正差比例 | FDR q | 通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rhythm phase | 48 | 1.000 | 0.329 | 0.671 | 1.000 | 3.55e-15 | 是 |
| Acoustic phase | 48 | 1.000 | 0.347 | 0.653 | 1.000 | 3.55e-15 | 是 |
| Chroma phase | 48 | 1.000 | 0.363 | 0.637 | 1.000 | 3.55e-15 | 是 |

![Construct calibration](../runs/phase_lifted_path_homology_20260802/figures/construct_calibration.png)

人工循环由同一个块序列精确平铺，因此三个视角的合成分数都达到理论上限 1.0；这是一项管线/构造校准，不是额外的经验发现。该校准只说明指标能否对预设的循环化操作作出响应，并不能证明真实音乐中的高分必然对应感知到的重复或专注效果。

## 5. validation 组间结果

| 视角 | Focus n | Classical n | Focus 180s | Classical 180s | 180s 效应 | 95% CI 下限 | 95% CI 上限 | 180s FDR | 300s 效应 | 300s FDR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acoustic phase | 60 | 60 | 0.414 | 0.361 | 0.459 | 0.269 | 0.640 | 4.42e-05 | 0.589 | 8.05e-08 |
| Chroma phase | 60 | 60 | 0.412 | 0.372 | 0.338 | 0.138 | 0.539 | 0.002 | 0.398 | 2.54e-04 |
| Rhythm phase | 60 | 60 | 0.372 | 0.355 | 0.175 | -0.045 | 0.391 | 0.099 | 0.301 | 0.004 |

![Validation distributions](../runs/phase_lifted_path_homology_20260802/figures/validation_loop_score_distributions.png)

![Validation effects](../runs/phase_lifted_path_homology_20260802/figures/validation_effect_sizes.png)

效应量为 rank-biserial(Open Focus − Classical)：正值表示 Open Focus 的 loop score 倾向更高，负值表示 Classical 更高。置信区间由固定随机种子的 3,000 次分组内 bootstrap 得到。

## 6. H1 过滤曲线

![Path H1 filtration](../runs/phase_lifted_path_homology_20260802/figures/validation_h1_filtration.png)

纵轴是在给定阈值仍保有完整 6 相位有向环的片段比例。曲线右移意味着更多片段的最弱边仍较强；它是 loop score 生存函数的路径同调解释，而不是另一个独立统计终点。

## 7. 代表性相位环

![Representative phase cycles](../runs/phase_lifted_path_homology_20260802/figures/representative_rhythm_phase_cycles.png)

图中选取 validation/180s 各组 rhythm loop score 最接近该组中位数的曲目。节点是 6 个相位，箭头数字是边权；最小边权即 loop score。线宽用于帮助观察边权差异，代表图不参与显著性检验。逐边数值见 `metadata/phase_lifted_path_homology_representative_edges.csv`。

## 8. 时长稳定性

| 视角 | 组别 | n | Spearman ρ | 300−180 中位差 | 配对 p |
|---|---:|---:|---:|---:|---:|
| Acoustic phase | Open Focus | 60 | 0.816 | 0.003 | 4.86e-05 |
| Acoustic phase | Classical | 60 | 0.484 | 0.006 | 0.301 |
| Rhythm phase | Open Focus | 60 | 0.778 | 0.000 | 0.012 |
| Rhythm phase | Classical | 60 | 0.436 | 5.17e-04 | 0.656 |
| Chroma phase | Open Focus | 60 | 0.883 | 0.000 | 0.513 |
| Chroma phase | Classical | 60 | 0.320 | 0.004 | 0.856 |

300s 不是独立样本，而是同一曲目的时长敏感性视图；因此不能把 180s 与 300s 的同向显著误写为独立复制。

## 9. holdout 描述统计

| 视角 | 组别 | n | 中位数 | 均值 |
|---|---:|---:|---:|---:|
| Acoustic phase | Classical | 45 | 0.357 | 0.358 |
| Acoustic phase | Open Focus | 45 | 0.404 | 0.411 |
| Chroma phase | Classical | 45 | 0.357 | 0.361 |
| Chroma phase | Open Focus | 45 | 0.376 | 0.402 |
| Rhythm phase | Classical | 45 | 0.349 | 0.348 |
| Rhythm phase | Open Focus | 45 | 0.363 | 0.380 |

此处没有对 holdout 计算或报告新的 p 值。当前 Classical holdout 不含钢琴独奏，并且其中部分曲目在旧切分中曾属于 discovery，不能称为 pristine 外部确认集。

## 10. 辅助分类

| 训练 n | 验证 n | CV Macro-F1 | Balanced accuracy | Validation Macro-F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| 389 | 120 | 0.744 | 0.667 | 0.667 | 0.725 |

分类结果只回答三个 loop score 是否具有联合判别信息，不等价于拓扑机制、感知效果或因果效应。

## 11. 结论与证据边界

### 可以支持

- 相位提升构造可被当前特征层稳定执行，并能以一个明确临界值连接相位一致性与 Path H1 过滤。
- discovery-only 人工循环/打乱校准可用于判断指标是否响应顺序化循环结构；具体通过情况见校准表。
- validation 中观察到的组间差异及其 300s 时长敏感性，可以描述为 Open Focus 与 Classical 在候选周期相位闭合强度上的差异。

### 不能支持

- 不能把本次 Classical 比较解释为原 Focus>Pop 假设的确认性复制。
- 不能由 6 节点人工相位环推出音乐本身具有一般意义上的复杂 H1 拓扑；这里的 H1 由预定义相位闭环构造诱导。
- 不能推出专注力改善、临床效果、生成质量或任何因果机制。
- 不能把 300s 结果当作独立数据集复制，也不能把 holdout 描述视为 pristine 外部验证。

### 方法局限

- 主导周期由全段距离对角线的中位数最小化得到，可能把缓慢结构重复与局部节拍重复混合。
- 相位节点数固定为 6，压缩了周期内部的细粒度变化；本次不做事后 K 值优化。
- `loop_score` 是最弱边统计量，对单个薄弱相位敏感；它与过滤曲线不是独立证据。
- Acoustic/Rhythm 使用固定 4 帧块，而 Chroma 的实际秒级步长取决于其时间戳；跨视角数值不应当直接作绝对大小比较。
- Classical 的风格与乐器构成可能成为组间差异来源，结果不能自动推广到所有非专注音乐。

## 12. 可复现产物

- `scripts/rerun_phase_lifted_path_homology.py`
- `metadata/phase_lifted_path_homology_features.csv`
- `metadata/phase_lifted_path_homology_tests.csv`
- `metadata/phase_lifted_path_homology_calibration.csv`
- `metadata/phase_lifted_path_homology_scale_stability.csv`
- `metadata/phase_lifted_path_homology_classification.csv`
- `metadata/phase_lifted_path_homology_representative_edges.csv`
- `metadata/phase_lifted_path_homology_exclusions.csv`
- `metadata/phase_lifted_path_homology_summary.json`
- `runs/phase_lifted_path_homology_20260802/figures/`（PNG 与 SVG）
