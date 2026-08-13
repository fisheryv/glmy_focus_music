# 调制视角共享SMP原型 Path Homology：完整重分析报告

生成日期：2026-08-05。Jamendo Open Focus 300首与Classical 300首，每首含180 s与300 s视图；处理1,200个源片段、3,600个片段乘状态规模图，失败0。主模型固定为 $K=10$，$K=8,12$只用于表示敏感性，不按显著性选择状态数。

> 证据边界：模型只在平衡的discovery/180 s窗口上拟合；validation/180 s为主分析，validation/300 s是同曲目时长敏感性。方案在既有holdout打开后提出，因此属于探索性验证，不能更新旧holdout gate，也不能称为冻结外部确认。

## 1. 结论摘要

- $K=10$ 的20项预设指标中，180 s有 **5项**通过BH-FDR $q\le0.05$；其中 **4项**在300 s同方向且仍通过FDR：状态数、边数、边密度与互惠性。
- Open Focus观察到更多原型状态和更多有向边，但相对图更稀疏、互惠边比例更低。这说明SMP谱形转移覆盖更广、连接更选择性，不是音乐质量、专注效果或因果证据。
- **$H_1$组间差异不受支持。** $K=10$主阈值下非零$H_1$为Classical 2/60、Focus 3/60；阈值扩展到0.05后为4/60与7/60。六项$H_1$指标均未通过180 s FDR。
- $K=8,10,12$的180 s发现数为3、5、6，跨时长稳定数为3、4、4。中位状态覆盖从$4/8$、$5/10$降至$5/12$，不能因发现数更多就偏好$K=12$。
- 相比旧三状态模型，本模型保留完整SMP谱形并出现少量有限$H_1$区间，但证据仍主要来自普通有向图组织与$H_0$。

## 2. SMP共享原型模型

对mel子带能量包络 $x_b[n]$，在4 s窗、2 s步长上计算

$$
P_t(f_m)=\sum_b\left|\sum_n w[n]x_b[n+tH]e^{-i2\pi f_mn/f_s}\right|^2,\qquad
\widetilde P_t(f_m)=\frac{P_t(f_m)}{\sum_{0.5\le f\le45}P_t(f)}.
$$

保留0.5–45 Hz的178维相对SMP。先作Hellinger平方根映射，再作discovery拟合的稳健标准化：

$$
h_{tj}=\sqrt{\widetilde P_t(f_j)},\qquad
z_{tj}=
\frac{h_{tj}-\operatorname{median}_D(h_{\cdot j})}{Q_{0.75,D}(h_{\cdot j})-Q_{0.25,D}(h_{\cdot j})}.
$$

共享PCA-32为 $y_t=W_{32}(z_t-\mu_D)$，累计解释方差为0.821。Classical有14,715个、Focus有16,822个可用discovery窗口；各平衡抽14,715个。对$K\in\{8,10,12\}$分别求

$$
\min_{c_1,\ldots,c_K}\sum_t\min_k\|y_t-c_k\|_2^2,\qquad
s_t=\arg\min_k\|y_t-c_k\|_2^2.
$$

三个码本共享同一预处理；原型按原始SMP频谱质心由低到高排序。

## 3. 有向图与Path Homology

$$
C_{uv}=|\{t:s_t=u,\ s_{t+1}=v\}|,\qquad
p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}},\qquad
G_\tau=(V,\{(u,v):u\ne v,\ p_{uv}\ge\tau\}).
$$

无效窗口两侧不跨越连边；自转移不进入图。每个源状态保留至多6条非自环边。主阈值为$\tau\in\{0.50,0.60,0.70,0.80,0.90,0.95\}$，0.05–0.40只作敏感性。

对允许的正则有向路径空间$A_p$，

$$
\partial e_{v_0\ldots v_p}=\sum_{i=0}^p(-1)^ie_{v_0\ldots\widehat{v_i}\ldots v_p},
$$

$$
\Omega_p=A_p\cap\partial^{-1}(A_{p-1}),\qquad
H_p^{\mathrm{path}}(G)=
\frac{\ker(\partial_p:\Omega_p\to\Omega_{p-1})}
{\operatorname{im}(\partial_{p+1}:\Omega_{p+1}\to\Omega_p)}.
$$

令$a=1-\tau$，持久秩不变量为

$$
\rho_p(a_i,a_j)=\operatorname{rank}\operatorname{im}
[H_p(G_{a_i})\to H_p(G_{a_j})],\qquad a_i\le a_j.
$$

实现只计算$H_0/H_1$，不作$H_2$声明。

## 4. 统计设计

每个$K$独立形成20指标family，做Kruskal–Wallis检验及BH-FDR，阈值$q\le0.05$。方向效应为

$$
r_{F-C}=\frac{2U_F}{n_Fn_C}-1.
$$

300 s统一绘制$r_{180}$对$r_{300}$，它不是独立复制。

## 5. 数值结果

### 5.1 K敏感性

| K | 角色 | 180 s FDR发现 | 300 s FDR发现 | 跨时长稳定 | 中位观察状态 | 中位保留边比例 |
|---:|---|---:|---:|---:|---:|---:|
| 8 | representation_sensitivity | 3 | 9 | 3 | 4.0/8 | 0.179 |
| 10 | primary | 5 | 11 | 4 | 5.0/10 | 0.144 |
| 12 | representation_sensitivity | 6 | 10 | 4 | 5.0/12 | 0.098 |

保留边比例为$|E|/[K(K-1)]$。它随$K$下降，反映单曲欠覆盖及冻结top-6/高阈值共同作用。

### 5.2 K=10完整20指标

| 指标 | Classical中位数 | Focus中位数 | $r_{180}$ | $q_{180}$ | $r_{300}$ | $q_{300}$ |
|---|---:|---:|---:|---:|---:|---:|
| vertex_count | 5.000 | 6.000 | 0.355 | 0.005 | 0.319 | 0.040 |
| edge_count | 11.500 | 14.000 | 0.279 | 0.033 | 0.251 | 0.042 |
| edge_density | 0.667 | 0.524 | -0.386 | 0.005 | -0.302 | 0.042 |
| reciprocity | 0.873 | 0.776 | -0.336 | 0.007 | -0.236 | 0.046 |
| self_transition_ratio | 0.448 | 0.401 | -0.228 | 0.070 | -0.202 | 0.093 |
| transition_entropy | 0.843 | 0.860 | 0.156 | 0.216 | 0.244 | 0.042 |
| path_entropy | 1.087 | 1.119 | 0.080 | 0.643 | 0.038 | 0.962 |
| directed_recurrence | 0.130 | 0.109 | -0.199 | 0.100 | -0.249 | 0.042 |
| h0_betti_auc | 1.300 | 1.625 | 0.220 | 0.075 | 0.257 | 0.042 |
| h0_betti_mean | 2.917 | 3.583 | 0.236 | 0.063 | 0.259 | 0.042 |
| h0_betti_max | 4.000 | 5.000 | 0.232 | 0.063 | 0.240 | 0.042 |
| h0_interval_count | 4.000 | 5.000 | 0.232 | 0.063 | 0.240 | 0.042 |
| h0_observed_persistence | 1.450 | 1.775 | 0.212 | 0.081 | 0.259 | 0.042 |
| h0_censored_count | 1.000 | 1.000 | 0.299 | 0.005 | 0.154 | 0.159 |
| h1_betti_auc | 0.000 | 0.000 | 0.017 | 0.683 | 2.78e-04 | 1.000 |
| h1_betti_mean | 0.000 | 0.000 | 0.017 | 0.683 | 2.78e-04 | 1.000 |
| h1_betti_max | 0.000 | 0.000 | 0.017 | 0.683 | -0.000 | 1.000 |
| h1_interval_count | 0.000 | 0.000 | 0.017 | 0.683 | -0.000 | 1.000 |
| h1_observed_persistence | 0.000 | 0.000 | -0.000 | 1.000 | 0.017 | 0.453 |
| h1_censored_count | 0.000 | 0.000 | 0.017 | 0.683 | -0.000 | 1.000 |

### 5.3 H1稀疏性

| K | Classical主阈值非零 | Focus主阈值非零 | Classical扩展阈值非零 | Focus扩展阈值非零 |
|---:|---:|---:|---:|---:|
| 8 | 0/60 | 1/60 | 1/60 | 2/60 |
| 10 | 2/60 | 3/60 | 4/60 | 7/60 |
| 12 | 0/60 | 2/60 | 3/60 | 10/60 |

机制示例为 focus_jamendo_1571054__180s。有限$H_1$区间出生于$\tau=0.50$、死亡于$\tau=0.20$，寿命0.30。它只解释计算机制，不是组间证据。

## 6. 可视化

![modulation_smp_prototypes](../runs/modulation_smp_prototype_path_homology/modulation_smp_prototypes.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_prototypes.svg)

![modulation_smp_example_trajectory](../runs/modulation_smp_prototype_path_homology/modulation_smp_example_trajectory.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_example_trajectory.svg)

![modulation_smp_directed_graph](../runs/modulation_smp_prototype_path_homology/modulation_smp_directed_graph.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_directed_graph.svg)

![modulation_smp_filtration](../runs/modulation_smp_prototype_path_homology/modulation_smp_filtration.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_filtration.svg)

![modulation_smp_persistence](../runs/modulation_smp_prototype_path_homology/modulation_smp_persistence.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_persistence.svg)

![modulation_smp_group_distributions](../runs/modulation_smp_prototype_path_homology/modulation_smp_group_distributions.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_group_distributions.svg)

![modulation_smp_betti_curves](../runs/modulation_smp_prototype_path_homology/modulation_smp_betti_curves.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_betti_curves.svg)

![modulation_smp_effect_sizes](../runs/modulation_smp_prototype_path_homology/modulation_smp_effect_sizes.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_effect_sizes.svg)

![modulation_smp_duration_stability](../runs/modulation_smp_prototype_path_homology/modulation_smp_duration_stability.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_duration_stability.svg)

![modulation_smp_k_sensitivity](../runs/modulation_smp_prototype_path_homology/modulation_smp_k_sensitivity.png)

[SVG](../runs/modulation_smp_prototype_path_homology/modulation_smp_k_sensitivity.svg)

## 7. 解释与局限

1. 共享SMP原型保留谱峰位置、宽度和多峰形状，比三状态总强度更有表达力。
2. Focus覆盖更多SMP原型，但图相对更稀疏、互惠性更低；该模式在$K=8,10,12$及180/300 s间大体一致。
3. 共享原型提高了环的可表达性，但六项$H_1$组间检验不显著，不能宣称稳定$H_1$差异。
4. top-6和0.50–0.95来自旧冻结图族，未针对10状态图调参；这避免结果驱动调参，但可能压低分散转移概率。
5. K=12的中位覆盖仅5/12，不建议继续无约束增加$K$。
6. 本方案在旧holdout打开后提出，不能并入旧确认性fingerprint；升级前必须冻结当前哈希并用新数据验证。
7. 新分支未覆盖modulation_tertile。旧模型解释总调制强度级别转移；本模型解释完整SMP谱形原型转移。

## 8. 复现与审计

PowerShell命令：

    $env:PYTHONPATH = "packages/pyglmy/src;src"
    .\.venv\Scripts\python.exe scripts\run_modulation_smp_prototype_analysis.py
    .\.venv\Scripts\python.exe scripts\render_modulation_smp_prototype_report.py

模型集合SHA-256：9206ebd71c402c83ac776297b32de0c81f3f225ed9c4e8e218e980f94c1d6943。共享变换SHA-256：fd921ffb7505eef2f221bb47d869ff9c241ae6f0960bb5698296626157fe93f1。数值表位于metadata/modulation_smp_prototype系列文件，模型位于features/models/modulation_smp系列文件，PNG/SVG及哈希清单位于runs/modulation_smp_prototype_path_homology。
