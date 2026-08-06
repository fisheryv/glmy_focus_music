# ACE-Step 纯 Path Homology 指纹推理期引导方案

日期：2026-08-03  
当前指纹：`focus_path_homology_fingerprint_v2`  
冻结 JSON SHA-256：`9bf64f3c1d79c12ec428f1d9f552827d07e9f5c445d9236e7ab676699a62ef1f`  
适用生成：ACE-Step 1.5 Turbo、PyTorch、180 s text-to-music

## 摘要

本报告以最新纯 Path Homology 验证结论替代历史“2 个 TDA H0 + 相位提升”方案。
当前生成目标为 51 维 `L+P` 指纹：

- `L`：Pitch、Rhythm、Modulation 三个局部 Path Homology 块；
- `P`：Acoustic phase 与 Chroma phase 的相位提升 Path Homology；
- `S`：Structure Path Homology 只作独立宏观辅助监测。

`L+P` 的 discovery 训练判别器在 validation/180 s 上 balanced accuracy=0.933、
ROC-AUC=0.982。相位加入 `L` 的 Δpseudo-F=+6.910、FDR=0.002；结构加入
`L+P` 的 Δpseudo-F=-4.477、p=1.000。因此引导目标使用 `L+P`，不加入结构，
也不使用任何 Vietoris–Rips TDA 端点。

工程路线保持分阶段：

1. exact Path Homology shadow scoring；
2. 同 prompt 候选池 exact reranking；
3. 训练可微 latent-to-topology surrogate（LTSN）；
4. 在 ACE-Step Turbo 第 4–6 步施加 RMS 裁剪的弱梯度；
5. 解码后 exact 复核并执行音质、prompt 一致性和盲听非劣检验。

当前已完成的是冻结指纹和设计，不是采样期引导效果验证。采样 corrector 必须默认
关闭；只有 exact reranking 和 LTSN 均通过资格门槛后才可开启实验性引导。

![当前证据](../runs/ace_topology_guidance_design/figures/qualification_evidence.png)

[当前证据 SVG](../runs/ace_topology_guidance_design/figures/qualification_evidence.svg)

## 1. 当前拓扑目标

### 1.1 局部块 L

每个局部视角使用冻结有向状态图和 20 个预设图/Path Homology 描述子：

- Pitch：beat-synchronous chroma → Tonnetz → 16 状态；
- Rhythm：8 维节奏窗口 → 10 状态；
- Modulation：谱调制能量 → Low/Medium/High 三状态。

对状态序列构造 top-6 非自环条件转移图，并在
\(\tau\in\{0.50,0.60,0.70,0.80,0.90,0.95\}\) 上计算：

\[
H_p=\ker\partial_p/\operatorname{im}\partial_{p+1}.
\]

局部块既包含状态数、边数、密度、自转移、熵、定向复现等图描述，也包含 H0/H1
Betti 与 persistence 汇总。主分析不支持把 H1/H2 单独作为方向性控制目标。

### 1.2 相位块 P

相位提升从块级距离矩阵选择主周期 \(P^*\)，映射到 6 个有向相位节点。环的
最弱边：

\[
\lambda=\min_k w_k
\]

即 `loop_score`。当前主 `P` 使用：

- `path_acoustic_phase__loop_score`；
- `path_chroma_phase__loop_score`。

Rhythm phase 只作敏感性分析，不进入主目标。Acoustic/Chroma phase 在
validation/180 s 中均为 Focus 更高，且 `P | L` 条件残差仍有组间分离。

### 1.3 结构块 S

Structure PH 在宏观段落状态图上工作，单块可分，但没有改善 `L+P`：

\[
F_{L+P+S}-F_{L+P}=-4.477.
\]

它仅用于检测生成音乐是否发生宏观段落退化，不参与主梯度。未来若要加入，必须在
新数据和预注册权重下重新验证。

## 2. 51 维纯 Path Homology 指纹

### 2.1 discovery 块变换

对每个块只用 discovery/180 s 拟合：

\[
Z_b=\frac{(X_b-\mu_b)W_b}{\sqrt{r_b}},
\qquad W_b^\top\Sigma_bW_b\approx I.
\]

有效秩为 Pitch 13、Rhythm 13、Modulation 14、两个 phase 各 1。

固定等权融合：

\[
L=\frac{1}{\sqrt3}
[Z_{pitch}|Z_{rhythm}|Z_{modulation}],
\]

\[
P=\frac{1}{\sqrt2}
[Z_{acoustic\ phase}|Z_{chroma\ phase}],
\]

\[
LP=\frac{1}{\sqrt2}[L|P].
\]

最终维数为 49+2=51。

### 2.2 Focus 判别分数

只在 discovery 上拟合 L2 逻辑回归：

\[
S_F(x)=w^\top LP(x)+b,
\qquad
p_F(x)=\sigma(S_F(x)).
\]

为了不把生成推向极端判别分数，使用 discovery Focus logit 的第 10 百分位
\(\tau_F\) 作为进入目标带的下界：

\[
L_{PH}(x)=\left[\max(0,\tau_F-S_F(x))\right]^2.
\]

该目标比旧版“到 Focus 单类中心的对称距离”更适合当前数据：它使用 Focus 与
Classical 的方向性边界，且在 validation 上有 0.982 AUC。但 \(p_F\) 只是数据集
判别概率，不是注意力有效概率。

![指纹组成](../runs/focus_path_homology_fingerprint_v2/figures/fingerprint_composition.png)

[指纹组成 SVG](../runs/focus_path_homology_fingerprint_v2/figures/fingerprint_composition.svg)

## 3. Exact teacher 与 LTSN

### 3.1 为什么 exact Path Homology 不直接反传

精确管线含状态量化、top-k、硬阈值、矩阵秩、周期 `argmin`、相位分箱和最弱边
`min`。这些步骤适合离线标签和最终复核，不适合每个采样步在线求导。

因此：

- exact PH 是教师和最终裁判；
- LTSN 提供采样内可微近似；
- 代理改善必须由 exact PH 复核。

### 3.2 正确的轨迹标签

ACE-Step flow 模型的预测干净潜变量为：

\[
\hat x_0=x_t-tv_t.
\]

必须解码第 4、5、6 步各自的 \(\hat x_0\)，再计算该快照自己的 51 维 PH 坐标和
Focus logit。不能把最终音频标签复制给所有中间步骤。

训练数据应包括：

- 真实 Focus/Classical 的 VAE 重建；
- 多 prompt、多 seed 的无引导 ACE-Step 输出；
- 第 4–6 步 \(\hat x_0\) 快照；
- 静音、削波、机械短循环等安全负例。

同一曲目、prompt 或生成轨迹的不同版本不能跨训练/验证切分。

### 3.3 LTSN 网络

完整的分层网络、损失函数、数据切分、训练配置与资格门槛见
`docs/ltsn-network-architecture-and-training.md`。本节只保留接口级摘要。

输入约为 `[B,4500,64]`：

- LayerNorm；
- Conv1d 64→128，kernel 9、stride 5；
- TCN residual blocks，dilation 1/4/16/64；
- timestep embedding；
- masked attention pooling + mean/std pooling；
- 输出 51 维 PH 坐标、Focus logit 与不确定度。

输出完整 51 维坐标有助于发现代理只拟合总分却破坏单视角的情况。

### 3.4 代理损失

\[
L_{coord}=\sum_j\alpha_j
\operatorname{Huber}(\hat z_j-z_j),
\]

\[
L_{score}=\operatorname{Huber}(\hat S_F-S_F),
\]

\[
L_{rank}=\max(0,1-operatorname{sign}(S_i-S_k)(\hat S_i-\hat S_k)),
\]

\[
L_{sur}=L_{coord}+0.5L_{score}+0.2L_{rank}+0.1L_{NLL}+0.1L_{OOD}.
\]

这些系数是 discovery/development 起点，必须在最终实验前冻结。

### 3.5 LTSN 资格门槛

- Focus logit Spearman \(\rho\ge0.70\)；
- 51 维坐标中位 Spearman \(\rho\ge0.50\)；
- Pitch/Rhythm/Modulation/Phase 各块聚合距离 \(\rho\ge0.50\)；
- 高/低四分位排序准确率 ≥0.65；
- 90% 预测区间覆盖率 0.85–0.95；
- 优化样本的 exact score 与代理 score 同向；
- 不确定度或 OOD 超阈时自动 no-op。

## 4. ACE-Step 采样插入点

当前本地 ACE-Step commit：
`a5632cda3084f1088e69b2057dde7047e1bb4839`。

Turbo 采样循环：

```text
ACE-Step-1.5/acestep/models/turbo/modeling_acestep_v15_turbo.py
```

现有顺序是：

```text
DiT velocity → Euler/Heun → DCW → Repaint injection
```

建议顺序：

```text
DiT velocity → Euler/Heun → DCW → topology corrector → Repaint injection
```

Repaint 放在最后，以保证受保护区域不被拓扑梯度覆盖。初版仅实现 PyTorch Turbo
ODE/Euler，不同时修改 MLX、Base、SFT 或 XL。

Turbo 是 CFG-distilled；`guidance_scale` 不是拓扑控制接口。

![门控架构](../runs/ace_topology_guidance_design/figures/guidance_architecture.png)

[门控架构 SVG](../runs/ace_topology_guidance_design/figures/guidance_architecture.svg)

## 5. 采样校正公式

DiT 保持 `torch.no_grad()`，仅对 detached \(\hat x_0\) 求代理梯度：

\[
\tilde x_0=\operatorname{stopgrad}(x_t-tv_t),
\qquad
\tilde x_0.\mathrm{requires\_grad}=\mathrm{True}.
\]

主引导损失：

\[
\mathcal L_t=
L_{PH}(g_\phi(\tilde x_0,t))
+\lambda_uU_{OOD}
+\lambda_mL_{macro}.
\]

`L_macro` 默认权重为 0；只有结构监测越过冻结安全界时触发 no-op，而不是把结构
梯度加入主目标。

梯度：

\[
g_t=\nabla_{\tilde x_0}\mathcal L_t.
\]

应用有效帧掩码、可选 1 s 低通与逐样本 RMS clip：

\[
\delta_t=-\eta_k\operatorname{LPF}(M\odot g_t),
\]

\[
\delta_t\leftarrow\delta_t
\min\left(1,
\frac{\rho\operatorname{RMS}(\tilde x_0)}
{\operatorname{RMS}(\delta_t)+\epsilon}\right).
\]

映射回下一时刻：

\[
x_{t_{next}}\leftarrow
x_{t_{next}}+(1-t_{next})a_k\delta_t.
\]

## 6. 步骤窗口与强度

默认 8 步、shift=3.0：

\[
[1.000,0.955,0.900,0.833,0.750,0.643,0.500,0.300].
\]

只在第 4–6 步启用，三角权重 \([0.5,1.0,0.5]\)。跳过高噪声前三步和直接输出
\(x_0\) 的最后一步。

RMS clip development 候选：0.25%、0.5%、1.0%。选择一次后冻结，最终实验不得
继续挑选。

![步骤窗口](../runs/ace_topology_guidance_design/figures/sampler_guidance_schedule.png)

[步骤窗口 SVG](../runs/ace_topology_guidance_design/figures/sampler_guidance_schedule.svg)

## 7. Corrector 接口

```python
class TopologyCorrector(Protocol):
    def __call__(
        self,
        *,
        xt_next: Tensor,
        xt_before_step: Tensor,
        velocity: Tensor,
        timestep: float,
        next_timestep: float,
        step_index: int,
        attention_mask: Tensor,
        repaint_mask: Tensor | None,
    ) -> Tensor: ...
```

伪代码：

```python
xt_next = euler_step(xt, velocity, timestep, next_timestep)
xt_next = dcw_corrector.apply(xt_next, pred_clean, timestep)

if topology_corrector is not None:
    xt_next = topology_corrector(
        xt_next=xt_next,
        xt_before_step=xt,
        velocity=velocity,
        timestep=timestep,
        next_timestep=next_timestep,
        step_index=step_index,
        attention_mask=attention_mask,
        repaint_mask=repaint_mask,
    )

xt_next = repaint_injection(xt_next, ...)
```

corrector 为 `None`、强度为零、步骤不在窗口、预测 OOD 或出现 NaN/Inf 时必须
逐位 no-op。

## 8. 分阶段实验

![阶段门槛](../runs/ace_topology_guidance_design/figures/stage_gates.png)

[阶段门槛 SVG](../runs/ace_topology_guidance_design/figures/stage_gates.svg)

### 阶段 0：Shadow mode

- 不改变生成；
- 保存第 4–6 步轨迹、最终潜变量与音频；
- exact 计算 51 维 PH、Focus logit 与 Structure 辅助指标；
- 记录模型、权重、VAE、fingerprint JSON 和代码 SHA-256。

### 阶段 1：Exact reranking

32 prompt × 8 seed，共 256 首 180 s 候选。比较：

1. 第一个候选；
2. 仅质量重排；
3. 质量 + PH `focus_band_loss` 重排。

通过门槛：

- exact PH 损失中位改善 ≥10%；
- 按 prompt 聚类 bootstrap 95% CI 不跨 0；
- 质量和 prompt 一致性满足非劣；
- 候选多样性没有明显下降。

### 阶段 2：LTSN

在未见 prompt、seed family 和轨迹上执行第 3.5 节资格测试。未通过时只保留 exact
reranking，不进入采样器。

### 阶段 3：采样开发

比较 0.25%、0.5%、1.0% RMS clip，以及：

- `L` only；
- `P` only；
- `L+P`；
- `L+P` 加 Structure no-op gate。

这属于 development，只能选择一次最终设置。

### 阶段 4：一次最终确认

32 prompt × 8 seed，每个 seed 生成 baseline/guided 配对，共 512 首 180 s 音频。
使用未参与代理训练和强度选择的新 prompt/seed。结果不得再次用于改目标或权重。

## 9. 评价指标

### 9.1 主要终点

\[
\Delta_i=L_{PH}^{guided}(i)-L_{PH}^{base}(i).
\]

负值表示生成进入更典型的 Focus 判别带。报告中位差、Hodges–Lehmann 位移和按
prompt 聚类 bootstrap 95% CI。

### 9.2 拓扑诊断

- exact Focus logit/probability；
- L、P 和各局部块距离；
- 40 项可解释方向签名；
- Structure PH 辅助指标；
- 代理与 exact 差异。

### 9.3 质量与安全

- prompt/CLAP 一致性或冻结等价指标；
- LUFS、true peak、削波、静音率；
- 频谱异常和重复坍缩；
- 音频质量、Focus-like 但不过度单调的双盲评分；
- 推理耗时、峰值显存、NaN 和失败率。

质量采用非劣检验。建议预先冻结 -0.20 SD 的标准化非劣界；若 PH 改善但质量越界，
仍判引导失败。

## 10. 风险与停止条件

### Proxy gaming

若代理分数改善但 exact PH 不改善，立即停止。不能只报告 LTSN 分数。

### 音乐退化

持续压低状态数、熵和 H0 可能导致过度平滑；持续抬高 phase loop 可能导致机械
短循环。分布带损失、弱中段校正、RMS clip 和 Structure 宏观监测共同用于限制。

### 表示冲突

Pitch 与 Rhythm 的 edge density、reciprocity 方向不同。不能把同名指标跨视角
统一写成“越高越好”；必须通过冻结的 51 维变换与分类器计算。

### 因果边界

成功只能称为 “Focus-like Path Homology steering”。不能声称提高专注力、学习
效果、治疗作用或生产率。

### 停止规则

出现任一情况即停止升级：

- exact reranking 改善不足 10%；
- LTSN 未通过相关、排序、校准门槛；
- 代理与 exact 系统性背离；
- 质量、prompt 一致性或多样性越过非劣界；
- Structure 监测显示宏观退化；
- NaN、OOM 或失败率显著增加；
- 需要用已开启 validation/holdout 继续调参。

## 11. 实施文件

当前已完成：

- `configs/focus_path_homology_fingerprint_v2.toml`
- `scripts/build_focus_path_homology_fingerprint.py`
- `metadata/focus_path_homology_fingerprint_v2.json`
- `metadata/focus_path_homology_fingerprint_v2_scores.csv`
- `metadata/focus_path_homology_fingerprint_v2_directions.csv`
- `docs/focus-path-homology-fingerprint-v2-analysis.md`

后续建议新增：

```text
src/generation/path_homology_surrogate.py
src/generation/path_homology_corrector.py
scripts/collect_ace_path_homology_trajectories.py
scripts/train_path_homology_surrogate.py
scripts/evaluate_path_homology_guidance.py
tests/test_path_homology_corrector.py
tests/test_path_homology_surrogate.py
```

## 12. 最低测试要求

- fingerprint JSON 哈希不匹配时拒绝启动；
- `None`/零强度逐位恒等；
- 相同 seed 确定性；
- 只在第 4–6 步调用；
- RMS clip 不越界；
- padding/Repaint 区域不被改写；
- OOD/NaN/Inf 自动 no-op；
- batch 与单样本一致；
- guided 输出必须执行 exact verifier；
- Structure 辅助层不能意外进入主梯度；
- TDA 字段出现在 v2 输入时拒绝加载。

## 13. 最终结论

最新方案已经不再依赖 TDA。ACE-Step 推理期的正确目标是纯 Path Homology
`L+P`：局部 Pitch/Rhythm/Modulation PH 与中尺度 Acoustic/Chroma phase PH。
Structure PH 是宏观辅助层。

该指纹具备强 validation 区分能力，并解决了旧单类 TDA 核心距离偏好 Classical
的问题。但它仍需完成 exact reranking、LTSN 和配对生成验证。现阶段推荐状态为：

```text
exact_scoring = enabled
shadow_mode = enabled
experimental_reranking = allowed
sampling_guidance = disabled_until_gates_pass
```

因此，下一步不是继续修改特征，而是固定 v2 哈希，收集无引导 ACE-Step 轨迹，
先完成 exact reranking 可辨识性试验。
