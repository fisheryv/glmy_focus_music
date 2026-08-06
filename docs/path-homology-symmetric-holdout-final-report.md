# 四视角 Path Homology 对称 holdout 最终研究报告

生成日期：2026-08-02。切分版本：`symmetric_holdout_v2`。

## 摘要

本研究按“局部三视角（音高、节奏、调制）先融合，再与宏观结构视角整合”的方案，重新完成了 600 首音乐的全流程 Path Homology。两组均严格切分为 195 discovery、60 validation、45 holdout。所有状态模型只用 discovery/180 s 拟合；validation 用于方案冻结；holdout 在 SHA-256 门控后只开启一次，且没有据其结果改参数、指标、方向、阈值、FDR 家族或融合权重。

最终结果是：局部三视角融合在 holdout/180 s 上确认组间分离（pseudo-F=5.359，p=0.001），但它没有优于单独音高视角；加入结构也没有增量。44 个 validation 锁定方向性指标中，43 个方向一致，42 个在四视角联合 BH-FDR q≤0.10 后复现。证据支持“Focus 与 Classical 的状态转移组织不同”，不支持“融合必然增强”、稳定 H1/H2、注意力效果、治疗作用或因果机制。

## 1. 证据地位与限制

- validation/180 s：方案冻结层；融合方案因参考过既往单视角结果，仍属于探索性整合。
- validation/300 s：时长敏感性。
- holdout/180 s：哈希门控后的单次操作性最终确认。
- 重要限制：Classical holdout 在旧切分中曾属于 discovery，因此不能称为 pristine 外部确认集；Classical holdout 还不含钢琴独奏，不能代表完整古典总体。
- 本研究是观察性声学比较，不支持认知、临床、生成质量或因果结论。

## 2. 流程与防泄漏设计

```mermaid
flowchart LR
    A["600 tracks<br/>Focus 300 + Classical 300"] --> B["Symmetric split<br/>195 / 60 / 45 per group"]
    B --> C["Discovery 180 s only<br/>fit all state models"]
    C --> D["Transform all splits<br/>no holdout summary"]
    D --> E["Four-view Path Homology"]
    E --> F["Validation 60 + 60<br/>freeze metrics and weights"]
    F --> G["SHA-256 gate<br/>f75a04a1b6ef…"]
    G --> H["One-time holdout 45 + 45"]
    H --> I["Final report<br/>no adaptation"]
```

预处理为 22,050 Hz、mono、float32、双遍 EBU R128 至 −15 LUFS、峰值上限 −1 dBFS。1200 个 180/300 s WAV 均通过哈希、响度、峰值和路径审计。新路径使用 `features/audio_symmetric_holdout_v2/`；音频字节与已验证旧预处理完全一致，避免重新编码漂移。

## 3. 四视角表示

1. 音高：beat-synchronous chroma → Tonnetz → 16 状态 discovery 平衡码本。
2. 节奏：8 维节奏窗口 → 标准化 → 10 状态聚类。
3. 调制：谱调制能量 → discovery 平衡三分位 → Low/Medium/High。
4. 结构：声学 SSM → Foote novelty 边界 → 段级 16 状态原型。

每个状态序列构造成 top-6 非自环有向图；主过滤阈值固定 0.50–0.95，扩展敏感阈值 0.05–0.95。指标族固定为 20 个图/H0/H1 描述子。

## 4. validation 结果与冻结决策

| 视角 | 180 s FDR 发现 | 300 s 同方向再现 | validation 融合 pseudo-F |
|---|---:|---:|---:|
| pitch | 14/20 | 13 | 7.588 |
| rhythm | 14/20 | 14 | 6.474 |
| modulation | 12/20 | 12 | 5.666 |
| structure | 6/20 | 5 | 2.503 |

局部等权融合（音高、节奏、调制各 1/3）在 validation/180 s 的 pseudo-F=6.696、p=0.001；音高单视角为 pseudo-F=7.588。局部融合相对音高的增量未得到支持。结构以 0.5 权重加入层级融合后，pseudo-F 降至 4.291；结构增量 Δpseudo-F=-2.405，单侧 p=1.000。

因此冻结：`local` 为 holdout 主终点，`hierarchical` 为次终点；局部权重 1/3–1/3–1/3，层级权重 local 0.5 / structure 0.5，不再改变。

![Validation fusion ablation](../runs/multiview_fusion/figures/multiview_permanova_ablation.png)

[SVG](../runs/multiview_fusion/figures/multiview_permanova_ablation.svg)

## 5. holdout 单次确认

门控 SHA-256：`f75a04a1b6efb82a9ca031900628adfcf54dee06a1813a8bad6183c8c3fa0617`。执行记录 SHA-256：`32e454d9ed18152c174ab0ad7784554e1c575763ccec056415c99241dff4507b`。

| 冻结表示 | 180 s pseudo-F | p | FDR q |
|---|---:|---:|---:|
| pitch | 7.402 | 0.001 | 0.002 |
| rhythm | 5.342 | 0.001 | 0.002 |
| modulation | 2.597 | 0.005 | 0.005 |
| structure | 2.829 | 0.005 | 0.005 |
| local | 5.359 | 0.001 | 0.001 |
| hierarchical | 3.853 | 0.001 | 0.002 |

局部融合相对音高：Δpseudo-F=-2.044，单侧 p=1.000。加入结构：Δpseudo-F=-1.506，单侧 p=1.000。两者均为负，说明“组间可分”不等于“融合带来额外信息”。

![Holdout frozen endpoints](../runs/symmetric_holdout_final/figures/holdout_frozen_endpoints.png)

[SVG](../runs/symmetric_holdout_final/figures/holdout_frozen_endpoints.svg)

### 5.1 validation 选定方向的复现

| 视角 | 锁定指标 | 方向一致 | BH q≤0.10 复现 |
|---|---:|---:|---:|
| pitch | 14 | 14 | 14 |
| rhythm | 14 | 14 | 14 |
| modulation | 10 | 10 | 10 |
| structure | 6 | 5 | 4 |

结构视角未复现的是 `edge_density`（方向反转）与 `reciprocity`（方向一致但 q=0.173）。其余三视角全部锁定指标复现。

![Directional replication](../runs/symmetric_holdout_final/figures/holdout_directional_replication.png)

[SVG](../runs/symmetric_holdout_final/figures/holdout_directional_replication.svg)

## 6. 科学结论

### 支持

- Focus 与 Classical 在四种状态转移表示上均存在可重复的组间差异；音高视角最强。
- Focus 更常呈现较少状态/边、更高自转移与定向复现、更低路径熵的局部组织；该方向在音高、节奏和调制中最稳定。
- 局部三视角融合本身是有效的组间表征，并在 300 s 敏感性中保持分离。

### 不支持

- 局部三视角融合优于音高单视角：validation 与 holdout 的增量均不支持。
- 结构对局部融合有正增量：加入结构在 validation 与 holdout 都降低 pseudo-F。
- 稳定、普遍或 Focus 特异的 H1/H2。调制 H1 为零，其他视角 H1 稀疏；最终锁定方向性家族没有 H1 指标。
- 由声学拓扑差异推出专注力提升、临床疗效、生成质量或因果机制。

## 7. 可复现与审计产物

- `metadata/split_assignment_v2.csv`
- `metadata/preprocessed_segments.csv`
- `metadata/feature_segments.csv`
- `features/models/state_model.json`
- `metadata/holdout_gate.json`
- `metadata/holdout_confirmation_execution.json`
- `metadata/holdout_confirmation_permanova.csv`
- `metadata/holdout_confirmation_incremental.csv`
- `metadata/holdout_confirmation_directional_metrics.csv`
- 四份更新后的单视角报告位于 `docs/path-homology-*-analysis.md`。

最终适应性审计：参数、指标、方向、融合权重、阈值和 FDR 家族在 holdout 后均未改变。
