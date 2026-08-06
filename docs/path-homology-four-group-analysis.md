# 四组音乐三视角 Path Homology 综合报告

## 1. 分析设计

本轮将 Focus Open 作为独立第四组，完成结构、Tonnetz pitch_v2 与节奏三个视角的全量重跑。
共有 2,200 个片段、1,100 首曲目和
6,600 个片段-视角结果。原三组验证产物保持不变。

主证据层为 Validation/180s；Validation/300s 是敏感性，Discovery/180s 是探索性。
状态模型只在四组 Discovery/180s 严格等量拟合。统一 omnibus FDR 家族覆盖三个视角的 60 个指标；
两两比较覆盖 60 指标 × 6 对比。Holdout 不进入四组检验。

## 2. 主结果概览

| 视角 | 主 Omnibus | 300s Omnibus | 主/300s 交集 | Focus vs Focus Open | Focus Open vs Pop FDR 通过 | H1 非零片段 |
|---|---:|---:|---:|---:|---:|---:|
| 结构 | 7/20 | 13/20 | 7/20 | 0/20 | 2/20 | 33/255 |
| 音高（Tonnetz pitch_v2） | 14/20 | 14/20 | 14/20 | 4/20 | 3/20 | 8/255 |
| 节奏 | 14/20 | 14/20 | 14/20 | 3/20 | 1/20 | 7/255 |

关键两两对比：

- **结构**：Focus vs Focus Open 支持差异的指标为 无；Focus Open vs Pop 为 `self_transition_ratio`, `edge_count`。
- **音高（Tonnetz pitch_v2）**：Focus vs Focus Open 支持差异的指标为 `self_transition_ratio`, `transition_entropy`, `path_entropy`, `directed_recurrence`；Focus Open vs Pop 为 `self_transition_ratio`, `transition_entropy`, `path_entropy`。
- **节奏**：Focus vs Focus Open 支持差异的指标为 `self_transition_ratio`, `transition_entropy`, `directed_recurrence`；Focus Open vs Pop 为 `edge_density`。

## 3. 跨视角解读

1. **不能把 Focus Open 与 Focus 合并。** 两者即使在部分指标上没有检出差异，也只能说明在当前
   样本量、表示和冻结检验下“未拒绝相同”，不等价于证明同分布。
2. **三视角回答不同问题。** 结构视角的 SSM 与边界描述宏观段落组织；pitch_v2 描述五度/三度
   谐波骨架的有向移动；节奏视角描述局部时间组织。跨视角同时出现且方向一致的差异更值得关注，
   但仍是观察性证据。
3. **优先解释图组织与 H0。** 状态数、边数、路径熵、复现率、互惠性与 H0 通常比稀疏 H1 稳定。
   当各组 H1 中位数均为零时，显著秩检验只表明零膨胀率或尾部不同。
4. **Focus Open 的位置不是单一“更像谁”。** 应分别在结构、音高和节奏报告中查看效应方向；
   不用一个未经预注册的综合距离把多维差异压缩成单一相似度排名。
5. **没有认知或因果结论。** 本分析不能证明某类音乐提升专注，也不能把拓扑差异解释为机制因果。

## 4. 报告与产物

- [结构四组报告](path-homology-structure-four-group-analysis.md)
- [音高四组报告](path-homology-pitch-v2-four-group-analysis.md)
- [节奏四组报告](path-homology-rhythm-four-group-analysis.md)

数值清单位于 `metadata/four_group_*`，图与同调结果分别位于带 `four_group` 的隔离目录；
每幅论文图同时提供 PNG 和 SVG。
> **历史结果声明（2026-08-02）：** 本报告包含已退役的 Brain.fm Focus 组及旧四组码本，只可追溯方法，不代表当前三组开放数据集。参见 [迁移报告](open-focus-migration-report.md)。
