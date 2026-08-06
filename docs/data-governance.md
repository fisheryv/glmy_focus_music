# 数据治理与公开边界

## 分级

| 级别 | 内容 | 公开规则 |
|---|---|---|
| Restricted | 归档的 Brain.fm 原音频、可逆 WAV、私有文件名映射 | 不公开、不再分发 |
| Publicly licensed | 规范 Focus、Classical 原音频；归档 Pop 原音频 | 仅按逐曲许可条款公开；NC/ND 不属于无条件开放 |
| Controlled | 生成音频、人工标注、许可边界待复核文件 | 审核后决定 |
| Derived | 状态序列、边表、拓扑描述子和聚合统计 | 评估反推风险后公开 |
| Public | 代码、参数、审计台账和允许公开的图表 | 可公开 |

## 强制规则

1. 规范 `focus` 是公开来源、逐曲 CC 许可的 Jamendo 曲库；`restricted=false`，且每首必须有非空来源 URL 和已核实许可。`restricted=false` 不等于无条件开放。
2. `redistribution_allowed=true` 不代表无条件使用。署名、NC、ND、SA 以 `metadata/licenses.csv` 和来源页为准。
3. Brain.fm 只能位于 `restricted_archive/brainfm_legacy_2026-08-02/`。该目录被 Git 忽略，不能进入发布包、公开日志或论文补充材料。
4. Pop 只能作为历史对照保存在 `dataset_archive/pop_music_legacy_2026-08-02/`，不得进入新的 Focus–Classical 规范清单、模型拟合或确认性检验。
5. 禁止把 Brain.fm 或旧三组历史统计改名后当作当前 Focus–Classical 结果。相关分析必须重新运行。
6. 状态模型只能在 discovery/180s 上拟合；validation 用于方案冻结前的验证；Open Focus 与 Classical 各45首 holdout 只允许在全部方法冻结后开启一次，不得用于调参或指标选择。
7. Jamendo `mp32` 不得表述为 320 kbps。报告实测码率和原始许可，不转码伪造质量证据。
8. 公开任何音频集合前，应再次执行 SHA-256、MPEG payload 去重、split 泄漏和逐曲许可审计。

当前迁移审计见 `metadata/open_focus_migration_audit.json`；旧数据逐文件清单仅保存在受限归档中。
