# 规范 Focus 数据集

自 2026-08-02 起，规范 `focus` 组由 300 首公开可获取、逐曲 CC 许可的 Jamendo 音乐构成，不再使用 Brain.fm 音频。实验标签仍为 `focus`，以保持代码接口稳定；来源由 `source_dataset`、`source_url` 和逐曲许可字段明确记录。

## 规模与切分

| split | Focus | Pop | Classical | 合计 |
|---|---:|---:|---:|---:|
| discovery | 195 | 224 | 221 | 640 |
| validation | 60 | 76 | 79 | 215 |
| holdout | 45 | 0 | 0 | 45 |
| 合计 | 300 | 300 | 300 | 900 |

Focus 总时长 29.50 小时；三组合计 76.68 小时。每首曲目具有 180 s 主尺度和 300 s 敏感性尺度，共 1,800 个预处理片段。

## 来源与许可

- 292 首来自当前 Jamendo API 筛选记录；8 首来自同一开放 Focus 早期本地候选池。
- 所有 300 首均有来源页、逐曲 Creative Commons 许可、SHA-256 和 MPEG 内容审计。
- 许可分布：CC BY 6、CC BY-SA 18、CC BY-NC 8、CC BY-NC-SA 41、CC BY-NC-ND 227。
- `redistribution_allowed=true` 表示原始文件可在遵守逐曲 CC 条款时再分发；它不免除署名、非商业、禁止演绎或相同方式共享等条件。
- 因多数曲目含 NC/ND 限制，本数据集不应表述为“无条件开放”，也不保证符合所有严格 Open Definition；其改进点是来源公开、许可逐曲可审计并摆脱专有授权依赖。
- Jamendo 的 `mp32` 是下载格式枚举，不表示 320 kbps；实测码率按原值记录。

规范文件为 `metadata/focus_manifest.csv`、`metadata/track_index.csv`、`metadata/licenses.csv` 和三个 split CSV。候选与排除历史保留在 `metadata/focus_open_candidates.csv`。

## 目录

- 规范原音频：`data_raw/focus_music/`
- 规范 WAV：`features/audio/{180s,300s}/focus/`
- 规范特征：`features/{acoustic,chroma,rhythm,modulation,structure}/.../focus/`
- 已排除候选：`data_raw/focus_open_music_excluded/`，不属于数据集
- 旧 Brain.fm：`restricted_archive/brainfm_legacy_2026-08-02/`，禁止公开或再分发

完整审计和迁移边界见 [open-focus-migration-report.md](open-focus-migration-report.md)。
