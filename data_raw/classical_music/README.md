# Classical control audio

正式 Classical 对照集共 300 首，清单及逐曲状态见 `metadata/control_classical.csv`。

- 基础池：95 首 Musopen 录音，包含钢琴独奏和弦乐四重奏。
- 扩展池：205 首 MusicNet 录音；排除 `source=Museopen` 以及与基础池重叠的作品，再按作曲家和编制平衡选取。
- 只有状态为 `verified` 且通过格式、时长和 SHA-256 校验的文件，才能进入统一实验索引。
- discovery/validation 按作曲家整组划分，避免同一作曲家跨集合。

基础池与扩展池的许可不同，实验产物应保留清单中的来源、作品和许可字段。
