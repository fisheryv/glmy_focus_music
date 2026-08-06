# 音频特征提取

正式输入为 `metadata/preprocessed_segments.csv` 中通过审计的 22,050 Hz、mono、
float32 WAV。参数固定在 `configs/pipeline.toml` 的 `[features]` 段。

## 流程

1. `extract` 生成 acoustic、chroma、rhythm、modulation 四类连续特征及逐片段
   JSON sidecar；
2. `fit-states` 只读取 discovery 的 180 秒片段，拟合节奏状态、调制 tertile、
   声学 PCA/K-means；
3. `transform-states` 把同一模型应用于所有 split 和两个尺度；
4. `run` 顺序执行以上三步。

```powershell
$env:PYTHONPATH = "src"

# 查看规模与固定配置哈希
python -m features.batch run --root . --dry-run

# 全量执行或断点续跑
python -m features.batch run --root . --workers 2

# 第二次执行会复验输入、配置、模型和四类输出哈希
python -m features.batch run --root . --workers 2
```

输出清单为 `metadata/feature_segments.csv`，汇总为
`metadata/feature_summary.json`。状态模型位于 `features/models/`，使用确定性 NPZ
和 JSON 保存，不使用 pickle。完整特征目录被 Git 忽略；公开前仍需按
`docs/data-governance.md` 检查反推风险。
