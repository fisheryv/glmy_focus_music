# Acoustic features

每个片段保存按 1 秒窗、0.5 秒步长池化的 64 维 log-mel、39 维 MFCC（含一阶、
二阶差分）、12 维 chroma、7 维 spectral contrast 和 32 维 tempogram，共 154 维。

统一声学原型只用 discovery 180 秒数据拟合：标准化后降到 32 维 PCA，再映射到
64 个 MiniBatch K-means 状态。validation、holdout 和 300 秒数据只做变换。
