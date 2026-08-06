# Modulation features

每个片段保存 64 通道 mel 包络合并后的 0.5–45 Hz broadband modulation spectrum，
以及 0.5–4、4–8、8–12、12–20、20–30、30–45 Hz 六个宽频带和 8、12–20、
32 Hz 三个重点频带的归一化能量。重点频带使用 discovery 180 秒数据的 tertile
边界量化为三元状态。

单片段参数、输入哈希和输出哈希记录在 `features/manifests/` 的 JSON sidecar 中。
