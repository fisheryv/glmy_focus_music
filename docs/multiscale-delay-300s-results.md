# 300 s multiscale delay-embedding homology

## Design

- Acoustic and rhythm trajectories were delay-embedded at 4, 8, 16, and 32 bars.
- The eight delay coordinates span approximately one candidate period.
- A one-bar block permutation preserves local content while destroying long-range order.
- Only segments with at least 270 s of effective features entered the primary analysis.
- Discovery selected one scale per modality; validation tested only the frozen endpoints.

## Discovery scale scan

| modality | scale | Focus median | Pop median | rank-biserial | one-sided p | primary-family FDR |
|---|---:|---:|---:|---:|---:|---:|
| acoustic | multi | 0.0367 | 0.0246 | 0.118 | 0.2449 | 0.3225 |
| acoustic | 4 | 0.0361 | 0.0235 | 0.163 | 0.1688 | 0.3225 |
| acoustic | 8 | 0.0325 | 0.0419 | -0.087 | 0.7005 | 0.7005 |
| acoustic | 16 | 0.0411 | 0.0222 | 0.299 | 0.03895 | 0.1298 |
| acoustic | 32 | 0.0222 | 0.0079 | 0.059 | 0.3668 | 0.4076 |
| rhythm | multi | 0.0165 | 0.0084 | 0.274 | 0.05276 | 0.1319 |
| rhythm | 4 | 0.0117 | 0.0053 | 0.111 | 0.258 | 0.3225 |
| rhythm | 8 | 0.0282 | 0.0063 | 0.299 | 0.03895 | 0.1298 |
| rhythm | 16 | 0.0248 | 0.0061 | 0.378 | 0.01264 | 0.1264 |
| rhythm | 32 | 0.0163 | 0.0108 | 0.128 | 0.2258 | 0.3225 |

## Frozen validation

| role | modality | scale | comparison | Focus median | comparator median | rank-biserial | one-sided p | FDR |
|---|---|---:|---|---:|---:|---:|---:|---:|
| duration_sensitivity_295s | acoustic | 16 | focus_greater_than_classical | 0.0245 | 0.0153 | 0.305 | 0.006401 | 0.01183 |
| duration_sensitivity_295s | acoustic | 16 | focus_greater_than_pop | 0.0245 | 0.0366 | -0.165 | 0.8258 | 0.8258 |
| duration_sensitivity_295s | rhythm | 16 | focus_greater_than_classical | 0.0200 | 0.0037 | 0.278 | 0.01183 | 0.01183 |
| duration_sensitivity_295s | rhythm | 16 | focus_greater_than_pop | 0.0200 | 0.0203 | -0.042 | 0.5975 | 0.8258 |
| frozen_validation | acoustic | 16 | focus_greater_than_classical | 0.0245 | 0.0148 | 0.316 | 0.004381 | 0.008761 |
| frozen_validation | acoustic | 16 | focus_greater_than_pop | 0.0245 | 0.0399 | -0.117 | 0.7783 | 0.7783 |
| frozen_validation | rhythm | 16 | focus_greater_than_classical | 0.0200 | 0.0044 | 0.271 | 0.01224 | 0.01224 |
| frozen_validation | rhythm | 16 | focus_greater_than_pop | 0.0200 | 0.0145 | 0.094 | 0.2734 | 0.5467 |

## Validation diagnostics

| modality | group | n | observed H1 | block-shuffle H1 | excess | positive tracks |
|---|---|---:|---:|---:|---:|---:|
| acoustic | classical | 60 | 0.1005 | 0.0844 | 0.0148 | 76.7% |
| acoustic | focus | 38 | 0.1122 | 0.0843 | 0.0245 | 86.8% |
| acoustic | pop | 23 | 0.1194 | 0.0837 | 0.0399 | 91.3% |
| rhythm | classical | 60 | 0.0981 | 0.0929 | 0.0044 | 58.3% |
| rhythm | focus | 38 | 0.1138 | 0.0998 | 0.0200 | 68.4% |
| rhythm | pop | 23 | 0.1173 | 0.1062 | 0.0145 | 60.9% |

## Interpretation

0 of 2 frozen Focus-vs-Pop endpoints showed a positive FDR-significant validation effect.
2 of 2 corresponding Focus-vs-Classical contrasts were positive and FDR-significant.
Across the computed validation rows, 72.3% of track-scale scores exceeded their own block-shuffle median.

The discovery preference for 16 bars did not survive independent Focus-vs-Pop validation. The method detects ordered long-range trajectories in both Focus and Pop, while separating them more clearly from Classical.

A positive score means that the observed temporal ordering supports a longer H1 class than the same local material after one-bar blocks are reordered. It does not by itself prove that the listener perceives a repeated motif.

## Frozen endpoints

- acoustic: 16 bars, h1_surrogate_excess
- rhythm: 16 bars, h1_surrogate_excess
