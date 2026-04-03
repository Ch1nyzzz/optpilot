# Experiment Results

## Completed Runs

| Run ID | Target | Baseline Test Acc | Final Test Acc | Delta | Initial Score | Best Score | Evolution Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_gaia_openevolve_native_gpt-oss-120b_20260403_013534_blind` | `linear_gaia` | 11.11% | 14.81% | +3.70 pts | 0.4300 | 0.4300 | 15.8 min |
| `linear_livecodebench_openevolve_native_gpt-oss-120b_20260403_013534_blind` | `linear_livecodebench` | 0.00% | 8.00% | +8.00 pts | 0.2600 | 0.6100 | 16.7 min |
| `linear_math_openevolve_native_gpt-oss-120b_20260403_013534_blind` | `linear_math` | 55.67% | 57.33% | +1.66 pts | 0.5800 | 0.5800 | 18.1 min |
| `linear_math_openevolve_native_gpt-oss-120b_20260403_030957_blind` | `linear_math` | 56.33% | 56.33% | +0.00 pts | 0.0000 | 0.0000 | 7.5 min |
| `linear_math_openevolve_native_gpt-oss-120b_20260403_031657_blind` | `linear_math` | 56.33% | 58.33% | +2.00 pts | 0.0000 | 0.0000 | 5.1 min |
| `linear_math_openevolve_native_gpt-oss-120b_20260403_055721_blind` | `linear_math` | 59.00% | 59.67% | +0.67 pts | 0.5450 | 0.6900 | 107.5 min |
| `linear_swebench_openevolve_native_gpt-oss-120b_20260403_013534_blind` | `linear_swebench` | 8.20% | 9.06% | +0.86 pts | 0.2050 | 0.3850 | 35.5 min |
| `linear_swebench_openevolve_native_gpt-oss-120b_20260403_055721_blind` | `linear_swebench` | 6.58% | 7.20% | +0.62 pts | 0.2800 | 0.3900 | 181.9 min |

## Best Result Per Target

| Target | Best Run | Final Test Acc | Best Score |
| --- | --- | ---: | ---: |
| `linear_gaia` | `linear_gaia_openevolve_native_gpt-oss-120b_20260403_013534_blind` | 14.81% | 0.4300 |
| `linear_livecodebench` | `linear_livecodebench_openevolve_native_gpt-oss-120b_20260403_013534_blind` | 8.00% | 0.6100 |
| `linear_math` | `linear_math_openevolve_native_gpt-oss-120b_20260403_055721_blind` | 59.67% | 0.6900 |
| `linear_swebench` | `linear_swebench_openevolve_native_gpt-oss-120b_20260403_013534_blind` | 9.06% | 0.3850 |

## Benchmark Breakdown

### `linear_math_openevolve_native_gpt-oss-120b_20260403_055721_blind`

| Benchmark | Baseline | Final |
| --- | ---: | ---: |
| `AIME2024` | 82.22% | 77.78% |
| `AIME2025` | 57.78% | 62.22% |
| `MMLU` | 87.62% | 87.62% |
| `OlympiadBench` | 20.95% | 22.86% |

### `linear_swebench_openevolve_native_gpt-oss-120b_20260403_055721_blind`

| Benchmark | Baseline | Final |
| --- | ---: | ---: |
| `SWE-bench-Lite` | 6.58% | 7.20% |

### `linear_livecodebench_openevolve_native_gpt-oss-120b_20260403_013534_blind`

| Benchmark | Baseline | Final |
| --- | ---: | ---: |
| `LiveCodeBench` | 0.00% | 8.00% |

### `linear_gaia_openevolve_native_gpt-oss-120b_20260403_013534_blind`

| Benchmark | Baseline | Final |
| --- | ---: | ---: |
| `GAIA` | 11.11% | 14.81% |

## Interrupted Or Artifacts-Only Runs

| Run | Status |
| --- | --- |
| `linear_livecodebench_openevolve_native_gpt-oss-120b_20260403_012229_blind_artifacts` | artifacts only |
| `linear_math_openevolve_native_gpt-oss-120b_20260403_015110_blind_artifacts` | artifacts only |
| `linear_swebench_openevolve_native_gpt-oss-120b_20260403_012228_blind_artifacts` | artifacts only |
| `linear_swebench_openevolve_native_gpt-oss-120b_20260403_015110_blind_artifacts` | artifacts only |
| `linear_swebench_openevolve_native_gpt-oss-120b_20260403_030957_blind_artifacts` | artifacts only |
| `linear_swebench_openevolve_native_gpt-oss-120b_20260403_031657_blind_artifacts` | artifacts only |
| `star_livecodebench_openevolve_native_gpt-oss-120b_20260403_012228_blind_artifacts` | artifacts only |
| `star_swebench_openevolve_native_gpt-oss-120b_20260403_012228_blind_artifacts` | artifacts only |

## Result Files

- `/data/home/yuhan/optpilot/results/linear_gaia_openevolve_native_gpt-oss-120b_20260403_013534_blind.json`
- `/data/home/yuhan/optpilot/results/linear_livecodebench_openevolve_native_gpt-oss-120b_20260403_013534_blind.json`
- `/data/home/yuhan/optpilot/results/linear_math_openevolve_native_gpt-oss-120b_20260403_013534_blind.json`
- `/data/home/yuhan/optpilot/results/linear_math_openevolve_native_gpt-oss-120b_20260403_030957_blind.json`
- `/data/home/yuhan/optpilot/results/linear_math_openevolve_native_gpt-oss-120b_20260403_031657_blind.json`
- `/data/home/yuhan/optpilot/results/linear_math_openevolve_native_gpt-oss-120b_20260403_055721_blind.json`
- `/data/home/yuhan/optpilot/results/linear_swebench_openevolve_native_gpt-oss-120b_20260403_013534_blind.json`
- `/data/home/yuhan/optpilot/results/linear_swebench_openevolve_native_gpt-oss-120b_20260403_055721_blind.json`
