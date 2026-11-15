# Multimodal Generalization and Efficiency Experiment

## Protocol

- Seeds: 5, beginning at 42 and spaced by 1009.
- Training direction/speed pairs: all pairs except ((-1, 2), (1, 3)).
- Training sizes: (10, 25, 50, 100, 250, 500); validation examples are sampled only from permitted training pairs.
- Horizons: (3, 6, 9, 12) movement intervals. Motion begins at frame 5; target frame is `motion_start + horizon`.
- The field is sized to avoid edge clipping for every tested horizon and speed.
- Audio cue relationship: carrier frequency is `120 + 180 × speed` Hz; direction is encoded by phase `±π/2`, with the sign-to-direction assignment randomized per seed. These are explicit factor cues; the test is compositional use of known factors, not discovery of an unknown physics law.
- Candidate ridge alpha is selected on validation combinations only. Feature normalization is fitted on training examples only. Held-out combinations do not enter training, normalization, or alpha selection.
- A–D processor equations remain fixed. E1 is a direct linear ridge baseline over the same fixed temporal sensory history. E2 is a privileged analytic oracle using ground-truth initial position, direction, and speed; it is not a fair competitor.

## Generalization at nine motion intervals

Mean absolute position error (pixels), averaged across seeds:

| Model | Observed combinations | Held-out combinations | Generalization gap (held-out − observed) |
|---|---:|---:|---:|
| A | 3.907 | 32.150 | 28.243 |
| B | 2.711 | 24.441 | 21.729 |
| C | 2.618 | 25.860 | 23.241 |
| D | 2.633 | 25.032 | 22.399 |
| E1 | 61.757 | 85.004 | 23.247 |
| E2_oracle | 0.000 | 0.000 | 0.000 |

The machine-readable CSV includes modality ablations, each horizon, individual held-out pairs, within-one-pixel accuracy, exact-position accuracy, seed standard deviations, and 95% t-intervals.

### A-D modality ablations at nine intervals

Mean absolute position error in pixels, averaged over seeds:

| Model | Vision-only, observed | Audio-only, observed | Combined, observed | Combined, held-out |
|---|---:|---:|---:|---:|
| A | 16.190 | 11.019 | 3.907 | 32.150 |
| B | 16.230 | 10.979 | 2.711 | 24.441 |
| C | 16.129 | 10.969 | 2.618 | 25.860 |
| D | 16.163 | 10.979 | 2.633 | 25.032 |

Combined input sharply improves seen-pair prediction because the initial visual position and audio command provide complementary information. Held-out recombination remains poor for all four processors; that is a failed compositional-generalization result under this readout and feature design, not evidence for reusable dynamics.

## Sample efficiency

Held-out combination MAE by independent training-set size; the threshold column is the first size with mean MAE ≤ 1 pixel (or not reached):

| Model | Episodes to ≤1 px MAE | MAE by training episodes |
|---|---:|---|
| A | not reached | 10: 31.63, 25: 32.58, 50: 32.52, 100: 32.95, 250: 33.27, 500: 32.15 |
| B | not reached | 10: 25.10, 25: 25.18, 50: 24.67, 100: 25.54, 250: 24.62, 500: 24.44 |
| C | not reached | 10: 26.99, 25: 26.57, 50: 25.69, 100: 25.93, 250: 26.59, 500: 25.86 |
| D | not reached | 10: 25.54, 25: 25.00, 50: 24.80, 100: 25.03, 250: 25.33, 500: 25.03 |
| E1 | not reached | 10: 70.98, 25: 72.25, 50: 72.71, 100: 73.38, 250: 86.84, 500: 85.00 |

Each size is fit from scratch using a prefix of a shared, balanced training pool; no fitted parameters or normalization are carried over from larger sizes. The test set is identical across models and sizes within a seed.

## Compute results

Repeated inference uses one warm-up trial followed by 5 timing passes. `compute_benchmark.csv` separates feature/preprocessing, internal processor, readout, and end-to-end timings. It records median/p95/p99 latency, process RSS, traced allocation, parameters, updates, and approximate operation counts. Environment thread variables were: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

| Model | Median episode latency (ms) | p95 episode latency (ms) | CPU seconds over 240 timed predictions | Active updates over timed predictions |
|---|---:|---:|---:|---:|
| A | 0.774 | 0.913 | 0.2031 | 1200 |
| B | 0.782 | 0.896 | 0.2031 | 1200 |
| C | 0.885 | 0.946 | 0.2344 | 2400 |
| D | 0.852 | 0.969 | 0.2188 | 1200 |

RSS is the process peak working set, a cumulative high-water mark, and cannot be attributed to one model from these sequential measurements. Per-run `tracemalloc` peaks are also recorded, but omit native NumPy/BLAS allocations.

- Event-driven D inference CPU time: 0.2188s; recurrent B: 0.2031s, over the same timed trials. These local measurements include Python instrumentation and should not be treated as portable speed rankings.
- B vs C compositional errors: observed 2.711 vs 2.618 pixels and held-out 24.441 vs 25.860 pixels. Statistical reliability should be judged from the seed-level confidence intervals in the CSV; this report does not infer a mechanism from low error.

## Answers and limits

1. **Does recurrence improve over feedforward history?** Compare A and B in the observed/held-out columns and learning curves. Any difference is task-, horizon-, and seed-dependent; the report does not treat the fixed random recurrent state as learning.
2. **Do continuous-time dynamics improve over ordinary recurrence?** Compare B with C at identical inputs and readout selection. C's Euler substeps add work; error differences alone do not establish a benefit.
3. **Does event-driven computation reduce actual CPU time?** Compare D's measured CPU and end-to-end latency with B/C. Update counts and FLOP estimates are supporting diagnostics, not timing substitutes.
4. **Do architectures need fewer examples?** The first ≤1px sample-size threshold and held-out learning curves provide the observed answer; `not reached` means the configured sizes did not meet it.
5. **Are differences reliable?** Five seeds and t-based 95% intervals are reported. Overlapping intervals are not evidence of a reliable separation; no multiple-comparison hypothesis test is claimed.
6. **Could simple relationships explain the results?** Yes. Frequency directly encodes speed, phase encodes direction, and the target is analytically `initial + direction × speed × horizon`. E1 and privileged E2 expose how much performance follows from a simple linear readout or the known kinematics. This benchmark tests composition of sensory factors, not discovery of governing equations.

Plots are descriptive views of these tables. All outputs are generated from the CSV records by the benchmark runner.
