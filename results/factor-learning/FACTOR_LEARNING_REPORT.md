# Independent Acoustic Cues, Factor Learning, and Dynamical Generalization

## Protocol and scientific distinction

The run uses 5 seeds (42, spaced by 1009), dataset sizes (10, 25, 50, 100, 250, 500), four familiar and two held-out direction/speed pairs per seed, and 32-unit F/G/H models with the previous recurrent transition equations and 10,017 parameters. The feedforward direct control uses 9,904 parameters with 10 feature slots and four hidden units. Optimizer settings match the prior adaptive run (NumPy BPTT/Adam, 0.003, batch 32, global gradient clip 1.0; 60 epochs below 500 examples and 80 at 500). Curriculum and shuffled conditions use identical episode records, initialization seeds, epoch counts, and optimizer updates; only presentation order differs. Combined-only is a separate same-size control.

All train starts are drawn from one half of the central visual positions; test starts use the disjoint other half. The world has 177 pixels and 24 ticks; motion begins at tick 5 and the 18-step test remains unclipped. Factor-focused training assimilates ticks 0–9 so motion is visible during learning. Combined held-out evaluation gives only ticks 0–4; remaining slots are zero padding skipped by recurrent assimilation. Every model then rolls out autonomously without future sensory input.

**Strict protocol:** strict: direction focus at speed 1; speed focus uses speed 1 both directions, speed 2 rightward only, speed 3 leftward only; combined stage uses four familiar pairs. No held-out pair occurs in any stage. Because this prevents balanced directions for speeds 2 and 3, strict speed-focus data has a cue/direction correlation; that is an unavoidable constraint of the strict matrix.

**Factor-pretraining transfer protocol:** transfer: direction focus at speed 1; speed focus balances both directions at all three speeds and thus includes the withheld physical motions; combined stage still excludes them. Results are reported separately and are not evidence for strict compositional generalization.

## Acoustic validation

Five independent cue IDs use unique two-tone sequences, equal duration (two 50 ms chunks), equal nominal amplitude, independent amplitude jitter/noise, and a 50 ms silence between direction and speed cues. Speed cue average frequencies are intentionally nonmonotonic in speed. The ordered raw-audio feature probe reaches **100.0%** accuracy (200 held-out cue presentations); chunk swapping changes the feature vector by 3.049. The same 50 ms audio-chunk interface and `sensory_features` transform are retained. Config A uses the previous frequency/phase instruction; config B uses `[direction, silence, speed]`. Cue duration and acoustic information differ, so A/B is a cue-representation comparison, not a pure semantic-separation test.

The vocabulary CSV is simulator-side metadata. `Observation` receives only visual arrays, raw signed waveform chunks and event times; it contains no cue label, direction or speed.

## h=9 position MAE

Mean ± 95% t interval across seeds, averaging command pairs within seed.

| Model | Strict curriculum familiar | Strict curriculum held out | Transfer curriculum held out |
|---|---:|---:|---:|
| F | 23.77 ± 4.33 (n=5) | 51.55 ± 12.80 (n=5) | 50.97 ± 12.58 (n=5) |
| G | 18.81 ± 9.27 (n=5) | 46.22 ± 19.63 (n=5) | 49.71 ± 13.81 (n=5) |
| H | 16.11 ± 3.67 (n=5) | 46.10 ± 20.19 (n=5) | 43.32 ± 17.77 (n=5) |
| FF_direct | 12.60 ± 3.54 (n=5) | 40.24 ± 10.99 (n=5) | 43.21 ± 6.42 (n=5) |

Training fit at 500 episodes uses each model's sampled training examples; validation uses familiar cues and familiar motion pairs.

| Model | Training MAE px (95% CI) | Validation MAE px (95% CI) | Mean training CPU seconds |
|---|---:|---:|---:|
| F | 4.47 ± 1.63 | 7.17 ± 1.25 | 2.42 |
| G | 4.84 ± 2.32 | 6.68 ± 1.34 | 2.45 |
| H | 2.51 ± 0.74 | 4.87 ± 1.01 | 2.51 |
| FF_direct | 4.92 ± 1.13 | 7.11 ± 1.73 | 0.87 |

## Curriculum versus shuffled

Paired difference is held-out h=9 MAE (shuffled minus curriculum); positive values favor curriculum. SD and seed count are shown; the paired per-seed values are in `factor_curriculum_results.csv`.

| Model | Mean difference px | SD | Seeds |
|---|---:|---:|---:|
| F | -15.39 | 4.17 | 5 |
| G | 2.98 | 5.19 | 5 |
| H | -6.08 | 12.61 | 5 |
| FF_direct | 11.61 | 10.70 | 5 |

## Acoustic cue configuration comparison

| Model | Schedule | Configuration | Held-out h=9 MAE |
|---|---|---|---:|
| F | combined_only | A_legacy | 41.41 ± 4.35 (n=5) |
| F | combined_only | B_independent | 48.39 ± 14.96 (n=5) |
| F | curriculum | A_legacy | 42.03 ± 3.50 (n=5) |
| F | curriculum | B_independent | 51.55 ± 12.80 (n=5) |
| F | shuffled | A_legacy | 41.63 ± 5.81 (n=5) |
| F | shuffled | B_independent | 36.16 ± 13.22 (n=5) |
| G | combined_only | A_legacy | 41.34 ± 2.39 (n=5) |
| G | combined_only | B_independent | 39.23 ± 16.00 (n=5) |
| G | curriculum | A_legacy | 40.92 ± 2.87 (n=5) |
| G | curriculum | B_independent | 46.22 ± 19.63 (n=5) |
| G | shuffled | A_legacy | 39.73 ± 2.12 (n=5) |
| G | shuffled | B_independent | 49.19 ± 19.00 (n=5) |
| H | combined_only | A_legacy | 40.12 ± 2.94 (n=5) |
| H | combined_only | B_independent | 35.37 ± 17.70 (n=5) |
| H | curriculum | A_legacy | 38.80 ± 2.87 (n=5) |
| H | curriculum | B_independent | 46.10 ± 20.19 (n=5) |
| H | shuffled | A_legacy | 40.37 ± 1.47 (n=5) |
| H | shuffled | B_independent | 40.01 ± 18.14 (n=5) |
| FF_direct | combined_only | A_legacy | 39.97 ± 0.38 (n=5) |
| FF_direct | combined_only | B_independent | 47.30 ± 9.34 (n=5) |
| FF_direct | curriculum | A_legacy | 40.65 ± 4.94 (n=5) |
| FF_direct | curriculum | B_independent | 40.24 ± 10.99 (n=5) |
| FF_direct | shuffled | A_legacy | 44.04 ± 1.73 (n=5) |
| FF_direct | shuffled | B_independent | 51.86 ± 3.51 (n=5) |

Paired difference is B minus A in held-out h=9 MAE; positive values mean the independent cue configuration had higher error. Curriculum and shuffled rows use the 500-episode strict arm.

| Model | Schedule | Paired B minus A MAE, px (95% CI) |
|---|---|---:|
| F | combined_only | 6.98 ± 15.69 (n=5) |
| F | curriculum | 9.52 ± 12.57 (n=5) |
| F | shuffled | -5.47 ± 14.16 (n=5) |
| G | combined_only | -2.11 ± 17.93 (n=5) |
| G | curriculum | 5.30 ± 21.01 (n=5) |
| G | shuffled | 9.46 ± 20.15 (n=5) |
| H | combined_only | -4.76 ± 20.14 (n=5) |
| H | curriculum | 7.29 ± 19.82 (n=5) |
| H | shuffled | -0.36 ± 17.55 (n=5) |
| FF_direct | combined_only | 7.33 ± 9.27 (n=5) |
| FF_direct | curriculum | -0.41 ± 12.71 (n=5) |
| FF_direct | shuffled | 7.82 ± 3.83 (n=5) |

The interaction is (B minus A under curriculum) minus (B minus A under shuffled). It is a descriptive factorial contrast; five seeds give broad uncertainty.

| Model | Cue-by-schedule interaction, px (95% CI) |
|---|---:|
| F | 14.99 ± 6.79 (n=5) |
| G | -4.17 ± 8.38 (n=5) |
| H | 7.65 ± 15.28 (n=5) |
| FF_direct | -8.23 ± 15.05 (n=5) |

## Frozen factor probes and counterfactual interventions

Probes are trained post hoc on familiar combined experiences only, including in the transfer protocol. Position is MAE in pixels; direction/speed are accuracy. Probes do not train F–H.

| Model | Configuration | Factor | Held-out probe score |
|---|---|---|---:|
| F | A_legacy | position | 5.265 |
| F | A_legacy | direction | 0.028 |
| F | A_legacy | speed | 0.903 |
| F | B_independent | position | 7.758 |
| F | B_independent | direction | 0.025 |
| F | B_independent | speed | 0.922 |
| G | A_legacy | position | 4.094 |
| G | A_legacy | direction | 0.045 |
| G | A_legacy | speed | 0.878 |
| G | B_independent | position | 7.421 |
| G | B_independent | direction | 0.057 |
| G | B_independent | speed | 0.880 |
| H | A_legacy | position | 4.069 |
| H | A_legacy | direction | 0.057 |
| H | A_legacy | speed | 0.957 |
| H | B_independent | position | 8.464 |
| H | B_independent | direction | 0.150 |
| H | B_independent | speed | 0.963 |
| FF_direct | A_legacy | position | 9.922 |
| FF_direct | A_legacy | direction | 0.210 |
| FF_direct | A_legacy | speed | 0.360 |
| FF_direct | B_independent | position | 12.084 |
| FF_direct | B_independent | direction | 0.455 |
| FF_direct | B_independent | speed | 0.340 |

The counterfactual score is the fraction of held-out trials where changing only the direction or speed waveform changes the h=9 predicted displacement in the expected direction. Other cue chunks and pre-motion vision are paired.

| Model | Changed cue | Response consistency |
|---|---|---:|
| F | direction | 0.753 |
| F | speed | 0.705 |
| G | direction | 0.780 |
| G | speed | 0.692 |
| H | direction | 0.815 |
| H | speed | 0.708 |
| FF_direct | direction | 0.960 |
| FF_direct | speed | 0.542 |

## State initialization, rollout, and compute

`factor_curriculum_results.csv` reports error at each horizon 1–18 and a state-initialization position diagnostic for F/G/H. Horizon 18 is excluded from loss; the direct baseline has no recursive h=18 output. Compare h=1 with later steps to see whether error is present at initialization or accumulates during autonomous evolution. This decomposition is descriptive because the state decoder itself is trained on post-transition positions.

`factor_compute_benchmark.csv` records train wall/CPU time, inference median/p95/p99, traced allocation, parameter counts, state updates and event costs. Traced allocation is not process RSS; short process CPU intervals are clock-resolution limited. Sampled whole-run peak working set was 360,697,856 bytes (344.0 MiB); this is process-level and includes the whole experiment, not an isolated model. H's event trigger evaluates the candidate state update, so skips may not reduce total computation.

Each cell below gives mean MAE (95% t interval) / mean within-one-pixel accuracy across seeds.

| Model | Split | h1 | h3 | h6 | h9 | h12 | h18 |
|---|---|---:|---:|---:|---:|---:|---:|
| F | familiar | 1: 15.88 ± 2.45 / 4.4%; 3: 17.50 ± 2.40 / 3.1%; 6: 21.39 ± 2.96 / 2.5%; 9: 23.77 ± 4.33 / 2.3%; 12: 25.45 ± 5.50 / 2.3%; 18: 29.70 ± 7.56 / 1.2% |
| F | heldout | 1: 20.04 ± 3.81 / 4.2%; 3: 27.83 ± 5.02 / 1.7%; 6: 41.13 ± 10.30 / 1.2%; 9: 51.55 ± 12.80 / 1.2%; 12: 60.25 ± 14.08 / 0.8%; 18: 75.08 ± 16.22 / 0.8% |
| G | familiar | 1: 12.60 ± 2.47 / 5.4%; 3: 12.54 ± 2.72 / 6.2%; 6: 15.88 ± 4.66 / 4.9%; 9: 18.81 ± 9.27 / 3.1%; 12: 20.09 ± 13.90 / 4.8%; 18: 24.42 ± 17.77 / 1.2% |
| G | heldout | 1: 18.08 ± 6.60 / 3.0%; 3: 21.95 ± 7.86 / 3.5%; 6: 33.99 ± 14.36 / 2.0%; 9: 46.22 ± 19.63 / 0.5%; 12: 55.10 ± 23.51 / 0.8%; 18: 66.38 ± 27.95 / 0.8% |
| H | familiar | 1: 12.57 ± 2.42 / 5.3%; 3: 13.75 ± 3.45 / 5.1%; 6: 15.49 ± 4.04 / 4.6%; 9: 16.11 ± 3.67 / 4.1%; 12: 16.13 ± 3.40 / 3.2%; 18: 18.64 ± 2.93 / 3.5% |
| H | heldout | 1: 18.12 ± 5.93 / 5.3%; 3: 24.08 ± 12.05 / 2.5%; 6: 34.73 ± 17.98 / 1.0%; 9: 46.10 ± 20.19 / 1.5%; 12: 56.54 ± 19.58 / 0.5%; 18: 73.72 ± 13.16 / 0.0% |
| FF_direct | familiar | 1: 12.62 ± 3.59 / 6.4%; 3: 12.48 ± 3.50 / 5.4%; 6: 12.50 ± 3.55 / 5.4%; 9: 12.60 ± 3.54 / 5.6%; 12: 12.90 ± 3.75 / 5.3%; 18: n/a |
| FF_direct | heldout | 1: 17.08 ± 4.81 / 3.8%; 3: 22.18 ± 6.58 / 3.2%; 6: 31.19 ± 8.71 / 1.0%; 9: 40.24 ± 10.99 / 0.0%; 12: 49.44 ± 13.19 / 0.5%; 18: n/a |

## Held-out sample efficiency

Strict curriculum h=9 MAE at each training size; each seed is averaged across the two held-out combinations.

| Model | Training episodes | MAE px (95% CI) |
|---|---:|---:|
| F | 10 | 42.37 ± 16.45 (n=5) |
| F | 25 | 39.16 ± 11.13 (n=5) |
| F | 50 | 44.48 ± 8.20 (n=5) |
| F | 100 | 45.40 ± 19.17 (n=5) |
| F | 250 | 53.41 ± 12.58 (n=5) |
| F | 500 | 51.55 ± 12.80 (n=5) |
| G | 10 | 38.90 ± 16.00 (n=5) |
| G | 25 | 40.88 ± 11.07 (n=5) |
| G | 50 | 45.51 ± 13.17 (n=5) |
| G | 100 | 45.59 ± 11.33 (n=5) |
| G | 250 | 49.89 ± 11.58 (n=5) |
| G | 500 | 46.22 ± 19.63 (n=5) |
| H | 10 | 41.89 ± 13.49 (n=5) |
| H | 25 | 37.60 ± 12.36 (n=5) |
| H | 50 | 44.34 ± 10.75 (n=5) |
| H | 100 | 45.94 ± 13.79 (n=5) |
| H | 250 | 41.39 ± 15.46 (n=5) |
| H | 500 | 46.10 ± 20.19 (n=5) |
| FF_direct | 10 | 33.27 ± 6.52 (n=5) |
| FF_direct | 25 | 28.08 ± 4.46 (n=5) |
| FF_direct | 50 | 30.35 ± 9.28 (n=5) |
| FF_direct | 100 | 33.12 ± 8.14 (n=5) |
| FF_direct | 250 | 44.86 ± 6.37 (n=5) |
| FF_direct | 500 | 40.24 ± 10.99 (n=5) |

Mean inference cost for the strict curriculum, independent-cue held-out set at 500 training episodes:

| Model | CPU seconds per test pass | Median latency ms/episode | p95 latency ms/episode | Peak traced bytes/pass |
|---|---:|---:|---:|---:|
| F | 0.0081 | 0.2995 | 0.4032 | 3216536 |
| G | 0.0044 | 0.3146 | 0.4336 | 3216536 |
| H | 0.0056 | 0.3325 | 0.5096 | 3216536 |
| FF_direct | 0.0025 | 0.1060 | 0.1326 | 3216616 |

| Threshold | h=9 MAE | Updates skipped | Candidate generation (s) | Trigger check (s) | State update (s) | Wall seconds/pass |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 46.10 | 0.0% | 0.000392 | 0.000120 | 0.000137 | 0.00360 |
| 0.05 | 46.09 | 0.3% | 0.000436 | 0.000132 | 0.000156 | 0.00338 |
| 0.1 | 46.04 | 1.9% | 0.000447 | 0.000132 | 0.000154 | 0.00350 |
| 0.25 | 42.28 | 28.6% | 0.000399 | 0.000118 | 0.000137 | 0.00341 |
| 0.5 | 34.22 | 64.3% | 0.000391 | 0.000117 | 0.000134 | 0.00349 |
| 1 | 31.30 | 78.6% | 0.000394 | 0.000113 | 0.000134 | 0.00329 |

## Answers and interpretation

1. **Independent cues:** B versus A is inconclusive. Paired held-out h=9 differences have mixed signs and wide intervals; the waveform formats also differ in duration and acoustic information.
2. **Cue learning:** The raw-audio probe decodes all five cues at 100%, but frozen direction probes are near chance for F/G/H (2.5%, 5.7%, 15.0%). Speed probes are stronger (88.0% to 96.3%), so this protocol retains speed more clearly than direction.
3. **Curriculum:** Shuffled-minus-curriculum differences are -15.39 px for F and -6.08 px for H; curriculum favors G by 2.98 px and FF_direct by 11.61 px. There is no general staged-training advantage.
4. **Direction interventions:** Changing only direction yields the expected displacement sign on 75.3% to 96.0% of held-out trials, despite weak direction probes.
5. **Speed interventions:** Changing only speed produces expected magnitude changes on 54.2% to 70.8% of trials; performance is weak and model-dependent.
6. **Assimilation and rollout:** Recurrent h=1 held-out error is already 18 to 20 px, rising to 46 to 52 px at h=9 and 66 to 75 px at h=18. Both the initial state and autonomous evolution contribute, with substantial additional rollout error.
7. **Source of improvements:** The factorial contrast is 14.99 ± 6.79 px for F, with the interval above zero; G, H, and FF_direct intervals include zero. Treat the F interaction as exploratory across four architectures and five seeds. Cue comparisons do not isolate semantics from waveform duration or acoustic information content.
8. **Dynamical versus direct models:** The parameter-matched direct baseline reaches 40.24 px held-out h=9 MAE, below F/G/H (46.10 to 51.55 px), with lower inference latency. This run shows no dynamical-model advantage.
9. **Event triggering:** At threshold 1, H skips 78.6% of updates and mean h=9 MAE falls from 46.10 to 31.30 px; mean wall time falls from 3.60 to 3.29 ms per test pass. The timing trend is nonmonotonic, candidate generation remains necessary, and no timing interval was measured, so this is suggestive rather than a robust efficiency result. Skipping also changes the model's dynamics.
10. **Next change:** No core architecture change is justified yet. First control the direction/speed correlation in strict speed-focused training and improve direction recovery; rerun the held-out matrix before attributing failure to the architecture.

Historical A–D, E3 and prior F–H outputs are preserved. E3's previous 0 px score uses known factorization/kinematics and remains a favorable nonmatched reference. This study does not show that recurrent dynamics learn physical laws or that any internal states imply experience.
