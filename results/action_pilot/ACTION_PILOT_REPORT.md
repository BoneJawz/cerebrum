# Neural Dynamics and Action-Based Prediction — Pilot

## Protocol

Two fixed network seeds (42, 43), with 12 training and 8 held-out trajectories per direction and seed. The object occupied one full vertical column of a binary 5×5 array and moved one column per frame from 0→4 or 4→0. Inter-frame durations were independently jittered from 8 to 14 ms. Whole trajectories, not individual frames, were split before fitting. STDP was disabled; the existing 25-input sparse visual projection and 18-cell LIF population were unchanged. Each trial began from a fresh, identically seeded network. The center observation is column 2; next-position prediction is evaluated there.

Visual arrays were the only sensory inputs. Direction and position labels were retained by the experiment controller only for splitting, fitting the readouts, and scoring. The Brian2 network received only binary visual spike events.

## Results

### 1. State at the same visual position

The center visual input is identical in both directions. The raw visual nearest-centroid control therefore scored **50.0%** overall. On held-out whole trajectories, the 18-neuron voltage snapshot at the center scored **100.0%** direction accuracy; concatenating the center snapshot with the preceding two neural snapshots scored **100.0%**. Seed-level accuracies (42, 43) were 42: 1.000, 43: 1.000 for the current neural state and 42: 1.000, 43: 1.000 for neural history.

Across held-out center snapshots, the opposite-direction centroid distance was 2.895 in the feature space of recorded membrane voltages (mV coordinates); mean within-direction distance to each centroid was 1.384. Mean center-frame associative spike counts were 12.00 for right-to-left and 12.00 for left-to-right. These are dynamical response differences in this seeded setup, not evidence that the network understands motion.

### 2. Next-position prediction at the center

| Method | Definition | Held-out MAE (columns) |
|---|---|---:|
| A. Static | Training mean next column conditioned only on current column 2 | 1.000 |
| B. Neural history | Nearest-centroid direction readout from the preceding three neural snapshots, then one-column step | 0.000 |
| C. Action-inspired cost | Select among ±1 using `J(a)=(a−â)²+0.5(|a|−ŝ)²`; â is B's train-fitted history estimate and ŝ is mean absolute training step | 0.000 |

Seed-level MAEs for A/B/C were 1.000/0.000/0.000 (seed 42) and 1.000/0.000/0.000 (seed 43). A predicts the training conditional mean (column 2) because opposite directions share that observation. The fixed action cost selects the same step as B: the learned speed prior is one column/frame, so both legal candidates satisfy it. C adds no independent predictive mechanism; matching B is expected and does not establish an action principle. The objective is only an action-inspired trajectory cost, not classical action minimization.

## Interpretation and limits

- The current image alone is direction-ambiguous by construction. Any above-chance result from voltage/history features indicates information about preceding inputs remains in the measured transient state over these short frame intervals.
- Results are conditional on two fixed projection seeds and this repeated five-column path. Jittered timing gives held-out sequences, but starting location is direction-correlated; this pilot does not show general motion abstraction or generalization to varied starts/speeds.
- The state-history classifier uses whole-trial train/test separation and train-only scaling/centroids. It does not alter neural dynamics or STDP.
- C is deliberately a tiny demonstration. Its speed prior is estimated from training trajectories and its action estimate reuses B, so it cannot independently establish a benefit. A more independent action-based comparison remains untested.
- Neural voltage separation is descriptive, not a semantic interpretation. One-step errors and small held-out counts are illustrative, not a benchmark claim.

## Smallest justified next experiment

Keep this network and learning rule fixed. Add a few independently seeded projection instances and randomly varied start columns (both directions through a shared interior position), preserve whole-trajectory train/test splits, and compare current-image, neural-state, and neural-history readouts. Only if history remains predictive should a separately trained transition model and a preregistered trajectory-consistency penalty be compared on held-out starts. Do not add recurrence based on this pilot.

## Runtime

Completed 32 held-out center evaluations across two seeds, preserving 80 train/test trajectory records. Simulated biological time: 4.411 s across all trajectories. Wall-clock generation, analysis, and plotting: 17.79 s. No parameter search or new dependencies were used.

Artifacts: `motion_trials.csv` contains train/test trajectory rows, full 1 ms sampled voltage traces, spike events, and held-out predictions/costs. `trajectory_comparison.png` plots population voltage traces and held-out errors.
