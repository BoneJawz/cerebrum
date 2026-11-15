"""Measured-results report for the adaptive dynamics experiment."""
import numpy as np


def format_report(results, sample_rows, probe_rows, compute_rows, config,
                  recurrent_parameters, direct_parameters, cluster_rows=(), timestep_rows=()):
    max_size=max(config.train_sizes)
    seeds=sorted({r["seed"] for r in results})
    models=sorted({r["model"] for r in results
                   if r["split"] in ("observed","heldout") and r["train_size"]==max_size})

    def ci(values):
        vals=np.asarray(values,dtype=float); vals=vals[np.isfinite(vals)]
        if not len(vals): return "n/a"
        avg=float(vals.mean()); sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.
        critical={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571}.get(len(vals),1.96)
        return f"{avg:.2f} ± {critical*sd/np.sqrt(len(vals)):.2f} (SD {sd:.2f}, n={len(vals)})"

    def per_seed(model, split, key, size=max_size):
        vals=[]
        for seed in seeds:
            rows=[r[key] for r in results if r["model"]==model and r["split"]==split
                  and r["seed"]==seed and r["train_size"]==size
                  and r.get("modality")=="combined" and np.isfinite(r[key])]
            if rows: vals.append(float(np.mean(rows)))
        return vals

    summary=[f"| {model} | {ci(per_seed(model,'observed','mae_h9'))} | {ci(per_seed(model,'heldout','mae_h9'))} |"
             for model in models]
    horizon=[f"| {model} | " + " | ".join(ci(per_seed(model,'heldout',f'mae_h{h}'))
              for h in (3,6,9,12,18)) + " |" for model in models]
    sample=[]
    for model in models:
        cells=[]
        for size in config.train_sizes:
            vals=[]
            for seed in seeds:
                rows=[r["mae_h9"] for r in results if r["model"]==model and r["split"]=="heldout"
                      and r["seed"]==seed and r["train_size"]==size and r.get("modality")=="combined"]
                if rows: vals.append(np.mean(rows))
            cells.append(f"{np.mean(vals):.2f}" if vals else "n/a")
        sample.append(f"| {model} | " + " | ".join(cells) + " |")
    balanced=[]
    for model in models:
        vals=[]
        for seed in seeds:
            rows=[r["mae_h9"] for r in results if r["model"]==model and r["split"]=="balanced_control" and r["seed"]==seed]
            if rows: vals.append(np.mean(rows))
        balanced.append(f"| {model} | {ci(vals)} |")
    paired=[]
    for other in ("G","H","FF_direct"):
        diffs=[]
        for seed in seeds:
            left=[r["mae_h9"] for r in results if r["model"]=="F" and r["split"]=="heldout" and r["seed"]==seed and r["train_size"]==max_size and r.get("modality")=="combined"]
            right=[r["mae_h9"] for r in results if r["model"]==other and r["split"]=="heldout" and r["seed"]==seed and r["train_size"]==max_size and r.get("modality")=="combined"]
            if left and right: diffs.append(float(np.mean(left)-np.mean(right)))
        paired.append(f"| F minus {other} | {ci(diffs)} |")
    probes=[]
    for model in models:
        for split in ("observed","heldout"):
            metrics=[]
            for label in ("position","direction","speed"):
                vals=[r["value"] for r in probe_rows if r["model"]==model and r["split"]==split and r["probe"]==label]
                metrics.append(f"{np.mean(vals):.3f}" if vals else "n/a")
            probes.append(f"| {model} | {split} | {metrics[0]} | {metrics[1]} | {metrics[2]} |")
    clusters=[]
    for model in models:
        for split in ("observed","heldout"):
            rows=[r for r in cluster_rows if r["model"]==model and r["split"]==split]
            if rows:
                clusters.append(f"| {model} | {split} | {np.mean([r['mean_within_distance'] for r in rows]):.3f} | "
                    f"{np.mean([r['mean_nearest_between_distance'] for r in rows]):.3f} | "
                    f"{np.mean([r['mean_silhouette'] for r in rows]):.3f} |")
    events=[]
    for threshold in (0.0,.05,.1,.25,.5,1.0):
        rows=[r for r in compute_rows if r.get("threshold")==threshold]
        if rows:
            events.append(f"| {threshold:g} | {np.mean([r['mae_h9'] for r in rows]):.2f} | "
                          f"{100*np.mean([r['skip_fraction'] for r in rows]):.1f}% | "
                          f"{np.mean([r['median_latency_ms'] for r in rows]):.4f} | "
                          f"{np.mean([r['cpu_seconds'] for r in rows]):.5f} |")
    timestep_text="No timestep refinement rows were supplied."
    if timestep_rows:
        timestep_text=(f"At dt=0.025 s (two Euler substeps per 0.05 s interval), h=9 MAE was "
            f"{np.mean([r['fine_mae_h9'] for r in timestep_rows]):.2f} px versus "
            f"{np.mean([r['coarse_mae_h9'] for r in timestep_rows]):.2f} px at dt=0.05 s; "
            f"mean absolute prediction change was {np.mean([r['mean_abs_prediction_change_h9'] for r in timestep_rows]):.2f} px.")
    compute=[]
    for model in models:
        rr=[r for r in compute_rows if r["model"]==model and r["split"]=="heldout" and r.get("threshold","")==""]
        tr=[r for r in sample_rows if r["model"]==model and r["modality"]=="combined" and r["train_size"]==max_size]
        if rr and tr:
            compute.append(f"| {model} | {np.mean([r['train_seconds'] for r in tr]):.3f} | "
                f"{np.mean([r['train_cpu_seconds'] for r in tr]):.3f} | "
                f"{np.mean([r['median_latency_ms'] for r in rr]):.4f} | "
                f"{np.mean([r['p95_latency_ms'] for r in rr]):.4f} | "
                f"{np.mean([r['p99_latency_ms'] for r in rr]):.4f} | "
                f"{np.mean([r['cpu_seconds'] for r in rr]):.5f} | "
                f"{np.mean([r['peak_tracemalloc_bytes'] for r in rr])/1024:.1f} |")
    size_header="| Model | " + " | ".join(map(str,config.train_sizes)) + " |\n|---|"+"---:|"*len(config.train_sizes)
    return f"""# Trainable Dynamical States and Recursive World-Model Learning

## Protocol

This run uses {config.seeds} seeds beginning at {config.seed}, spaced by 1009, and preserves held-out pairs `(-1, 2)` and `(1, 3)`. Training, validation and test trajectories are independent. The visual field is expanded to permit h=18 evaluation, so its pixel count and initial-position range differ from the earlier 129-pixel experiment. Models receive combined sensory input through tick 4.

F/G/H use 32 hidden variables and {recurrent_parameters:,} trainable parameters. The direct MLP has {direct_parameters:,}. Each episode begins at zero state. F uses learned tanh recurrence; G uses explicit Euler with dt=0.05 s and tau=0.1 s; H shares G's continuous training dynamics and uses fixed hard thresholds only during evaluation. H's threshold is not learned. NumPy reverse-mode BPTT with Adam (0.003), batches of 32, gradient clipping at 1.0, and 60 epochs (80 at 500 examples) trains position targets at intervals 1–12 with a shared scalar decoder. Normalization uses training data only. No direction/speed labels or motion equation enter model inputs or targets, and prediction receives no future observations. h=18 is excluded from training loss.

The direct MLP maps the flattened five-tick sequence to one output per trained horizon; it has no autonomous transition and does not report h=18. Historical A–D and E1–E3 results remain unchanged in `results/diagnostics/DIAGNOSTIC_REPORT.md`. E3 previously scored 0 px using known factors and kinematics, making it a favorable nonmatched reference.

## Nine-step MAE on original compositional split

Mean and 95% t interval across seeds, after averaging each seed over command pairs.

| Model | Observed | Held out |
|---|---:|---:|
{chr(10).join(summary)}

## Held-out MAE by rollout horizon

| Model | h=3 | h=6 | h=9 | h=12 | h=18 extrapolation |
|---|---:|---:|---:|---:|---:|
{chr(10).join(horizon)}

## Sample efficiency

{size_header}
{chr(10).join(sample)}

## Balanced-distribution control

This secondary control trains on all six direction/speed pairs and tests independent trajectories from all pairs; it does not replace the original holdout.

| Model | h=9 MAE, mean ± 95% CI |
|---|---:|
{chr(10).join(balanced)}

## Paired model differences

Differences are held-out h=9 MAE, paired by seed and averaged across held-out command pairs. Negative values favor F.

| Comparison | Difference, mean ± 95% CI (px) |
|---|---:|
{chr(10).join(paired)}

## Frozen-state probes

Linear probes are fit after model training using observed training combinations. Position reports MAE; direction and speed report accuracy. Probe labels do not feed back into model training.

| Model | Split | Position MAE | Direction accuracy | Speed exact accuracy |
|---|---|---:|---:|---:|
{chr(10).join(probes)}

## Latent command-pair clustering

The descriptive silhouette score compares standardized latent-state distances within a direction/speed pair and to the nearest other pair. It is sensitive to initial-position variation and is not evidence that a latent dimension has a physical meaning.

| Model | Split | Mean within distance | Nearest between distance | Mean silhouette |
|---|---|---:|---:|---:|
{chr(10).join(clusters)}

## H event threshold sweep

Thresholds act on the candidate autonomous state change. Computing that candidate is part of trigger evaluation, so skipped writes need not reduce total compute. CPU process-clock time can be coarse for short runs; latency quantiles and component wall time are also recorded.

The reduced held-out error at high thresholds is consistent with suppressing some trained motion updates on the direction-reversed combinations; it is a prediction change from damping, not evidence that event selection learned a useful computation policy.

| Threshold | h=9 MAE | Skipped opportunities | Median ms/episode | CPU s/pass |
|---:|---:|---:|---:|---:|
{chr(10).join(events) if events else '| No event rows | | | | |'}

## G timestep refinement

{timestep_text} This inference-only sensitivity check uses the same learned parameters and is not a retrained convergence study.

## Training and inference compute

Inference timings compare a common h=3,6,9,12 request; CPU time is per complete held-out test pass and latency is per episode. The h=18 extrapolation is measured separately for accuracy. Peak traced allocation excludes native NumPy/BLAS allocations and is not process RSS.

| Model | Train wall s | Train CPU s | Median ms/episode | p95 ms | p99 ms | Inference CPU s/pass | Peak traced KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(compute)}

## Artifacts and findings

`adaptive_dynamics_results.csv` contains per-seed/per-pair errors and balanced-control results. `adaptive_initial_position_results.csv` stratifies errors by starting position. `adaptive_sample_efficiency.csv` contains train/validation scores by size; `adaptive_learning_curves.csv` contains epoch loss; `adaptive_compute_benchmark.csv` contains latency quantiles, CPU, memory and update counts; `adaptive_latent_probes.csv`, `adaptive_latent_cluster_diagnostics.csv`, `adaptive_timestep_sensitivity.csv` and `adaptive_paired_differences.csv` contain latent, integration and paired-seed diagnostics. Plots cover generalization, learning loss, rollout error and accuracy versus compute. Checkpoints under `checkpoints/` contain weights and normalizers. Reproduce with `python -m sensory_lab.adaptive_cli --outdir results/adaptive-dynamics`.

1. Accuracy and compositional generalization are measured above; historical B was 2.71 px observed and 24.44 px held out at h=9. Paired uncertainty is reported before attributing change to architecture.
2. The original holdout and balanced control separate recombination from the speed/direction training correlation.
3. F/G/H use one shared transition and decoder at h=3, 6, 9 and 12.
4. h=18 measures extrapolation beyond the loss horizon; the direct-head model has no h=18 output.
5. F/G compare discrete and Euler-integrated evolution at matched hidden width; measured accuracy and timing determine whether G helps.
6. H reports skips and costs. Since trigger evaluation computes the candidate transition, skipped assignments alone do not establish efficiency.
7. Probes measure linear recoverability; they do not prove that individual units map to physical factors.
8. E3 is a zero-error prior physics-informed reference with a favorable known factorization.
9. F/G/H share data, objective and width; the direct MLP has similar parameter count but a flattened feedforward encoder and direct-head objective, so it does not isolate the head alone. The balanced control changes the data distribution. No one comparison isolates every cause.
10. The smallest next change is to use probe and balanced-control results to locate information loss in observed encoding or autonomous transition, then alter one component under matched data and parameter budget.

These measurements do not establish biological validity or subjective experience. Negative results are retained.
"""
