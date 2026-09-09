# gnn-comsol

A graph neural network surrogate for a 2D channel flow solved in COMSOL
Multiphysics.

The COMSOL mesh is read as a graph: each node carries the local state
`(u, v, p)` — the two velocity components and the pressure — and each
edge carries a mesh adjacency with a weight. The networks learn the
one-step time-advance operator

```
X^k  ──►  X^(k+1)
```

so every snapshot of the simulation is one training sample: a graph with
fixed topology and changing node features.

---

## Quick start

```bash
pip install -e .
```

```bash
python scripts/run_experiment.py configs/virtual_node.yaml
```

Everything the run produces lands in `outputs/<name>_<timestamp>/`:
the resolved config, one checkpoint per network, loss curves, the sweep
table if there was one, and `metrics.json`.

PyTorch and PyTorch Geometric with CUDA need their own package index —
install those first, following the instructions on the PyTorch site for
your CUDA version, then `pip install -e .` for the rest.

---

## Full pipeline: COMSOL time-dependent → GNN → GNN-initialized stationary solve

This is the complete loop, from a transient COMSOL simulation to using a
trained GNN as the nonlinear initial guess of a stationary COMSOL solve.
It spans two machines: **everything below runs on the Windows/CUDA work
PC** (COMSOL + MATLAB LiveLink, and the Python environment with `torch`,
`torch_geometric`, `h5py`), never on a machine that only has this
repository checked out. `src/gnn_comsol/Matlab/` holds reviewed copies
of the MATLAB scripts for reference; the versions that actually run live
under the COMSOL LiveLink folder on the work PC — keep them in sync by
hand.

### Stage 0 — One-time COMSOL model setup

The MATLAB scripts below do not build the `.mph` model, they only drive
one that already has the right structure:

- **Study 1** (`std1` / `sol1`): time-dependent Laminar Flow (`spf`),
  already configured to solve and store the transient solution.
- **Study 3** (`std3` / `sol3`): a Stationary study on the same physics,
  used later as the BDF1-equivalent solve. It needs a Volume Force
  feature (`vf1`) on `spf` (its `F` expression is overwritten by the
  MATLAB scripts at every transition) and a Dependent Variables node
  (`v1`) whose initial-guess source the scripts toggle between `sol1`
  and the physics Initial Values.
- **Physics Initial Values** (`init1`) on `spf`, with `u`, `v`, `p`
  driven by three interpolation functions (`int4`, `int5`, `int6`) that
  read `u_gnn.txt` / `v_gnn.txt` / `p_gnn.txt` — the files the MATLAB
  scripts write the GNN prediction to before refreshing them.
- **`dset2`**, a stationary dataset exposing the boundary-distance
  variables `G2`/`G3`/`G4` (wall/inlet/outlet) and their direction
  fields (`wd.Ddirx/y`, `wd2.*`, `wd3.*`), used to build the geometry
  features.

This only has to be built once per geometry template, then reused.

### Stage 1 + 2 — Run the transient simulation and export the dataset

Edit the *User configuration* block at the top of `first_database.m`
(model path, dataset tags, output `.mat` path), then run it in MATLAB
with COMSOL LiveLink connected:

```matlab
first_database
```

It will: load the `.mph`, run Study 1 if `sol1` is not already solved,
evaluate `u, v, p` and the physics-derived features
(`du_dx, du_dy, dv_dx, dv_dy, div_conv`) at every mesh node and
timestep, evaluate the static geometry features (distance/direction to
wall/inlet/outlet), extract the mesh graph (`edge_index`, `edge_weight`
from `w_ij = 1/(1+d_ij/h)`), save everything to the output `.mat`, and
save the `.mph` back to disk with the time-dependent solution now
stored in it. See [The input dataset](#the-input-dataset) below for the
exact keys it writes.

Run it once per geometry you want in the training set.

### Stage 3 — Train the GNN

On the work PC, in the Python environment for this project:

```bash
python scripts/run_experiment.py configs/<your_config>.yaml
```

Point `dataset.path` (one geometry) or `dataset.paths` (several) at the
`.mat`(s) from Stage 2. Decide there whether to turn on
`use_physics_features` and/or `use_geometry_features` on the `pressure`
network (both need `architecture: bsms` and a dataset that actually
contains them — see [Layout](#layout) and the two flags' docstrings in
`data/features.py`).

The run lands in `outputs/<name>_<timestamp>/`: `velocity.pth` and
`pressure.pth` (weights + every normalizer used to build their inputs
and targets + metadata), loss curves, resolved `config.json`,
`metrics.json`. **Note the run directory name** — it is `run_dir` in
Stage 5.

### Stage 4 — GNN inference (usually automatic)

`scripts/evaluate_test.py <run_dir> --dataset <path.mat> --no-animation`
loads a trained run, runs one-step inference on every transition of the
given dataset, and writes the predictions to a fixed path
(`gnn_predictions.mat`, currently hardcoded near the bottom of the
script). **You normally do not call this by hand**: every MATLAB script
in Stage 6 calls it automatically through `prepare_gnn_comsol_pipeline`.
Run it manually only to sanity-check a run's predictions on their own.

### Stage 5 — Point MATLAB at the trained run

All paths used by the MATLAB side live in one place,
`gnn_comsol_config.m`. Update it whenever the model, the dataset, the
Python environment, or the trained run change:

```matlab
cfg.model_file            % the .mph from Stage 2 (sol1 already stored)
cfg.dataset_file          % the .mat from Stage 2, matching that model
cfg.python_exe            % python.exe inside the GNN virtual environment
cfg.python_script         % scripts/evaluate_test.py
cfg.run_dir               % outputs/<name>_<timestamp> from Stage 3
cfg.gnn_predictions_file  % where evaluate_test.py writes its output
cfg.u_gnn_file / v_gnn_file / p_gnn_file   % the COMSOL interpolation files
```

### Stage 6 — Stationary solve with the GNN initial guess

Run one of, in MATLAB with COMSOL LiveLink connected:

```matlab
complete_pipeline_single_step      % one transition (k = 1000 by default)
complete_pipeline_all_timesteps    % a range (k_start:k_end)
```

Both call `prepare_gnn_comsol_pipeline()` (loads the model, runs Stage 4
automatically, checks that the COMSOL and GNN `Δt` arrays line up) and
then `run_gnn_vs_standard_transitions()`, which for every transition
`X_k → X_(k+1)`:

1. sets `dt_step` and rewrites the BDF1-equivalent Volume Force
   `F = ρ(u_k − u)/Δt` (see the docstring at the top of
   `complete_pipeline_single_step.m` for the derivation);
2. solves the stationary problem once with the **standard** initial
   guess `X^(0) = X_k` (from `sol1`);
3. writes the GNN prediction to the interpolation files, refreshes
   them, and solves again with the **GNN** initial guess
   `X^(0) = X̂_(k+1)`;
4. compares both converged solutions to each other and to the true
   `X_(k+1)` from `sol1`, and times both solves.

`complete_pipeline_all_timesteps.m` additionally saves the per-transition
results to `stationary_gnn_comparison.mat`.

---

## The input dataset

`first_database.m` (Stage 2 above) produces the MATLAB v7.3 file (HDF5)
that `scripts/run_experiment.py` reads via `dataset.path`:

| key                     | shape        | meaning                                              |
|-------------------------|--------------|-------------------------------------------------------|
| `X`                     | (3, N, T)    | state per node and timestep, MATLAB order              |
| `edge_index`            | (2, E)       | mesh connectivity, zero-based                          |
| `edge_weight`           | (E,)         | edge weights, `w_ij = 1/(1+d_ij/h)`                     |
| `t`                     | (T,)         | simulation time of each snapshot                        |
| `h`                     | —            | local mesh size                                         |
| `P`                     | (2, N)       | mesh-node coordinates                                    |
| `physics_features`      | (5, N, T)    | optional, per node and timestep — `du_dx, du_dy, dv_dx, dv_dy, div_conv` |
| `geometry_features`     | (6, N) or (N, 6) | optional, **static per node** (not per timestep) — wall/inlet/outlet distance × direction |

`physics_features` and `geometry_features` are optional: a dataset
without them loads and trains fine, as long as no network in the
experiment config sets `use_physics_features` / `use_geometry_features`
to `true`.

---

## Layout

```
configs/                  experiments, as data rather than code
scripts/
  run_experiment.py       Stage 3 - train
  evaluate_test.py        Stage 4 - one-step inference, GNN predictions -> .mat
src/gnn_comsol/
  Matlab/                 reviewed copies of the COMSOL LiveLink scripts
                           (canonical versions live on the work PC, see
                           "Full pipeline" above)
    first_database.m                    Stage 2 - build the dataset
    gnn_comsol_config.m                 Stage 5 - the one place all paths live
    prepare_gnn_comsol_pipeline.m       load model + run Stage 4 + check alignment
    run_gnn_vs_standard_transitions.m   Stage 6 - shared standard-vs-GNN comparison
    complete_pipeline_single_step.m     Stage 6 - one transition
    complete_pipeline_all_timesteps.m   Stage 6 - a range of transitions
    delta_t_statistics.m                exploratory: Δt distribution
    delta_t_pressure_correlation.m      exploratory: Δt vs pressure change
    pressure_statistics.m               exploratory: pressure distribution
    pressure_statistics_boundary.m      exploratory: pressure by mesh region
  data/
    loading.py            reading the .mat, building (input, target) pairs
    splitting.py          train/val/test, temporal by default
    normalization.py      StateNormalizer / PhysicsNormalizer / GeometryNormalizer, the scaling contract
    features.py           time encodings + optional physics/geometry blocks appended to node features
    graphs.py             arrays -> PyTorch Geometric Data objects
  models/
    gcn.py                the plain GCN
    virtual_node.py       GCN + global virtual node
    bsms_pressure.py      U-Net on the graph, used for pressure
    bsms_ops.py           message-passing building blocks (third-party, see below)
  graph/
    bsms.py               bi-stride multi-scale graph hierarchy (third-party, see below)
    clustering.py         mesh coarsening
  train.py                the training loop
  evaluate.py             metrics and one-step inference
  checkpoints.py          saving, with every scaling attached
  config.py               experiment file schema
  plots.py                figures
tests/                    what can be tested without a GPU or the dataset
outputs/                  run directories (git-ignored)
```

---

## Experiments

Each config is one step in the investigation, and they are best read in
order.

| config | idea | outcome |
|---|---|---|
| `monolithic.yaml` | one network for `u, v, p` | velocity learned well, **pressure not** — error 3–4 orders of magnitude larger. This is why everything else exists. |
| `separate.yaml` | one network per group, pressure gets Fourier time features and 25 layers | depth alone is an expensive way to reach non-locality |
| `virtual_node.yaml` | a virtual node connected to every mesh node | same reach with 8 layers instead of 25 |
| `sweep_example.yaml` | how to sweep hyperparameters | — |

### Why pressure is the hard part

In an incompressible flow the pressure satisfies a Poisson equation: it
is **non-local**, so a disturbance anywhere is felt everywhere at once.
A GCN with `L` layers only sees a neighbourhood of `L` hops. Every
experiment after the first is an attempt to widen that receptive field:
by depth (experiment 2), by a global hub node (experiment 3), or by a
coarse graph (the multiscale model, not yet wired up).

---

## Four invariants worth knowing about

**1. The scaling is a contract, and it lives in one place.**

Training minimises `‖ net(x) − T(y) ‖²` and inference computes
`T⁻¹(net(x))`. These must be inverses. They used to live in different
files and drifted apart: the pressure network was trained on physical
units while inference de-standardized its output anyway, so predictions
were wrong by a factor of the pressure standard deviation — about three
orders of magnitude.

`StateNormalizer` is now the only definition of `T`, it is **saved
inside the checkpoint**, and `load_checkpoint` refuses a checkpoint that
does not carry one. A wrong scaling produces plausible numbers rather
than an error, which is the worst way to fail.

**2. The split is temporal, not random.**

Consecutive snapshots of one simulation are nearly identical. With a
shuffled split the nearest training neighbour of a test sample was **one
snapshot away**, which is why validation loss used to read `3.9e-4`
against a test loss of `4.3e-1`.

The default is contiguous blocks in time, with a one-sample gap between
them: sample `i` is the pair `X[i+1] → X[i+2]`, so without the gap one
snapshot would be both the last training target and the first validation
input.

`mode: random` still exists, to reproduce the old numbers for
comparison. `mode: group` keeps whole simulations together and is the
right protocol once there is more than one simulation — which there is
not yet.

Because the blocks cover different phases of the simulation they are not
identically distributed, so every run prints per-block statistics. Read
them before reading the errors: if the test block spans a range of
pressure the training block never contains, no model can fit it.

**3. `Δt` is the step being predicted.**

Sample `i` is the transition `X[skip+i] → X[skip+i+1]`, and
`delta_t[i]` is its duration, `t[skip+i+1] − t[skip+i]`. The original
code passed the *previous* step instead, which is a different number
whenever the solver used a variable time step — and adaptive solvers
normally do.

Every run prints the coefficient of variation of `Δt`. If it is ~0 the
solver used a constant step and the time feature carries no
information at all.

**Dropping the start of the simulation.** `dataset.skip_initial`
(default 1) drops the first snapshots. The initial condition is not a
state of the flow: it is usually artificial and does not satisfy the
governing equations, and the first step of an adaptive solver is
atypically small. The cost is one sample per snapshot dropped.

Whether 1 is enough depends on how long the solver takes to relax the
initial condition — `plot_pressure_statistics` shows it. Worth checking,
because with a temporal split any startup transient sits entirely in the
training block.

**4. Absolute state vs. increment is a per-network config choice, not
a code change.**

Two consecutive states are nearly identical, so a network predicting
the absolute next state `X^(k+1)` spends most of its capacity
reproducing the identity map. Set `predict_delta: true` on a network to
have it predict the increment `ΔX = X^(k+1) − X^k` instead — it then
only has to learn the dynamics.

This works on any architecture, for any network (`velocity`,
`pressure`, or a single `state` network), independently of every other
per-network flag:

```yaml
networks:
  velocity:
    predicts: velocity
    predict_delta: true
    ...
```

The increment has a completely different scale from the absolute
field — near zero, much smaller spread — so it is fitted its own
`StateNormalizer` (`delta_normalizer` in the checkpoint) rather than
reusing the state one. **What comes out of training and inference is
still always the absolute field**: `scripts/evaluate_test.py` adds the
predicted physical increment back onto the physical input state before
writing `gnn_predictions.mat`, so nothing downstream — the MATLAB
pipeline included — ever sees a delta. `velocity` and `pressure` can
independently be absolute or delta; the only combination refused at
config-load time is `predict_delta: true` on `velocity` together with
`use_predicted_velocity: true` on another network, because that
predicted-velocity feature currently assumes the velocity network's raw
output is already the absolute next state.

---

## Known open points

- **Fourier features are computed on the normalized `Δt`**, so the
  frequencies have no physical meaning.
- **The multiscale model is not wired up.** `forward()` needs a cluster
  vector and a coarse `edge_index` on top of the graph, so it does not
  match the `net(batch)` call of the training loop, and the cluster
  tensors would need per-graph offsets when batching. The clustering
  itself works: see `gnn_comsol.graph.clustering`.
- **Deep GCN stacks over-smooth.** 25 `GCNConv` layers with no residual
  connections and no normalization make node representations converge to
  each other.
- **One simulation only.** Everything above about generalisation is
  limited by this.

---

## Third-party code

`src/gnn_comsol/models/bsms_ops.py` and `src/gnn_comsol/graph/bsms.py`
are adapted from [Eydcao/BSMS-GNN](https://github.com/Eydcao/BSMS-GNN)
(Apache License 2.0). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact upstream
files, the commit this was verified against, what was changed, the full
license text, and the citation for the original paper:

> Cao, Y., Chai, M., Li, M., & Jiang, C. (2023). *Efficient Learning of
> Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural
> Network*. ICML 2023.

---

## Tests

```bash
python -m pytest
```

The tests cover what can be checked without a GPU or the dataset: the
splitting logic, the scaling round-trip, the config schema and the graph
coarsening. Anything that needs `torch` to actually run a network is not
covered.
