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

## The input dataset

`scripts/run_experiment.py` expects the path in `dataset.path` of the
config to point at a MATLAB v7.3 file (HDF5) containing:

| key           | shape     | meaning                                  |
|---------------|-----------|------------------------------------------|
| `X`           | (3, N, T) | state per node and timestep, MATLAB order |
| `edge_index`  | (2, E)    | mesh connectivity, zero-based             |
| `edge_weight` | (E,)      | edge weights                              |
| `t`           | (T,)      | simulation time of each snapshot          |
| `h`           | —         | local mesh size                           |

**This file is produced outside this repository**, by a COMSOL + MATLAB
LiveLink script that is not versioned here. Until that script is added,
the project cannot be reproduced from scratch — only re-run on an
existing `.mat`. Adding it under `scripts/` is the single most valuable
missing piece.

---

## Layout

```
configs/                  experiments, as data rather than code
scripts/run_experiment.py the only entry point
src/gnn_comsol/
  data/
    loading.py            reading the .mat, building (input, target) pairs
    splitting.py          train/val/test, temporal by default
    normalization.py      StateNormalizer, the scaling contract
    features.py           time encodings appended to node features
    graphs.py             arrays -> PyTorch Geometric Data objects
  models/
    gcn.py                the plain GCN
    virtual_node.py       GCN + global virtual node
    multiscale.py         U-Net on the graph (not yet wired up)
  graph/clustering.py     mesh coarsening for the multiscale model
  train.py                the training loop
  evaluate.py             metrics and one-step inference
  checkpoints.py          saving, with the scaling attached
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

## Three invariants worth knowing about

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

---

## Known open points

- **The targets are absolute states, not increments.** Two consecutive
  states are nearly identical, so the network spends most of its
  capacity reproducing the identity map. Predicting
  `ΔX = X^(k+1) − X^k` and reconstructing `X^(k+1) = X^k + net(x)` would
  make it learn only the dynamics. This is the natural next step.
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

## Tests

```bash
python -m pytest
```

The tests cover what can be checked without a GPU or the dataset: the
splitting logic, the scaling round-trip, the config schema and the graph
coarsening. Anything that needs `torch` to actually run a network is not
covered.
