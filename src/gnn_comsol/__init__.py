"""
GNN surrogate for a 2D channel flow solved in COMSOL.

The mesh is read as a graph: nodes carry the local state (u, v, p),
edges carry the mesh adjacency, and the networks learn the one-step
time-advance operator

    X^k  ->  X^(k+1)

Layout
------
    data/         reading, splitting, scaling, feature encodings
    models/       network architectures
    graph/        mesh coarsening for the multiscale model
    train.py      the training loop
    evaluate.py   metrics and one-step inference
    checkpoints.py  saving, with the scaling attached
    config.py     experiment files
    plots.py      figures

Experiments live in configs/ and are run with
scripts/run_experiment.py.
"""

__version__ = "0.1.0"
