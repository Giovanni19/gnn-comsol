# Reference code

This project incorporates source code from third-party projects. This
file lists what was used, from where  and what was
changed. 

---

## BSMS-GNN

**Source**: https://github.com/Eydcao/BSMS-GNN
**License**: Apache License, Version 2.0 (full text below)
**Verified against**: commit `accc09213cf94cdd56c416d4a047e4f6cda3b0a3`
of `main` (2024-08-16). The exact commit the code was originally copied
from was not recorded at the time, so this is the closest reproducible
reference — the relevant classes and their docstrings are unchanged
between that commit and the code below.

**Used in this repository**:

| This file | Adapted from upstream |
|---|---|
| [`src/gnn_comsol/models/bsms_ops.py`](src/gnn_comsol/models/bsms_ops.py) | `src/ops/BSMS.py`, `src/ops/basic.py` |
| [`src/gnn_comsol/graph/bsms.py`](src/gnn_comsol/graph/bsms.py) | `src/graph_wrappers/bsms_graph_wrapper.py`, `src/graph_wrappers/graph_wrapper.py` |

**Notable changes made relative to upstream** (see the header comment of
each file above for the full, current list):

- The two source files behind each of our files were merged into one.
- `bsms_graph_wrapper.py`'s two-hop adjacency product in
  `bstride_selection()` used Intel MKL via `sparse_dot_mkl`; this was
  replaced with plain `scipy.sparse` matrix multiplication to drop a
  platform-specific dependency. No change to the pooling algorithm.
- The `BSGMP` constructor argument `hidden_layer` was renamed to
  `hidden_layers`, to match this project's naming convention.
- `bsms_ops.py`/`graph/bsms.py` do not depend on the rest of the
  upstream `BSMS-GNN` package (its `datasets/`, `trainer/`, `utils/`,
  training loop, etc.) — only the message-passing/pooling core is used
  here, wired into this project's own training loop
  ([`train.py`](src/gnn_comsol/train.py)) and data pipeline.

**Citation**. If you use this code, cite the original paper:

```bibtex
@inproceedings{cao2023efficient,
  title     = {Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network},
  author    = {Cao, Yadi and Chai, Menglei and Li, Minchen and Jiang, Chenfanfu},
  booktitle = {International Conference on Machine Learning},
  year      = {2023},
  url       = {https://openreview.net/forum?id=2Mbo7IEtZW}
}
```
