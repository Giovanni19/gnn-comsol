"""
The mesh-static quantities the physics loss needs, one per simulation.

Everything here depends on the mesh alone, not on the snapshot: the WLSQ
stencils, their operators, the cell connectivity and the control-volume
geometry are the same for every timestep of a simulation and for every
epoch of a training run. They are therefore built once and cached, keyed
by (device, dtype).

That caching is not a micro-optimization. The first version of the
physics loss rebuilt a Python list of N small tensors for every graph of
every batch of every epoch, which cost more than the training step it
was attached to.
"""

import torch

from .mesh import build_cell_geometry
from .wlsq import build_wlsq_operators


class PhysicsGeometry:
    """
    WLSQ operators and control volumes of one simulation.

    Parameters
    ----------
    neighbors : list[np.ndarray]
        Zero-based WLSQ stencil of every node.

    G_wlsq : list[np.ndarray]
        Per-node (2, k_i) gradient operators.

    cell_index : np.ndarray, shape (Nc, 3)
        Node indices of each triangular cell.

    pos : np.ndarray, shape (N, 2)
        Node coordinates, from which the cell areas and face normals
        are derived.

    num_nodes : int

    fluid : FluidProperties or None
        Density and viscosity, needed by the momentum residual and not
        by the continuity one. May be attached after construction, once
        the experiment config has been read.

    simulation_id : int, optional
        Only used in error messages.
    """

    def __init__(
        self,
        neighbors,
        G_wlsq,
        cell_index,
        pos,
        num_nodes,
        fluid=None,
        simulation_id=None,
    ):

        self.neighbors = neighbors
        self.G_wlsq = G_wlsq
        self.raw_cell_index = cell_index
        self.pos = pos
        self.num_nodes = num_nodes
        self.fluid = fluid
        self.simulation_id = simulation_id

        self._operators = {}
        self._cells = {}

    @classmethod
    def from_simulation(cls, simulation, fluid=None):
        """
        Build from a RawDataset, or return None if it carries no WLSQ
        data: the older .mat files predate the WLSQ export, and ordinary
        supervised training must still run on them.
        """

        has_wlsq = (
            simulation.neighbors is not None
            and simulation.G_wlsq is not None
            and simulation.cell_index is not None
        )

        if not has_wlsq:
            return None

        return cls(
            neighbors=simulation.neighbors,
            G_wlsq=simulation.G_wlsq,
            cell_index=simulation.cell_index,
            pos=simulation.pos,
            num_nodes=simulation.num_nodes,
            fluid=fluid,
            simulation_id=simulation.simulation_id,
        )

    def operators(self, device=None, dtype=torch.float32):
        """The WLSQ operators on `device`, built on first use."""

        key = (str(device), dtype)

        if key not in self._operators:

            self._operators[key] = build_wlsq_operators(
                self.neighbors,
                self.G_wlsq,
                num_nodes=self.num_nodes,
                device=device,
                dtype=dtype,
            )

        return self._operators[key]

    def cells(self, device=None, dtype=torch.float32):
        """The control volumes on `device`, built on first use."""

        key = (str(device), dtype)

        if key not in self._cells:

            self._cells[key] = build_cell_geometry(
                self.pos,
                self.raw_cell_index,
                device=device,
                dtype=dtype,
            )

        return self._cells[key]

    def require_fluid(self):
        """
        The fluid properties, or a refusal that says how to supply them.
        """

        if self.fluid is None:
            raise ValueError(
                f"Simulation {self.simulation_id} has no density and "
                "viscosity, so the momentum residual cannot be "
                "computed for it. Either regenerate its .mat with a "
                "first_database.m that exports rho and mu, or give "
                "them in the experiment file:\n"
                "\n"
                "    fluid:\n"
                "      rho: 1.0\n"
                "      mu: 0.005\n"
                "\n"
                "or, per simulation:\n"
                "\n"
                "    fluid:\n"
                f"      {self.simulation_id}: "
                "{rho: 1.0, mu: 0.005}"
            )

        return self.fluid

    @property
    def num_cells(self):
        return self.raw_cell_index.shape[0]

    def __repr__(self):

        return (
            f"PhysicsGeometry(simulation_id={self.simulation_id}, "
            f"num_nodes={self.num_nodes}, "
            f"num_cells={self.num_cells}, "
            f"fluid={self.fluid})"
        )
