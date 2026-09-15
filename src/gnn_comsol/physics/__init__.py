"""
Physics-informed losses: PDE residuals evaluated on the mesh.

The public surface is small on purpose - the training loop should only
ever need a geometry object, a fluid, and a residual.
"""

from .fluid import FluidProperties
from .geometry import PhysicsGeometry
from .mesh import (
    CellGeometry,
    build_cell_geometry,
    divergence_over_faces,
    find_boundary_nodes,
    interpolate_to_cell,
)
from .navier_stokes import (
    continuity_loss,
    continuity_residual,
    momentum_loss,
    momentum_residual,
)
from .wlsq import (
    WLSQOperators,
    build_wlsq_operators,
    node_gradient_to_cell,
    wlsq_gradient,
    wlsq_gradient_reference,
)

__all__ = [
    "CellGeometry",
    "FluidProperties",
    "PhysicsGeometry",
    "WLSQOperators",
    "build_cell_geometry",
    "build_wlsq_operators",
    "continuity_loss",
    "continuity_residual",
    "divergence_over_faces",
    "find_boundary_nodes",
    "interpolate_to_cell",
    "momentum_loss",
    "momentum_residual",
    "node_gradient_to_cell",
    "wlsq_gradient",
    "wlsq_gradient_reference",
]
