"""
The fluid properties the momentum residual needs.

Continuity is free of them - div(u) = 0 whatever the fluid is - but the
momentum equation

    rho (du/dt + (u.grad) u) = -grad p + mu laplacian(u)

is not, and neither rho nor mu can be recovered from the state: a
velocity field is consistent with infinitely many fluids at infinitely
many pressures. They have to be told.

They can come from two places, in this order:

1. the .mat, if `first_database.m` managed to read them out of the
   COMSOL model. This is the one that cannot drift: it is whatever the
   solver actually used;

2. the experiment YAML, per simulation or as one pair for all of them.

The second exists because the first depends on COMSOL variable names
that this repository cannot verify - if `spf.rho` and `spf.mu` do not
resolve in a given model, the export is simply absent and the config
carries the numbers instead. Nothing downstream can tell the difference.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FluidProperties:
    """
    Density and dynamic viscosity of one simulation, in SI units.

    rho : float
        kg/m^3.

    mu : float
        Pa s. This is the DYNAMIC viscosity, the mu multiplying the
        Laplacian - not the kinematic nu = mu / rho.

    source : str
        Where the numbers came from, for the run log. A residual that
        silently used a default would be worse than one that failed.
    """

    rho: float
    mu: float
    source: str = "unknown"

    def __post_init__(self):

        for name in ("rho", "mu"):

            value = getattr(self, name)

            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be a number, got {value!r}."
                )

            if not value > 0:
                raise ValueError(
                    f"{name} must be strictly positive, got {value!r}."
                )

    @property
    def nu(self):
        """Kinematic viscosity, mu / rho."""

        return self.mu / self.rho

    def reynolds(self, velocity, length):
        """
        The Reynolds number these properties imply for a given scale.

        Only used to cross-check against the number in the dataset file
        name: if they disagree, either the properties or the reference
        scales are not what they are assumed to be, and the momentum
        residual would be wrong by that factor without ever failing.
        """

        return velocity * length / self.nu

    @classmethod
    def from_simulation(cls, simulation):
        """From the .mat, or None if it does not carry them."""

        if simulation.rho is None or simulation.mu is None:
            return None

        return cls(
            rho=float(simulation.rho),
            mu=float(simulation.mu),
            source=f"{simulation.file_path} (COMSOL)",
        )

    @classmethod
    def from_config(cls, entry, simulation_id=None):
        """
        From a `fluid:` mapping in the experiment YAML.

        Accepts either one pair for every simulation

            fluid:
              rho: 1.0
              mu: 0.005

        or one pair per simulation id

            fluid:
              0: {rho: 1.0, mu: 0.025}
              1: {rho: 1.0, mu: 0.0167}
        """

        if entry is None:
            return None

        if "rho" in entry and "mu" in entry:
            return cls(
                rho=float(entry["rho"]),
                mu=float(entry["mu"]),
                source="config",
            )

        if simulation_id is None:
            return None

        per_simulation = entry.get(simulation_id)

        if per_simulation is None:
            return None

        return cls(
            rho=float(per_simulation["rho"]),
            mu=float(per_simulation["mu"]),
            source=f"config[{simulation_id}]",
        )

    def __repr__(self):

        return (
            f"FluidProperties(rho={self.rho:g}, mu={self.mu:g}, "
            f"nu={self.nu:g}, source={self.source!r})"
        )
