import pytest
from gnn_comsol.physics import FluidProperties

def test_from_config_shared():
    f = FluidProperties.from_config({"rho": 1.0, "mu": 0.005})
    assert (f.rho, f.mu) == (1.0, 0.005)
    assert f.nu == 0.005
    assert f.source == "config"

def test_from_config_per_simulation():
    entry = {0: {"rho": 1.0, "mu": 0.025}, 1: {"rho": 2.0, "mu": 0.5}}
    assert FluidProperties.from_config(entry, 1).mu == 0.5
    assert FluidProperties.from_config(entry, 7) is None
    assert FluidProperties.from_config(None, 0) is None

def test_rejects_nonsense():
    for bad in ({"rho": 0.0, "mu": 1.0}, {"rho": 1.0, "mu": -1.0}):
        with pytest.raises(ValueError, match="positive"):
            FluidProperties(**bad)

def test_reynolds():
    f = FluidProperties(rho=1.0, mu=0.01)
    assert f.reynolds(velocity=2.0, length=1.0) == pytest.approx(200.0)
