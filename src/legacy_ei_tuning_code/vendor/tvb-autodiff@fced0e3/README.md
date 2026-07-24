# tvb-autodiff

![PyPI Version](https://img.shields.io/pypi/v/tvb_autodiff.svg)

[![Documentation Status](https://readthedocs.org/projects/tvb-autodiff/badge/?version=latest)](https://tvb-autodiff.readthedocs.io/en/latest/?version=latest)



A thin layer to provide TVB backends for automatic model differentiation. All additional model information is provided through The Virtual Brain Ontology [TVBO](https://github.com/virtual-twin/tvbo-python).

<img src="./diff_TVB.png" alt="tvb-autodiff" width="400"/>
<!-- - **Free software:** GNU General Public License v3 -->
<!-- - **Documentation:** [https://tvb-autodiff.readthedocs.io](https://tvb-autodiff.readthedocs.io) -->

## Quickstart

Currently 3 backends are available, based on:

* [JAX](https://jax.readthedocs.io/en/latest/)
* [PyTensor](https://pytensor.readthedocs.io/en/latest/)
* Julia based on [NetworkDynamics.jl](https://pik-icone.github.io/NetworkDynamics.jl/dev/)

```python
from tvb_autodiff.jax import JaxBackend
from tvb_autodiff.pytensor import PyTensorBackend
from tvb_autodiff.julia import JuliaBackend
``` 

A standard TVB simulator object can be used to generate backend specific code:

```python
from tvb.simulator.simulator import Simulator
from tvb.simulator.models import ReducedWongWang
from tvb.simulator.monitors import Raw
from tvb.simulator.coupling import Linear
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import HeunDeterministic

sim = Simulator(
    connectivity=Connectivity.from_file(), 
    model=ReducedWongWang(), 
    integrator=HeunDeterministic(dt = 1.0), 
    coupling = Linear(),
    monitors=[Raw()], 
    )

sim.configure();

JB = JaxBackend()
kernel, parameters = JB.build(sim, print_source=True)
```

Or run simulations directly (Jax and PyTensor only):

```python
JB.run(sim)
```

## Development Status

The JAX backend is the most tested and feature complete backend. The PyTensor backend should be on par but is less tested due to its lower performance on average. The Julia backend is a prototype and can only be used for code generation, though an integration via `juliacall` is planned.