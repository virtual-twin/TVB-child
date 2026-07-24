=====================
tvb-autodiff
=====================

.. image:: https://img.shields.io/pypi/v/tvb_autodiff.svg
   :alt: PyPI Version

.. image:: https://readthedocs.org/projects/tvb-autodiff/badge/?version=latest
   :alt: Documentation Status

A lightweight dependency to provide TVB backends for automatic model differentiation. All additional model information is provided through The Virtual Brain Ontology [TVBO](https://github.com/virtual-twin/tvbo-python).

*Free software:* GNU General Public License v3
*Documentation:* https://tvb-autodiff.readthedocs.io

Quickstart
----------

Currently 3 backends are available:

.. code-block:: python

   from tvb_autodiff.jax import JaxBackend
   from tvb_autodiff.pytensor import PyTensorBackend
   from tvb_autodiff.julia import JuliaBackend

A standard TVB simulator object can be used to generate backend specific code:

.. code-block:: python

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
   sim.configure()

   JB = JaxBackend()
   kernel, parameters = JB.build(sim, print_source=True)

Or run simulations directly (Jax and PyTensor only):

.. code-block:: python

   JB.run(sim)