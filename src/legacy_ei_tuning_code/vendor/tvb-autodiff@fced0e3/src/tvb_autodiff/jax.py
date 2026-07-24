# -*- coding: utf-8 -*-


"""
A JAX backend which uses templating to generate simulation
code.

.. moduleauthor:: Marius Pille <marius.pille@bih-charite.de>

"""

import sys
import tempfile

from .templates import MakoUtilMix
from .backend_utils import BackendUtils

from tvb.simulator.lab import *

from jax import config

class JaxBackend(MakoUtilMix, BackendUtils):

    def __init__(self, enable_x64 = True):
        self.cgdir = tempfile.TemporaryDirectory()
        sys.path.append(self.cgdir.name)
        config.update("jax_enable_x64", enable_x64)

    # Just for documentation and testing
    _supported_models = (
                    models.LarterBreakspear,
                    models.ReducedWongWang,
                    models.Generic2dOscillator,
                    models.WilsonCowan,
                    models.SupHopf,
                    models.Kuramoto,
                    models.ReducedWongWangExcInh,
                    models.Linear,
                    models.JansenRit,
                    models.MontbrioPazoRoxin,
        )

    _supported_integrators = (
                    integrators.HeunStochastic,
                    integrators.HeunDeterministic,
                    integrators.EulerStochastic,
                    integrators.EulerDeterministic,
                    integrators.Identity,
                    integrators.IdentityStochastic,
                    integrators.RungeKutta4thOrderDeterministic,
        )
    
    # Just for documentation and testing
    _supported_couplings = (
                    coupling.HyperbolicTangent,
                    coupling.Kuramoto,
                    coupling.Linear,
                    coupling.Sigmoidal,
                    coupling.Difference,
                    coupling.SigmoidalJansenRit,
        )
                
    _supported_monitors = (
                    monitors.MEG,
                    monitors.Raw,
                    monitors.RawVoi,
                    monitors.SubSample,
                    monitors.TemporalAverage,
                    monitors.Bold,
                    monitors.EEG,
        )
    
    def build(self, sim, simulation_length=None, save_at=None, print_source=False, replace_temporal_averaging=False, return_constructors=False, return_new_ics=False, small_dt=False, use_tvbo=True):
        return super().build(
            sim,
            simulation_length=simulation_length,
            save_at=save_at,
            print_source=print_source,
            replace_temporal_averaging=replace_temporal_averaging,
            return_constructors=return_constructors,
            return_new_ics=return_new_ics,
            use_tvbo=use_tvbo,
            small_dt=small_dt,
            template='<%include file="jax-sim.py.mako"/>'
        )

    def extract_entity(self, sim, entity, simulation_length=None, print_source=False, replace_temporal_averaging=False):
        return super().extract_entity(
            sim,
            entity,
            simulation_length=simulation_length,
            print_source=print_source,
            replace_temporal_averaging=replace_temporal_averaging,
            template='<%include file="jax-sim.py.mako"/>'
        )

    