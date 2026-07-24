# -*- coding: utf-8 -*-

"""
A plain PyTensor backend which uses templating to generate simulation
code.

"""

import os
import sys
import importlib
import autopep8
import tempfile

from .templates import MakoUtilMix
from .backend_utils import BackendUtils

from tvb.simulator.lab import *
from collections import namedtuple

import numpy as np
import pytensor.tensor as pyt


class PytensorBackend(MakoUtilMix, BackendUtils):

    def __init__(self):
        self.cgdir = tempfile.TemporaryDirectory()
        sys.path.append(self.cgdir.name)

    # Just for documentation
    _supported_models = (
                    models.LarterBreakspear,
                    models.ReducedWongWang,
                    models.WilsonCowan,
                    models.SupHopf,
                    models.Kuramoto,
                    models.Generic2dOscillator,
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
    
    # Just for documentation
    _supported_couplings = (
                    coupling.Kuramoto,
                    coupling.Linear,
                    coupling.Sigmoidal,
                    coupling.Difference,
                    coupling.SigmoidalJansenRit,
        )
                
    _supported_monitors = (
                    monitors.Raw,
                    monitors.RawVoi,
                    monitors.SubSample,
                    monitors.TemporalAverage,
                    monitors.Bold,
                    monitors.EEG,
                    monitors.MEG,
        )
    
    def build(self, sim, simulation_length=None, save_at=None, print_source=False, replace_temporal_averaging=False, return_constructors=False, return_new_ics=False, use_tvbo=True):
        if return_new_ics:
            raise NotImplementedError("return_new_ics is not implemented for PytensorBackend")
        
        kernel, params = super().build(
            sim,
            simulation_length=simulation_length,
            save_at=save_at,
            print_source=print_source,
            replace_temporal_averaging=replace_temporal_averaging,
            return_constructors=return_constructors,
            return_new_ics=return_new_ics,
            use_tvbo=use_tvbo,
            template='<%include file="pytensor-sim.py.mako"/>'
        )

        yh = pyt.as_tensor_variable(np.zeros((int(np.ceil(sim.simulation_length/sim.integrator.dt)), sim.model.nvar, sim.connectivity.number_of_regions)), name="yh")
        current_state = pyt.as_tensor_variable(params.initial_conditions.current_state, name="current_state")
        weights = pyt.as_tensor_variable(params.weights, name="weights")
        history = pyt.as_tensor_variable(params.initial_conditions.history, name="history")

        if isinstance(sim.integrator, integrators.IntegratorStochastic):
            noise = pyt.as_tensor_variable(params.noise, name="noise")
            return kernel, (yh, current_state, weights, history, params.params_integrate, params.params_monitors, noise, params.delay_indices)
        else:
            return kernel, (yh, current_state, weights, history, params.params_integrate, params.params_monitors, params.delay_indices)
        
    def run(self, sim, simulation_length=None, print_source=False, replace_temporal_averaging=False):
        kernel, args = self.build(sim, simulation_length=simulation_length, print_source=print_source, replace_temporal_averaging=replace_temporal_averaging)
        return kernel(*args)
        
    def extract_entity(self, sim, entity, simulation_length=None, print_source=False, replace_temporal_averaging=False):
        return super().extract_entity(
            sim,
            entity,
            simulation_length=simulation_length,
            print_source=print_source,
            replace_temporal_averaging=replace_temporal_averaging,
            template='<%include file="pytensor-sim.py.mako"/>'
        )
