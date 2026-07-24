# -*- coding: utf-8 -*-


"""
A Julia backend which uses templating to generate simulation
code.

.. moduleauthor:: Marius Pille <marius.pille@bih-charite.de>

"""

import sys
import tempfile
import os

from mako.template import Template
from mako.lookup import TemplateLookup
from mako.exceptions import text_error_template


here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
# from .templates import MakoUtilMix
from .backend_utils import BackendUtils

from tvb.simulator.lab import *

# from jax import config

class JuliaBackend(BackendUtils):

    def __init__(self, enable_x64 = True):
        self.cgdir = tempfile.TemporaryDirectory()
        sys.path.append(self.cgdir.name)
        # config.update("jax_enable_x64", enable_x64)
    
    # Dedicated Mako Utils

    @property
    def lookup(self):
        lookup = TemplateLookup(directories=[here])
        return lookup

    def render_template(self, source, content):
        template = Template(source, lookup=self.lookup, strict_undefined=True)
        try:
            source = template.render(**content)
        except Exception as exc:
            print(text_error_template().render())
            raise exc
        return source

    def insert_line_numbers(self, source):
        lines = source.split('\n')
        numbers = range(1, len(lines) + 1)
        nu_lines = ['%03d\t%s' % (nu, li) for (nu, li) in zip(numbers, lines)]
        nu_source = '\n'.join(nu_lines)
        return nu_source
    
    def build_py_func(self, template_source, content, name='kernel', print_source=True, fname=None, do_line_numbers=False):
        "Build and retrieve one or more Python functions from template."
        source = self.render_template(template_source, content)
        # source = autopep8.fix_code(source)
        if print_source:
            if do_line_numbers:
                source = self.insert_line_numbers(source)
            print(source)
        if fname is not None:
            # check if fname dir exists
            dirname = os.path.dirname(fname)
            if os.path.exists(dirname):
                print("Saving at:")
                print(fname)
                with open(fname, 'w') as fd:
                    fd.write(source)
            else: 
                # warn
                print(f"WARNING: Directory {dirname} does not exist. Not saving simulation")
    
        return source
            

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
    
    def build(self, sim, simulation_length=None, save_at=None, print_source=False, replace_temporal_averaging=False, return_constructors=False, return_new_ics=False, use_tvbo=True):
        """
        Build Julia code from a simulator object.

        * `sim`: is a simulator object.
        * `simulation_length`: is the length of time to simulate, default is sim.simulation_length.
        * `save_at`: Specify a path where the simulation will be saved as .py file. 
        * `print_source`: prints the generated source code to stdout.
        * `replace_temporal_averaging`: if False the simulator will use the TemporalAverage monitor in the Bold and EEG monitor as TVB does. If True the simulator will use the SubSample monitor instead, which is faster and gives very similar results. 
        * `return_constructors`: See below
        * `return_new_ics`: if True, the kernel function will return updated initial conditions (current state + history) along with the simulation output. This is of interest if you want continue simulations. The output of kernel will change from result to [result, (current_state, history)]
        * `use_tvbo`: if True, TVBO will be used to add missing fields to the model and coupling classes if possible. 

        Returns a tuple of (code, params):
        
        * code is the main simulator code which can be evaluated in Julia via `juliacall` or saved for later use
        * params is a tuple of arguments containing the state of the simulator:
            
        current_state, weights, delay_indices, history, dt, nt, noise, params_integrate, params_monitor

        If return_constructors = True returns also the constructors for the named tuples:

        (kernel, params(), [params, params_dfun, params_cfun, params_stim, params_integrate, params_monitors_constructors])
        """
        return super().build(
            sim,
            simulation_length=simulation_length,
            save_at=save_at,
            print_source=print_source,
            replace_temporal_averaging=replace_temporal_averaging,
            return_constructors=return_constructors,
            return_new_ics=return_new_ics,
            use_tvbo=use_tvbo,
            template='<%include file="ND_sim.jl.mako"/>'
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

    