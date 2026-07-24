# -*- coding: utf-8 -*-

import numpy as np
import pytensor
import pytensor.tensor as pyt
from collections import namedtuple

<%include file="pytensor-coupling.py.mako" />
<%include file="pytensor-dfuns.py.mako" />
<%include file="pytensor-integrate.py.mako" />
<%include file="pytensor-monitors.py.mako" />
<%namespace name="monitors" file="pytensor-monitors.py.mako"/>

<%
    from tvb.simulator.integrators import IntegratorStochastic
    from tvb.simulator.noise import Additive
    stochastic = isinstance(sim.integrator, IntegratorStochastic)
    any_delays = sim.connectivity.idelays.any()
%>

timeseries = namedtuple("timeseries", ["time", "trace"])
% for i, monitor in enumerate(sim.monitors):
${monitors.create_monitor(i, monitor)}
% endfor

## TODO handle multiplicative noise
## % if isinstance(sim.integrator, IntegratorStochastic):
## def default_noise(nsig):
##     n_node = ${sim.connectivity.weights.shape[0]}
##     n_svar = ${len(sim.model.state_variables)}
##     nt = ${int(sim.simulation_length/sim.integrator.dt)}
##     sqrt_dt = ${np.sqrt(sim.integrator.dt)}
##     np.random.seed(42)

##     white_noise = sqrt_dt * pyt.as_tensor_variable(np.random.randn(nt, n_svar, n_node))
##     noise_gfun = pyt.sqrt(2 * nsig)
##     return pyt.transpose(pyt.transpose(white_noise, (1, 0, 2)) * noise_gfun, (1, 0, 2))
## % else:
## def default_noise():
##     # no noise function rendered for integrator ${type(sim.integrator)}
##     return None
## % endif

def kernel(trace, current_state, weights, history, params_integrate, params_monitors
           ${', noise' if stochastic else ''}
           , delay_indices
           ):

    # problem dimensions
    n_node = ${sim.connectivity.weights.shape[0]}
    n_svar = ${len(sim.model.state_variables)}
    n_cvar = ${len(sim.model.cvar)}
    nt = ${int(sim.simulation_length/sim.integrator.dt)}
    nh = ${sim.connectivity.horizon}
    time_steps = pyt.arange(0, nt)

    # work space arrays
    dX = pyt.zeros((${sim.integrator.n_dx}, n_svar, n_node))
    cX = pyt.zeros((n_cvar, n_node))

    def scan_fn(${'noise,' if stochastic else ''} history, current_state):
        state = (history, current_state)
        history, next_state = integrate(state, weights, dX, cX, params_integrate, delay_indices ${', noise' if stochastic else ''})
        return [history, next_state]

    ([_, trace], updates) = pytensor.scan(fn=scan_fn, outputs_info=[history, current_state], non_sequences=[], n_steps=nt ${', sequences=[noise]' if stochastic else ''})
    ## trace = trace[:, :, 0, :]

    ## Apply variables of interest and generate derived variables
    <%
    vois = sim.model.variables_of_interest
    svars = sim.model.state_variables
    svars_is_vois = svars == vois
    %>

    ## Apply monitors to trace
    ${('result = [' if (len(sim.monitors) > 1) else 'result = ')} \
    % for i, monitor in enumerate(sim.monitors):
    ${monitors.apply_monitor(i, monitor)}
    % endfor
    ${']' if (len(sim.monitors) > 1) else ''}
    ${'return result' if (len(sim.monitors) > 1) else 'return result[0]'}
