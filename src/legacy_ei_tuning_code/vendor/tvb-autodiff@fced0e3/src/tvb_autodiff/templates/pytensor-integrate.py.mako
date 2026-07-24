# -*- coding: utf-8 -*-

import numpy as np
import pytensor
from pytensor import tensor as pyt

<%
    from tvb.simulator.integrators import (IntegratorStochastic,
        EulerDeterministic, EulerStochastic,
        HeunDeterministic, HeunStochastic,
        Identity, IdentityStochastic, RungeKutta4thOrderDeterministic,
        SciPyODEBase)
    from tvb.simulator.noise import Additive, Multiplicative

    if isinstance(sim.integrator, SciPyODEBase):
        raise NotImplementedError

    if isinstance(sim.integrator, IntegratorStochastic):
        if isinstance(sim.integrator.noise, Multiplicative):
            raise NotImplementedError

    any_delays = sim.connectivity.idelays.any()
%>

<%def name = "array_input(array)" filter="trim">
        np.array(${np.array2string(array, separator=",")})
</%def>

def integrate(state, weights, dX, cX, params_integrate, delay_indices
% if isinstance(sim.integrator, IntegratorStochastic):
    , z_t
% endif
):
    dt = ${sim.integrator.dt}
    params_dfun, params_cfun, params_stimulus = params_integrate
    history, current_state = state

    cX = cfun(cX, weights, history, current_state, params_cfun, delay_indices)

    dX = pyt.set_subtensor(dX[0], dfun(dX[0], current_state, cX, params_dfun))
% if isinstance(sim.integrator, EulerDeterministic):
    next_state = current_state + dt * dX[0]
% endif
% if isinstance(sim.integrator, EulerStochastic):
    next_state = current_state + dt * dX[0] + z_t
% endif
% if isinstance(sim.integrator, HeunDeterministic):
    dX = pyt.set_subtensor(dX[1], dfun(dX[1], current_state + dt * dX[0], cX, params_dfun))
    next_state = current_state + dt / 2 * (dX[0] + dX[1])
% endif
% if isinstance(sim.integrator, HeunStochastic):
    dX = pyt.set_subtensor(dX[1], dfun(dX[1], current_state + dt * dX[0] + z_t, cX, params_dfun))
    next_state = current_state + dt / 2 * (dX[0] + dX[1]) + z_t
% endif
% if isinstance(sim.integrator, Identity):
    next_state = dX[0]
% endif
% if isinstance(sim.integrator, IdentityStochastic):
    next_state = dX[0] + z_t
% endif
% if isinstance(sim.integrator, RungeKutta4thOrderDeterministic):
    dX = pyt.set_subtensor(dX[1], dfun(dX[1], current_state + dt / 2 * dX[0], cX, params_dfun))
    dX = pyt.set_subtensor(dX[2], dfun(dX[2], current_state + dt / 2 * dX[1], cX, params_dfun))
    dX = pyt.set_subtensor(dX[3], dfun(dX[3], current_state + dt * dX[2], cX, params_dfun))
    next_state = current_state + dt / 6 * (dX[0] + 2*(dX[1] + dX[2]) + dX[3])
% endif
    ## state = pyt.set_subtensor(state[:], pyt.roll(state, 1, axis=1))
    ## state = pyt.set_subtensor(state[:, 0], next_state)

    ## return state

% if any_delays:
    cvar = ${array_input(sim.model.cvar)}
    history = pyt.set_subtensor(history[:], pyt.roll(history, -1, axis=1))
    history = pyt.set_subtensor(history[:, -1, :], next_state[cvar, :])
% else:
    history = next_state
% endif
    return (history, next_state)
