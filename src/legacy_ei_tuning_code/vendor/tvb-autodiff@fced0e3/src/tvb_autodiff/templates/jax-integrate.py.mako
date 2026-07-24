# -*- coding: utf-8 -*-
<%namespace name="utils" file="jax-utils.py.mako"/>

<%
    from tvb.simulator.integrators import (IntegratorStochastic,
        EulerDeterministic, EulerStochastic,
        HeunDeterministic, HeunStochastic,
        Identity, IdentityStochastic, RungeKutta4thOrderDeterministic,
        SciPyODEBase)
    from tvb.simulator.noise import Additive, Multiplicative
    from tvb.simulator.monitors import AfferentCoupling
    import jax.numpy as jnp

    if isinstance(sim.integrator, SciPyODEBase):
        raise NotImplementedError

    ## TODO handle multiplicative noise
    if isinstance(sim.integrator, IntegratorStochastic):
        if isinstance(sim.integrator.noise, Multiplicative):
            raise NotImplementedError

    stochastic = isinstance(sim.integrator, IntegratorStochastic)
    any_delays = sim.connectivity.idelays.any()
    monitor_node_coupling = any([isinstance(monitor, AfferentCoupling) for monitor in sim.monitors])
    has_stimulus = sim.stimulus != None
%>

## Stimulus
% if has_stimulus:
where = jnp.where

def get_stimulus(t, params_stimulus):
    \
% for par, _ in sim.stimulus.temporal.parameters.items():
${par}, \
% endfor
spatial_pattern, \
= params_stimulus

    dt = ${sim.integrator.dt}
    stimulus = jnp.zeros(${sim._prepare_stimulus().shape[0:2]})
    temporal_pattern = lambda var: ${sim.stimulus.temporal.equation}

    stim_at_t = spatial_pattern * temporal_pattern(t * dt)
    stimulus = stimulus.at[${utils.array_input(sim.model.stvar)}, :].set(jnp.squeeze(stim_at_t))
    return stimulus
% endif

def integrate(state, weights, params_integrate, delay_indices, external_input):
    ${'t, noise' if stochastic else 't'} = external_input
    dt = ${sim.integrator.dt}
    params_dfun, params_cfun, params_stimulus = params_integrate

    history, current_state = state
    
    ${'stimulus = get_stimulus(t, params_stimulus)' if has_stimulus else ''}

    ## cX = cfun(weights, history, current_state, params_cfun, delay_indices, t)
    cX = jax.vmap(cfun, in_axes=(None, -1, -1, None, None, None), out_axes=-1)(weights, history, current_state, params_cfun, delay_indices, t)
    ## cX = jnp.squeeze(cX, axis=-1)

## Only integrated state variables
%if sim.model.nintvar  == sim.model.nvar:
    dX0 = dfun(current_state, cX, params_dfun) 
% if isinstance(sim.integrator, EulerDeterministic):
    next_state = current_state + dt * (dX0 ${'+ stimulus' if has_stimulus else ''})
% endif
% if isinstance(sim.integrator, EulerStochastic):
    next_state = current_state + dt * (dX0 ${'+ stimulus' if has_stimulus else ''}) + noise
% endif
% if isinstance(sim.integrator, HeunDeterministic):
    dX1 = dfun(current_state + dt * (dX0 ${'+ stimulus' if has_stimulus else ''}), cX, params_dfun)
    next_state = current_state + dt * (0.5 * (dX0 + dX1) ${'+ stimulus' if has_stimulus else ''})
% endif
% if isinstance(sim.integrator, HeunStochastic):
    z = noise
    dX1 = dfun(current_state + dt * (dX0 ${'+ stimulus' if has_stimulus else ''}) + z, cX, params_dfun)
    next_state = current_state + dt * (0.5 * (dX0 + dX1) ${'+ stimulus' if has_stimulus else ''}) + z
% endif
% if isinstance(sim.integrator, Identity):
    next_state = dX0 ${'+ stimulus' if has_stimulus else ''}
% endif
% if isinstance(sim.integrator, IdentityStochastic):
    next_state = dX0 ${'+ stimulus' if has_stimulus else ''} + noise
% endif
% if isinstance(sim.integrator, RungeKutta4thOrderDeterministic):
    dX1 = dfun(current_state + dt / 2 * dX0, cX, params_dfun)
    dX2 = dfun(current_state + dt / 2 * dX1, cX, params_dfun)
    dX3 = dfun(current_state + dt * dX2, cX, params_dfun)
    next_state = current_state + dt * ((1 / 6) * (dX0 + 2 * (dX1 + dX2) + dX3) ${'+ stimulus' if has_stimulus else ''})
% endif

## Integrated and non-integrated state variables - no stimulus yet!
% else:
<%
    _itg_state_idxs = [i for i, var in enumerate(sim.model.state_variables) if var not in sim.model.non_integrated_variables]
%>
    itg_state_idxs = jnp.array(${_itg_state_idxs})
    current_state = current_state[itg_state_idxs]

    dX0, niX = dfun(current_state, cX, params_dfun)
% if isinstance(sim.integrator, EulerDeterministic):
    next_state = jnp.vstack([current_state + dt * dX0, niX])
% endif
% if isinstance(sim.integrator, EulerStochastic):
    next_state = jnp.vstack([current_state + dt * dX0 + noise[itg_state_idxs], niX])
% endif
% if isinstance(sim.integrator, HeunDeterministic):
    dX1, _ = dfun(current_state + dt * dX0, cX, params_dfun)
    next_state = jnp.vstack([current_state + dt / 2 * (dX0 + dX1), niX])
% endif
% if isinstance(sim.integrator, HeunStochastic):
    z = noise[itg_state_idxs]
    dX1, _ = dfun(current_state + dt * dX0 + z, cX, params_dfun)
    next_state = jnp.vstack([current_state + dt / 2 * (dX0 + dX1) + z, niX])
% endif
% if isinstance(sim.integrator, Identity):
    next_state = jnp.vstack([dX0, niX])
% endif
% if isinstance(sim.integrator, IdentityStochastic):
    next_state = jnp.vstack([dX0 + noise[itg_state_idxs], niX])
% endif
% if isinstance(sim.integrator, RungeKutta4thOrderDeterministic):
    dX1, _ = dfun(current_state + dt / 2 * dX0, cX, params_dfun)
    dX2, _ = dfun(current_state + dt / 2 * dX1, cX, params_dfun)
    dX3, _ = dfun(current_state + dt * dX2, cX, params_dfun)
    next_state = jnp.vstack([current_state + dt / 6 * (dX0 + 2*(dX1 + dX2) + dX3), niX])
% endif
% endif

## Clip state if bounds are defined
% if sim.model.state_variable_boundaries is not None:
    inf = jnp.inf
    ## Clip by bounderies if present else set bounderies to +-inf -> no clip
    min_bounds = jnp.array(${[[[sim.model.state_variable_boundaries[svar][0]]] if svar in sim.model.state_variable_boundaries else [[-jnp.inf]] for svar in sim.model.state_variables]})
    max_bounds = jnp.array(${[[[sim.model.state_variable_boundaries[svar][1]]] if svar in sim.model.state_variable_boundaries else [[jnp.inf]] for svar in sim.model.state_variables]})

    next_state = jnp.clip(next_state, min_bounds, max_bounds)    
% endif

## Return for scan: carry, result
% if any_delays:
    cvar = ${utils.array_input(sim.model.cvar)}
    % if small_dt:
    history = history.at[:, t, :].set(next_state[cvar, :])
    % else:
    _h = jnp.roll(history, -1, axis=1)
    history = _h.at[:, -1, :].set(next_state[cvar, :])
    % endif
% else:
    history = next_state
% endif
## Return for scan coupling if monitored
% if monitor_node_coupling:
    return (history, next_state), (next_state, cX)
% else:
    return (history, next_state), next_state
% endif


