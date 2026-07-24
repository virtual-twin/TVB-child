# -*- coding: utf-8 -*-

<%include file="ND_coupling.jl.mako" />
<%include file="ND_dfuns.jl.mako" />
## <%include file="jax-integrate.py.mako" />
## <%include file="jax-monitors.py.mako" />
## <%namespace name="monitors" file="jax-monitors.py.mako"/>
## <%namespace name="utils" file="jax-utils.py.mako"/>

<%
    from tvb.simulator.integrators import IntegratorStochastic
    stochastic = isinstance(sim.integrator, IntegratorStochastic)
    any_delays = sim.connectivity.idelays.any()
%>

## Monitors
## timeseries = namedtuple("timeseries", ["time", "trace"])
## % for i, monitor in enumerate(sim.monitors):
## ${monitors.create_monitor(i, monitor)}
## % endfor

## ## Main Function
## def kernel(initial_conditions, weights, delay_indices, dt, nt, noise, params_integrate, params_monitors):
##     # problem dimensions
##     n_node = ${sim.connectivity.weights.shape[0]}
##     n_svar = ${len(sim.model.state_variables)}
##     n_cvar = ${len(sim.model.cvar)}
##     n_modes = ${sim.model.number_of_modes}
##     nh = ${sim.connectivity.horizon}
##     time_steps = jnp.arange(0, nt)

##     ## initial conditions go through the carry of scan
##     current_state, history = initial_conditions
##     % if any_delays:
##         history = jnp.concatenate([jnp.empty((n_cvar, nt, n_node, n_modes)), history], axis = 1)
##     %endif
##     state = (history, current_state)
    
##     op = lambda state, external_input: integrate(state, weights, params_integrate, delay_indices, external_input)
##     % if stochastic:

##     _, trace = jax.lax.scan(op, state, (time_steps, noise))
##     % else:
##     _, trace = jax.lax.scan(op, state, time_steps)
##     % endif
    
##     ## Extract new initial conditions if needed
##     % if return_new_ics:
##     ics = namedtuple("initial_conditions", ["current_state", "history"])

##     new_current_state = trace[-1, :, :, :]
##     % if any_delays:
##     cvar = ${utils.array_input(sim.model.cvar)}
##     new_history = jnp.transpose(trace[-nh:, cvar, :, :], (1, 0, 2, 3))
##     % else:
##     new_history = = trace[-1, :, :, :]
##     % endif
##     new_ics = ics(new_current_state, new_history)
##     % endif

##     ## Apply variables of interest and generate derived variables
##     <%
##     vois = sim.model.variables_of_interest
##     svars = sim.model.state_variables
##     svars_is_vois = svars == vois
##     %>

##     ## Generate expressions for derived variables and potentially remove and reorder state variables
##     % if not svars_is_vois:
##     # state variables: ${svars} to variables of interest: ${vois}
##         trace = jnp.hstack((
##             % for var in vois:
##                 % if var in svars:
##                     trace[:, [${(svars.index(var))}], :]\
##                 % else:
##                     ${utils.gernerate_derived_expresion(var, svars)}\
##                 % endif
##                 , # ${var}
##             % endfor
##         ))
##     % endif

##     t_offset = ${sim.current_step}
##     time_steps = time_steps + 1
    
##     ## Apply monitors to trace
##     ${('result = [' if (len(sim.monitors) > 1) else 'result = ')} \
##     % for i, monitor in enumerate(sim.monitors):
##     ${monitors.apply_monitor(i, monitor)}
##     % endfor
##     ${']' if (len(sim.monitors) > 1) else ''}
##     ${'result = result[0]' if (len(sim.monitors) == 1) else ''}
##     % if not return_new_ics:
##     return result
##     % else:
##     return (result, new_ics)
##     % endif