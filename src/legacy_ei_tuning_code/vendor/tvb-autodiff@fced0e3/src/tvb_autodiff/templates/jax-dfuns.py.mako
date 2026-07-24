# -*- coding: utf-8 -*-

<%namespace name="utils" file="jax-utils.py.mako"/>

import jax.numpy as jnp
import jax.scipy as jsp

from collections import namedtuple

## Derivatives of state variables
def dfun(current_state, cX, params_dfun, local_coupling=0):
    \
% for par in sim.model.global_parameter_names + sim.model.spatial_parameter_names:
${par}, \
% endfor
= params_dfun

    pi = jnp.pi
    where = jnp.where
    sqrt = jnp.sqrt
    log = jnp.log
    erfc = jsp.special.erfc
    exp = jnp.exp

    # unpack coupling terms and states as in dfun
    % for i, cterm in enumerate(sim.model.coupling_terms):
    ${cterm} = cX[${i}]
    % endfor

    <%
    if sim.model.non_integrated_variables == None:
        integrated_state_variables = sim.model.state_variables
    else:
        integrated_state_variables = [var for var in sim.model.state_variables if var not in sim.model.non_integrated_variables]
    %>
    % for i, ivar in enumerate(integrated_state_variables):
    ${ivar} = current_state[${i}]
    % endfor

    # compute internal states for dfun
    % for var, term in sim.model.state_variable_dfuns.items():
        %if var not in integrated_state_variables:
        ${var} = ${term}    
        %endif
    % endfor

    %if sim.model.non_integrated_variables == None:
    return jnp.array([
        % for svar in sim.model.state_variables:
            ${sim.model.state_variable_dfuns[svar]},
        % endfor
        ])
    %else:
    # compute integrated variables
    ivars = jnp.array([
        % for svar in sim.model.state_variables:
            %if svar not in sim.model.non_integrated_variables:
            ${sim.model.state_variable_dfuns[svar]},
            %endif
        % endfor
        ])
    # non-integrated variables
    nivars = jnp.array([
        % for var in sim.model.non_integrated_variables:
            ${var},
        % endfor
        ])
    return (ivars, nivars)
    %endif

    
