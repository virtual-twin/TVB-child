# -*- coding: utf-8 -*-

<%namespace name="utils" file="ND_utils.jl.mako"/>

## give model to here in order to get funs for simple models without the need to create a whole sim

function NodeDynamics!(du, u, p, t, c = zeros(${len(sim.model.coupling_terms)}))
    \
% for par in sim.model.global_parameter_names + sim.model.spatial_parameter_names:
${par}, \
% endfor
= p
    local_coupling = 0
<%
integrated_state_variables = sim.model.state_variables
%>
    % for i, ivar in enumerate(integrated_state_variables):
    ${ivar} = u[${i+1}]
    % endfor
    
    % for i, cterm in enumerate(sim.model.coupling_terms):
    ${cterm} = c[${i+1}]
    % endfor

    % for var, term in sim.model.state_variable_dfuns.items():
    %if var not in integrated_state_variables:
    ${var} = ${utils.py2jl(term) }   
    %endif
    % endfor

% for i, svar in enumerate(sim.model.state_variables):
    du[${i+1}] = ${utils.py2jl(sim.model.state_variable_dfuns[svar])}
% endfor

    return du
end

BKDynamics(u, p) = NodeDynamics!(similar(u), u, p, 0.0)

## ## Derivatives of state variables
## def dfun(current_state, cX, params_dfun):
##     \
## % for par in sim.model.global_parameter_names + sim.model.spatial_parameter_names:
## ${par}, \
## % endfor
## = params_dfun

##     pi = jnp.pi

##     # unpack coupling terms and states as in dfun
##     % for i, cterm in enumerate(sim.model.coupling_terms):
##     ${cterm} = cX[${i}]
##     % endfor

##     <%
##     if sim.model.non_integrated_variables == None:
##         integrated_state_variables = sim.model.state_variables
##     else:
##         integrated_state_variables = [var for var in sim.model.state_variables if var not in sim.model.non_integrated_variables]
##     %>
##     % for i, ivar in enumerate(integrated_state_variables):
##     ${ivar} = current_state[${i}]
##     % endfor

##     # compute internal states for dfun
##     % for var, term in sim.model.state_variable_dfuns.items():
##         %if var not in integrated_state_variables:
##         ${var} = ${term}    
##         %endif
##     % endfor

##     %if sim.model.non_integrated_variables == None:
##     return jnp.array([
##         % for svar in sim.model.state_variables:
##             ${sim.model.state_variable_dfuns[svar]},
##         % endfor
##         ])${'[:, None]' if sim.connectivity.weights.shape[0] == 1 else ''} ## restore node dimension for connectivities with only one node
##     %else:
##     # compute integrated variables
##     ivars = jnp.array([
##         % for svar in sim.model.state_variables:
##             %if svar not in sim.model.non_integrated_variables:
##             ${sim.model.state_variable_dfuns[svar]},
##             %endif
##         % endfor
##         ])${'[:, None]' if sim.connectivity.weights.shape[0] == 1 else ''}
##     # non-integrated variables
##     nivars = jnp.array([
##         % for var in sim.model.non_integrated_variables:
##             ${var},
##         % endfor
##         ])${'[:, None]' if sim.connectivity.weights.shape[0] == 1 else ''}
##     return (ivars, nivars)
##     %endif