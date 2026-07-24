# -*- coding: utf-8 -*-

import numpy as np
import pytensor
from pytensor import tensor as pyt

def dfun(dX, state, cX, params_dfun
## % for par in sim.model.parameter_names:
##     % if par in mparams:
##         , ${par}
##     % endif
## % endfor
):

    \
% for par in sim.model.global_parameter_names + sim.model.spatial_parameter_names:
${par}, \
% endfor
= params_dfun

## % for par in sim.model.global_parameter_names:
##     % if not par in mparams:
##     ${par} = ${getattr(sim.model, par)[0]}
##     % endif
## % endfor

## % for par in sim.model.spatial_parameter_names:
##     % if not par in mparams:
##     ${par} = parmat[${loop.index}]
##     % endif
## % endfor

    pi = np.pi

% for _, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
    ${cterm} = cX[${loop.index}]
% endfor

<%
if sim.model.non_integrated_variables == None:
    integrated_state_variables = sim.model.state_variables
else:
    integrated_state_variables = [var for var in sim.model.state_variables if var not in sim.model.non_integrated_variables]
%>

% if sim.model.non_integrated_variables == None:
    % for svar in sim.model.state_variables:
        ${svar} = state[${loop.index}]
    % endfor
% else:
    % for svar in sim.model.state_variables:
        % if svar not in sim.model.non_integrated_variables:
            ${svar} = state[${loop.index}] 
        % endif
    % endfor
% endif
    
    # compute internal states for dfun    
% for var, term in sim.model.state_variable_dfuns.items():
    %if var not in integrated_state_variables:
        ${var} = ${term}
    % endif
% endfor

    # compute dfun
%if sim.model.non_integrated_variables == None:
    % for svar in sim.model.state_variables:
        dX = pyt.set_subtensor(dX[${loop.index}], ${sim.model.state_variable_dfuns[svar]});
    % endfor
    return dX
%else:
    # compute integrated variables
    % for svar in sim.model.state_variables:
        % if svar not in sim.model.non_integrated_variables:
            dX = pyt.set_subtensor(dX[${loop.index}], ${sim.model.state_variable_dfuns[svar]});
        % endif
    % endfor
    # non-integrated variables
    nivars = [
    % for svar in sim.model.non_integrated_variables:
        ${svar},
    % endfor
        ]
    return (dX, nivars)
%endif
