# -*- coding: utf-8 -*-

import numpy as np
import pytensor
from pytensor import tensor as pyt

sin, cos, exp, tanh = pyt.sin, pyt.cos, pyt.exp, pyt.tanh

def cfun(cX, weights, history, current_state, params_cfun, delay_indices
## % if sim.connectivity.idelays.any():
##     , delay_indices
## % endif
## % for par in sim.coupling.parameter_names:
##     % if par in cparams:
##         , ${par}
##     % endif
## % endfor
):

    n_node = ${sim.connectivity.number_of_regions}
    n_cvar = ${len(sim.model.cvar)}
    nh = ${sim.connectivity.horizon}

    \
% for par in sim.coupling.parameter_names:
${par}, \
% endfor
= params_cfun

## % for par in sim.coupling.parameter_names:
##     % if not par in cparams:
##     ${par} = ${getattr(sim.coupling, par)[0]}
##     % endif
## % endfor

% if 'x_i' in sim.coupling.pre_expr:
    x_i = pyt.zeros((n_cvar, n_node))
% for cvar_idx, cvar in enumerate(sim.model.cvar):
    x_i = pyt.set_subtensor(x_i[${cvar}], current_state[${cvar}, :])
% endfor
    x_i = x_i.transpose(1, 0)
    x_i = pyt.shape_padright(x_i)
% endif

% if 'x_j' in sim.coupling.pre_expr:
    x_j = pyt.zeros((n_cvar, n_node, n_node))
% for cvar_idx, cvar in enumerate(sim.model.cvar):
    %if sim.connectivity.idelays.any():
    x_j = pyt.set_subtensor(x_j[${cvar_idx}], history[${cvar_idx}, delay_indices[0].T, delay_indices[1]])
    % else:
    x_j = pyt.set_subtensor(x_j[${cvar_idx}], history[${cvar}, delay_indices[1]])
    % endif
% endfor
    x_j = x_j.transpose(1, 0, 2)
% endif
    pre = ${sim.coupling.pre_expr}

% for cvar_idx, cvar in enumerate(sim.model.cvar):
    gx = pyt.sum(weights * pre[:, ${cvar_idx}, :], axis=-1)
    cX = pyt.set_subtensor(cX[${cvar_idx}], ${sim.coupling.post_expr})
% endfor

    return cX

## ## generate code per cvar
## % for cvar, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
## ## don't generate x_i if not required
## % if 'x_i' in sim.coupling.pre_expr:
##     x_i = pyt.transpose(pyt.reshape(pyt.tile(state[${cvar}, :], (1, n_node)), (n_node, n_node))) # Reshaping and transposing to match the order of indexing between x_i and x_j
## % endif

## % if sim.connectivity.idelays.any():
##     x_j = history[${cvar}, delay_indices[0].T, delay_indices[1]]
##     ## x_j = pyt.flatten(state[${cvar}])[delay_indices]
## % else:
##     x_j = history[${cvar}, delay_indices[1]]
##     ## x_j = pyt.tile(state[${cvar}, 0], (n_node, 1))
## % endif
##     x_j = x_j.transpose(1, 0)
##     pre = ${sim.coupling.pre_expr}

##     gx = pyt.sum(weights * pre, axis=-1)
##     cX = pyt.set_subtensor(cX[${loop.index}], ${sim.coupling.post_expr})
## % endfor

##     return cX
