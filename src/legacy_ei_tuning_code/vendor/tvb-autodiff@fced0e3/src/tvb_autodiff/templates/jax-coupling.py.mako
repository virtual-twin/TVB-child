# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from collections import namedtuple

sin, cos, exp, tanh = jnp.sin, jnp.cos, jnp.exp, jnp.tanh

<% 
has_delay = sim.connectivity.idelays.any()
%>
## Coupling function
def cfun(weights, history, current_state, params_cfun, delay_indices, t):
    n_node = weights.shape[0]
    
## Force the right indentation
    \
% for par in sim.coupling.parameter_names:
${par}, \
% endfor
= params_cfun

## History is sparse, only states that are coupled are stored in history therefore we use cvar_idx (assumes cvar is ordered)
## JAX does not throw out of bounds errors but returns the last valid index, so be careful! 
## Collect x_i and x_j as needed, pre needs all cvars at the same time

% if 'x_i' in sim.coupling.pre_expr: ## don't generate x_i if not required
    x_i = jnp.array([
% for cvar_idx, cvar in enumerate(sim.model.cvar):
    current_state[${cvar}, :],
% endfor
    ])
    x_i = x_i.transpose(1, 0)
    ## %if has_delay:
    x_i = jnp.expand_dims(x_i, axis=-1)
    ## %endif
% endif

## if no non-zero idelays, use current state
% if 'x_j' in sim.coupling.pre_expr: ## don't generate x_j if not required
    x_j = jnp.array([
% for cvar_idx, cvar in enumerate(sim.model.cvar):
    % if has_delay:
    % if small_dt:
    history[${cvar_idx}, delay_indices[0].T + t, delay_indices[1]],
    % else:
    history[${cvar_idx}, delay_indices[0].T, delay_indices[1]],
    % endif
    % else:
    history[${cvar}, delay_indices[1]],
    % endif
% endfor
    ])
    ## %if has_delay:
    x_j = x_j.transpose(1, 0, 2) ## (n_node, n_cvar, ...) expected 
    ## %else:
    ## x_j = x_j.transpose(1, 0) ## (n_node, n_cvar) expected
    ## %endif
        
% endif

## Apply pre-expression this can reduce and collapse the cvar dimension, eg. SigmoidalJansenRit
    pre = ${sim.coupling.pre_expr}
    ## %if has_delay:
    pre = pre.reshape(n_node, -1 ,n_node) ## Restore collapsed dimension if necessary
    ## %else:
    ## pre = pre.reshape(n_node, -1) ## Restore collapsed dimension if necessary
    ## %endif


## Apply weights
## delay dotproduct -> sum: (nnodes x nnodes) x (nnodes x nnodes
## % if has_delay:
    op = lambda x: jnp.sum(weights * x, axis=-1)
    gx = jax.vmap(op, in_axes=1)(pre)
## % else:
## no-delay matmul: (nnodes x nnodes) x (nnodes x n_cvar) = (nnodes x n_cvar)
    ## gx = jnp.matmul(weights, pre)
    ## gx = gx.T
## % endif
    return ${sim.coupling.post_expr}
