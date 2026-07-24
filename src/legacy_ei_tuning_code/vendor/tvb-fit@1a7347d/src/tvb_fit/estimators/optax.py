import numpy as np
import jax
import jax.numpy as jnp
import optax

from tvb_fit.estimators.estimator import PointEstimator
from tvb_fit.estimators.callbacks import *
from tvb_fit.base.parameter import Parameters

import copy
from functools import partial

class OptaxPointEstimator(PointEstimator):

    def simulate(self,parameters, ics, metadata):
        result, ics_new = self.gm.kernel(self.gm.preprocess(parameters), ics, noise = self.gm.noise)
        prediction = self.gm.observation_model.with_metadata(result, parameters, metadata)
        return prediction, ics_new

    def loss(self, parameters, ics, metadata):
        prediction, ics_new = self.simulate(parameters, ics, metadata)
        loss_value = self.metric(prediction, self.observation, parameters, self.gm.params)
        return (loss_value, (prediction, ics_new))

    def fit(self, iter_max = 1, mode = "rev", initial_parameters = None, initial_conditions = None):
        """
        
        """
        @jax.jit
        def simulate(parameters, ics, metadata):
            result, ics_new = self.gm.kernel(self.gm.preprocess(parameters), ics, noise = self.gm.noise)
            prediction = self.gm.observation_model.with_metadata(result, parameters, metadata)
            return prediction, ics_new

        @jax.jit
        def loss(parameters, ics, metadata):
            prediction, ics_new = simulate(parameters, ics, metadata)
            loss_value = self.metric(prediction, self.observation, parameters, self.gm.params)
            return (loss_value, (prediction, ics_new))

        if mode == "rev":
            def value_and_grad(loss, params, argnums=0, has_aux=True):
                return jax.value_and_grad(loss, argnums=argnums, has_aux=has_aux)
        elif mode == "fwd":
            def value_and_grad(loss, params, argnums=0, has_aux=True):
                return value_and_grad_fwd(loss, params, argnums=argnums, has_aux=has_aux)
        elif mode == "lin":
            def value_and_grad(loss, params, argnums=0, has_aux=True):
                return value_and_grad_lin(loss, params, argnums=argnums, has_aux=has_aux)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented. Must be 'rev', 'fwd', or 'lin'")

        def step(parameters, ics, metadata, opt_state):
            out = value_and_grad(loss, parameters, argnums=0, has_aux=True)(parameters, ics, metadata)
            (loss_value, (prediction, ics_new)), grads = out
            updates, opt_state = self.optimizer.update(grads, opt_state, parameters)
            parameters = optax.apply_updates(parameters, updates)
            return parameters, ics_new, opt_state, loss_value, prediction, grads
        
        if initial_parameters is None:
            parameters = self.gm.params
        else:
            parameters = initial_parameters

        if initial_conditions is None:
            ics = self.gm.initial_conditions
        else:
            ics = initial_conditions

        opt_state = self.optimizer.init(parameters)
        metadata = self.gm.observation_model.metadata
        fitting_data = dict() # a place to store data during fitting
        for i in range(iter_max):
            # ics need to be converted to tuple, otherwise recompilation will be triggered each iteration - alternative: unpack named tuple in state, history.
            parameters, ics, opt_state, loss_value, prediction, grads = step(parameters, tuple(ics), metadata, opt_state)
            if self.callback is not None:
                stop, parameters, ics, metadata = self.callback(i, parameters, ics, metadata, fitting_data, self.gm, prediction, loss_value, grads)
                if stop:
                    print("Stopping due to callback")
                    break
            
        return parameters, ics, metadata, fitting_data

# Below are convenience functions mimicking jax.value_and_grad but based on jvp and linearize - refactor later to module
def get_tangents(params):
    payload, aux_data = params.tree_flatten() # tuple(tuple(array))

    zero_template = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    tangents = []
    for i, p in enumerate(payload):
        for index, _ in np.ndenumerate(p[0]):
            tan = copy.deepcopy(zero_template)
            tan[i]._value = zero_template[i].value.at[index].set(1.0)
            tangents.append(tan)
    return tangents

def arr_to_par_builder(aux):
    def _fun(arr):
        payload = ()
        start_idx = 0
        for el in aux:
            shape = el[1][0]
            length = np.prod(shape)
            val = jnp.reshape(np.hstack(arr[start_idx:start_idx+length]), shape)
            start_idx += length
            payload += ((val,),)
        return Parameters.tree_unflatten(aux, payload)
    return _fun

def value_and_grad_fwd(fun, params, argnums=0, has_aux=False):
    tangents = get_tangents(params)
    payload, aux_data = params.tree_flatten()
    a2p = arr_to_par_builder(aux_data)
    def _fun(*args, **kwargs):
        f = jax._src.linear_util.wrap_init(fun, kwargs)
        f_partial, dyn_args =jax.api_util.argnums_partial(f, argnums, args, require_static_args_hashable=False)
        pushfwd = partial(jax._src.api._jvp, f_partial, has_aux=has_aux)
        grads = []
        y = None
        for tangent in tangents:
            y, grad, *aux = pushfwd((*dyn_args,), (tangent,))
            grads.append(grad)
        if has_aux:
            return (y, *aux), a2p(grads)
        else:
            return y, a2p(grads)            
    return _fun

def value_and_grad_lin(fun, params, argnums=0, has_aux=False):
    tangents = get_tangents(params)
    payload, aux_data = params.tree_flatten()
    a2p = arr_to_par_builder(aux_data)
    def _fun(*args, **kwargs):
        f = jax._src.linear_util.wrap_init(fun, kwargs)
        f_partial, dyn_args =jax.api_util.argnums_partial(f, argnums, args, require_static_args_hashable=False)        
        y, fun_jvp, *aux = jax._src.api.linearize(f_partial.call_wrapped, *dyn_args, has_aux=has_aux)
        grads = [fun_jvp(v) for v in tangents]
        if has_aux:
            return (y, *aux), a2p(grads)
        else:
            return y, a2p(grads)
    return _fun
