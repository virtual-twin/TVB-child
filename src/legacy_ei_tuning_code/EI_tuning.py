import numpy as np
import jax
import jax.numpy as jnp

import copy

from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from tvb_autodiff.jax import JaxBackend

from tvb_fit.base.parameter import *
from tvb_fit.base.prior import *
from tvb_fit.base.gen_model import GenerativeModel, GenerativeModelBuilder
from tvb_fit.base.observation_models import *

from functools import partial

# Building loss function
def fc_corr(fc1, fc2):
    return jnp.corrcoef(fc1.flatten(), fc2.flatten())[0, 1]

def rmse(matrix1, matrix2):
    rmse_value = jnp.sqrt(jnp.square(matrix1 - matrix2).mean())
    return rmse_value

def setup_EI(sim, fic_target = np.array([0.25]), eta_fic = 0.1, fic_window = 100, sl_fc = 2.5 * 60_000, enable_x86 = False):
    gmb_mfm = GenerativeModelBuilder(sim = sim, backend = JaxBackend(enable_x64=enable_x86))

    p_init = gmb_mfm.select_parameters(["coupling_w", "model_w_p"])

    ## FC
    def get_fc(trace):
        _fc = jnp.corrcoef(trace, rowvar=False)
        # diag elements 0
        return _fc.at[jnp.diag_indices(_fc.shape[0])].set(0)
        
    om = ObservationModel(transforms = [
                        SelectMonitor(mon_idx=1),
                        # Calculate FC + FCD from the bold signal skipping the transient 
                        TransformPrediction(lambda y: get_fc(y.trace[10:, 0, :, 0])),
                        ], metadata = None)
    om.build()

    ## FIC
    pre_state_var_ind = 1
    post_state_var_ind = 0 

    # define observation model to calculate the fic parameter update
    def distance(x):
            return x - fic_target

    def get_parameter_update(y):
        pre = jnp.mean(y[:, pre_state_var_ind], axis = 0)
        post = jnp.mean(y[:, post_state_var_ind], axis = 0)
        dist = distance(post)
        # return eta * (pre * post - fic_target * pre), dist  
        return eta_fic * (pre * dist), dist  

    om_fic = ObservationModel(
        transforms=[
            # Select Raw
            SelectMonitor(mon_idx=0),
            SelectTrace(),
            # Select only the second half of trace and discard the first as transient
            # Subset(idx_time=jnp.s_[int(fic_window/2):], idx_mode=0),
            Subset(idx_time=jnp.s_[:], idx_mode=0),
            TransformPrediction(get_parameter_update),
        ],
        metadata=None,
    )

    om_fic.build()

    n_nodes = sim.connectivity.weights.shape[1]
    
    gm_mfm_fc = gmb_mfm.build(replace_temporal_averaging = True, print_source = False, simulation_length = sl_fc, use_tvbo = False)
    gm_mfm_fc.params.model_w_p.set_shape((n_nodes, 1))
    gm_mfm_fc.observation_model = om
    sl_fic = fic_window
    gm_mfm_fic = gmb_mfm.build(replace_temporal_averaging = True, print_source = False, simulation_length = sl_fic, use_tvbo = False)
    gm_mfm_fic.observation_model = om_fic
    gm_mfm_fic.params.model_w_p.set_shape((n_nodes, 1))
    
    ## Debug printing to detect any nans
    # Noise
    print(f"All finite Noise: {np.isfinite(gm_mfm_fc.noise).all()}")
    print(f"All finite Noise: {np.isfinite(gm_mfm_fic.noise).all()}")
    # ICS
    print(f"All finite ICS: {np.isfinite(gm_mfm_fc.initial_conditions.current_state).all()}")
    print(f"All finite ICS: {np.isfinite(gm_mfm_fc.initial_conditions.history).all()}")
    print(f"All finite ICS: {np.isfinite(gm_mfm_fic.initial_conditions.current_state).all()}")
    print(f"All finite ICS: {np.isfinite(gm_mfm_fic.initial_conditions.history).all()}")


    return gm_mfm_fc, gm_mfm_fic, p_init

def tune_EI(gm_fc, gm_fic, p_init, target_fc, N = 10, eta_EI = 0.1, N_fic_init = 15000, N_fic = 1000, abs_tol = 0.025, verbose = False, patience = 10, min_delta = 1e-4, jit_fic = False):
    n_nodes = p_init.coupling_w.shape[0]
    # Loop core for FIC update
    def update_parameters(input):
        p, ics, dist, i = input
        print(i)
        ts, new_ics = gm_fic.kernel(gm_fic.preprocess(p), ics, noise = gm_fic.noise)
        om_res = gm_fic.observation_model(ts, gm_fic.preprocess(p))
        update, dist = om_res[0]
        p_new = Parameters([p[0], Parameter(p[1].name, p[1].value - jnp.reshape(update, p[1].value.shape), p[1].low, p[1].high, p[1].inds, p[1].doc)])

        return (p_new, tuple(new_ics), dist, i + 1)
    
    # if not jit_fic: # jit is the default in while_loop
    #     update_parameters = partial(jax.jit, static_argnums=())(update_parameters)

    def fic_fun(p, ics, N, jit_fic):
        if jit_fic:
            p, ics_opt, dist, i = jax.lax.while_loop(lambda x: jnp.any(jnp.abs(x[2]) > abs_tol) & (x[3] < N), update_parameters, (p, ics, jnp.ones(n_nodes), 0))
        else:
            with jax.disable_jit():
                p, ics_opt, dist, i = jax.lax.while_loop(lambda x: jnp.any(jnp.abs(x[2]) > abs_tol) & (x[3] < N), update_parameters, (p, ics, jnp.ones(n_nodes), 0))

        return p, ics_opt, dist, i

    # wrap and compile fc calculation
    def calc_fc(p):
        ts, new_ics = gm_fc.kernel(gm_fc.preprocess(p), gm_fc.initial_conditions, noise = gm_fc.noise)
        om_res = gm_fc.observation_model(ts, gm_fc.preprocess(p))
        new_fc = om_res[0]
        return new_fc

    calc_fc_jit = jax.jit(calc_fc)

    # Initial FIC tuning
    p = p_init.copy()
    p_best_seen = p.copy()
    p, ics_opt, dist, i = fic_fun(p, tuple(gm_fic.initial_conditions), N_fic_init, jit_fic)
    # p, ics_opt, dist, i = jax.lax.while_loop(lambda x: jnp.any(jnp.abs(x[2]) > abs_tol) & (x[3] < N_fic_init), update_parameters, (p, tuple(gm_fic.initial_conditions), jnp.ones(n_nodes), 0))
    # p, ics_opt, dist, i = jax.lax.while_loop(lambda x: jnp.any(jnp.abs(x[2]) > abs_tol) & (x[3] < N_fic_init), update_parameters_jit, (p, tuple(gm_fic.initial_conditions), jnp.ones(n_nodes), 0))
    if verbose:
        print(f"Initial FIC tuning - Iterations: {i}, mean distance to target: {jnp.mean(dist)}")
    # ics = gm_fic.initial_conditions
    ics = ics_opt
    print(f"optimized ICS finite: {np.isfinite(ics_opt[0]).all()}")
    print(f"optimized ICS finite history: {np.isfinite(ics_opt[1]).all()}")
    # p = copy.deepcopy(p_init)

    if verbose:
            sl = gm_fc.simulation_length
            print(f"Running {N} iterations with simulation length {(sl/1000):.2f}s at learning rate {eta_EI}")
    
    # Init early stopping
    patience_counter = 0
    best_loss = np.inf
    loss_rmse = []
    loss_fcr = []
    # EI tuning
    t = trange(N, desc='EI tuning', leave=True)
    for i in t:
        # calc FC
        fc_pred = calc_fc_jit(p)
        # differences to true FC
        diff_FC = target_fc - fc_pred
        rmse_FC = rmse(target_fc, fc_pred)
        mrmse = np.mean(rmse_FC)
        fcr = fc_corr(fc_pred, target_fc)
        loss_rmse.append(rmse_FC)
        loss_fcr.append(fcr)
        t.set_postfix_str(f'|RMSE| = {mrmse:.4f}, FC r = {fcr:.4f}')

        # early stopping
        if mrmse < best_loss - min_delta:
            best_loss = mrmse
            patience_counter = 0
            p_best_seen = p.copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping, no improvement of minimum {min_delta} after {patience} iterations!")
                break

        # max difference for scaling the learning rate for faster convergence
        # mdf = np.max(diff_FC)
        # mrmse = np.max(rmse_FC)
        wLRE = p.coupling_w.value[:, 0, :]
        wFFI = p.coupling_w.value[:, 1, :]

        # Calculate parameter update and new Parameters
        wLRE = jnp.clip(wLRE + eta_EI * (diff_FC) * rmse_FC, 0, None)
        wFFI = jnp.clip(wFFI - eta_EI  * (diff_FC) * rmse_FC, 0, None)
        # wLRE = jnp.clip(wLRE + (eta_EI * jnp.minimum(10, (1/mdf)*(1/mrmse))) * (diff_FC) * rmse_FC, 0, None)
        # wFFI = jnp.clip(wFFI - (eta_EI * jnp.minimum(10, (1/mdf)*(1/mrmse))) * (diff_FC) * rmse_FC, 0, None)
        w = jnp.stack([wLRE,wFFI], axis = 1)
        p = Parameters([Parameter(p[0].name, w, p[0].low, p[0].high, p[0].inds, p[0].doc), p[1]])

        # Retune FIC
        p, ics, dist, i = fic_fun(p, tuple(ics), N_fic, jit_fic)

        # p, ics, dist, _i = jax.lax.while_loop(lambda x: jnp.any(jnp.abs(x[2]) > abs_tol) & (x[3] < N_fic), update_parameters, (p, tuple(ics), jnp.ones(n_nodes), 0))
        # p, ics, dist, _i = jax.lax.while_loop(lambda x: jnp.any(jnp.abs(x[2]) > abs_tol) & (x[3] < N_fic), update_parameters_jit, (p, tuple(ics), jnp.ones(n_nodes), 0))

        # Additional info
        # if (i % np.maximum(int(N/10), 1) == 0) and verbose:
        #     print(f"Iteration {i}, mean rmse {jnp.mean(rmse_FC):.3f}, FC corr {fc_corr(fc_pred, target_fc):.3f}")
            # print(f"mdf {mdf:.3f} mrmse {mrmse:.3f} sf: {(1/mdf)*(1/mrmse):.2f}")
            # print(f"I: {_i}, mean dist: {jnp.mean(dist)}")
            # plt.imshow(fc_pred, vmax = 0.8)
            # plt.title(f"FC{i} r = {fc_corr(fc_pred, target_fc):.3f}")
            # plt.show()
        
    p, ics, dist, i = fic_fun(p, tuple(ics), N_fic, jit_fic)
    # p, ics, dist, _i = jax.lax.while_loop(lambda x: jnp.any(jnp.abs(x[2]) > abs_tol) & (x[3] < N_fic), update_parameters, (p, tuple(ics), jnp.ones(n_nodes), 0))
    # p, ics, dist, _i = jax.lax.while_loop(lambda x: jnp.any(jnp.abs(x[2]) > abs_tol) & (x[3] < N_fic), update_parameters_jit, (p, tuple(ics), jnp.ones(n_nodes), 0))
        
    return p, ics, p_best_seen, loss_rmse, loss_fcr

def run_EI(sim, target_fc, EI_protocol, fic_target = np.array([0.25]), rel_tol = 0.1, abs_tol = None, eta_fic = 0.1, fic_window = 100, N_fic_init = 15000, N_fic = 1000,  p_init = None, verbose = False, patience = 10, min_delta = 10e-4, update_with_best_seen = True, jit_fic = True, enable_x86 = False):
    if abs_tol is None:
        abs_tol = rel_tol * fic_target.min()

    p_next = p_init
    total_loss_rmse = []
    total_loss_fcr = []
    for setting in EI_protocol:
        N, sl, eta_EI = setting
        

        gm_mfm_fc, gm_mfm_fic, p_mfm_init = setup_EI(sim, fic_target = fic_target, eta_fic = eta_fic, fic_window = fic_window, sl_fc = sl, enable_x86=enable_x86)
        if p_next is None:
            p_next = p_mfm_init

        p_last, ics, p_best_seen, loss_rmse, loss_fcr = tune_EI(gm_mfm_fc, gm_mfm_fic, p_next, target_fc, N = N, N_fic_init=N_fic_init, N_fic = N_fic, eta_EI=eta_EI, abs_tol = abs_tol, verbose = verbose, patience = patience, min_delta = min_delta, jit_fic=jit_fic)
        total_loss_rmse = total_loss_rmse + loss_rmse
        total_loss_fcr = total_loss_fcr + loss_fcr
        if update_with_best_seen:
            p_next = p_best_seen
        else:
            p_next = p_last 

    return p_last, ics, p_best_seen, total_loss_rmse, total_loss_fcr