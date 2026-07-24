import argparse
import numpy as np

from tvb.simulator.lab import *
from tvb.simulator.models import JansenRit, ReducedWongWangExcInh
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import (EulerStochastic, HeunStochastic)
from tvb.simulator.noise import Additive
from tvb.simulator.monitors import Raw, SubSample, Bold, AfferentCoupling
from tvb.simulator.simulator import Simulator
from tvb_autodiff.jax import JaxBackend
import tvb.basic.neotraits as nt

import jax
import jax.numpy as jnp
jax.config.update("jax_debug_nans", True)


import matplotlib.pyplot as plt


from tvb_fit.base.parameter import *
from tvb_fit.base.gen_model import GenerativeModel, GenerativeModelBuilder
from tvb_fit.base.observation_models import *

from EI_RWW_Model import EIB_Linear, EIB_ReducedWongWangExcInh
from EI_tuning import run_EI, setup_EI, fc_corr

from utils import *

def main(sc_path, fc_path, experiment_path, subject):
    # Your main processing logic here
    print(f"Processing SCs from: {sc_path}")
    print(f"Processing FCs from: {fc_path}")
    print(f"Experiment location: {experiment_path}")
    print(f"Processing subject: {subject}")
    fc, conn = load_data_cluster(sc_path, fc_path, subject)
    print("Loaded Data")
    
    dt = 6.0 # The larger dt the faster the simulation (stability is the limit)
    integrator = integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig = np.array([5e-6])))
    # G_init = 0.005
    G_init = 0.05

    sim_eib = Simulator(
        connectivity=conn,
        model=EIB_ReducedWongWangExcInh(J_i = np.array([1.2]), w_p = np.array([1.2]), lamda = np.array([0.1])),
        integrator=integrator,
        coupling=EIB_Linear(a = np.array([G_init]), w = np.ones((conn.weights.shape[0], 2, conn.weights.shape[0]))),
        monitors=[Raw(), Bold(period = 720)],
        simulation_length=5_000,
    )

    sim_eib.initial_conditions = 0.1 *np.ones((1, 2, conn.weights.shape[0], 1))

    sim_eib.configure();
    print("Created Simulator")

    # FIC params
    fic_target = np.array([0.25])
    eta_fic = 0.02
    fic_window = 200
    N_fic_init = 60000
    N_fic = 1000
    rel_tol = 0.1

    # EI
    p_init = None # start with params from sim
    target_fc = fc
    patience = 10 # Number of iterations to wait for improvement
    min_delta = 1e-4 # minimum change of mean rmse to be classified as improvement

    # Define a protocol as list of tuples with the structure (iterations, simulation_length, eta)
    eta = 2
    EI_protocol = [(100, 3*60_000, 0.5*eta), (50, 6*60_000, eta), (30, 10*60_000, eta)]
    # EI_protocol = [(5, 3*60_000, 0.5*eta), (5, 6*60_000, eta)]

    # %%
    p, ics, p_best_seen, total_loss_rmse, total_loss_fcr = run_EI(sim_eib, target_fc, EI_protocol, fic_target=fic_target, N_fic_init=N_fic_init, N_fic = N_fic, rel_tol = rel_tol, verbose = True, patience = patience, min_delta = min_delta, update_with_best_seen = True, jit_fic=True, enable_x86=True, eta_fic = eta_fic, fic_window = fic_window)
    save_results(os.path.join(experiment_path, f"fits/{subject}.pkl"), (p, p_best_seen, total_loss_rmse, total_loss_fcr))
    print(f"Saved results for {subject}, final rmse: {total_loss_rmse[-1]:.3f}, final fc r: {total_loss_fcr[-1]:.3f}")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subject data.")
    parser.add_argument("--sc_path", type=str, required=True, help="Path to the SC directory")
    parser.add_argument("--fc_path", type=str, required=True, help="Path to the FC directory")
    parser.add_argument("--experiment_path", type=str, required=True, help="Path to the experiment directory")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    
    args = parser.parse_args() 
    
    main(args.sc_path, args.fc_path, args.experiment_path, args.subject)
