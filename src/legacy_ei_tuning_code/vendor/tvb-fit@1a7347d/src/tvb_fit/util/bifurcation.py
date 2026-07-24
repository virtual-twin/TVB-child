from collections import namedtuple
import numpy as np
import matplotlib as mpl
import pandas as pd
from tvb_fit.base.gen_model import GenerativeModelBuilder
from tvb_autodiff.jax import JaxBackend
import copy

def sample_initial_conditions(sim, number_ics = 1):
    """
    Sample initial conditions based on the ranges defined for each model.
    Stack `number_ics` initial conditions in the mode dimension.
    """
    n_time, n_svar, n_node, _ = sim.good_history_shape
    history = sim.model.initial(None, (n_time, n_svar, n_node, number_ics))
    initial_conditions = history[n_time - 1]
    # select only svars needed for coupling
    history = history[:, sim.model.cvar, :, :].transpose(1, 0, 2, 3)
    return namedtuple("initial_conditions", ["current_state", "history"])(initial_conditions, history)


def adiabatic_scan(sim, om, parameter, p_start, p_end, n_steps, number_ics = 1, backend = JaxBackend(enable_x64=True), simulation_length = 2_000, skip_length = 1_000, bothways = False, inner_aggregation = [np.mean, np.max, np.min], outer_aggregation = [np.mean, np.std], verbose = False):
    """
    Perform an adiabatic scan of the parameter space for the given simulator and observation model.

    * `sim`: Simulator object.
    * `om`: Observation model object.
    * `parameter`: Name of the parameter to scan.
    * `p_start`: Start of the parameter range to scan.
    * `p_end`: End of the parameter range to scan.
    * `n_steps`: Number of steps to take in the parameter range.
    * `number_ics`: Number of initial conditions to sample.
    * `backend`: Backend to use for the simulation.
    * `simulation_length`: Length of the simulation to run [ms].
    * `skip_length`: Length of the simulation to skip at the beginning [ms].
    * `bothways`: Whether to scan the parameter range both ways, usefull for detecting hysteresis.
    * `inner_aggregation`: Aggregation functions to apply to the inner dimension (time dimension).
    * `outer_aggregation`: Aggregation functions to apply to the outer dimension (node and initial condition dimension).
    * `verbose`: Whether to print the progress.

    Returns a DataFrame with the results (`p`, [inner + "_" + outer for inner in inner_aggregation for outer in outer_aggregation]) of the adiabatic scan as columns.
    """

    sim.model.number_of_modes = number_ics
    gmb = GenerativeModelBuilder(sim, backend = backend)
    gmb.sim.simulation_length = simulation_length
    gmb.observation_model = copy.copy(om)
    p_init = gmb.select_parameters(parameter)
    if verbose: print("Scanning parameter p: \n", p_init)
    gm = gmb.build(update_ics=True)

    ics = sample_initial_conditions(sim, number_ics)
    gm.initial_conditions = ics

    p_range = np.linspace(p_start, p_end, n_steps)
    if bothways:
        p_range = np.hstack([p_range, p_range[::-1][1:]])   
    
    names = ["p"] + [f"{inner.__name__}_{outer.__name__}" for inner in inner_aggregation for outer in outer_aggregation]
    df = pd.DataFrame(columns = names, dtype = float)
    for p in p_range:
        if verbose: print("p:", p)
        getattr(gm.params, parameter).set_value(p)
        ts = gm.run()[0][-skip_length::]
        res = [p]
        for inner in inner_aggregation:
            in_state = inner(ts, axis = 0)
            for outer in outer_aggregation:
                res.append(outer(in_state))
        df.loc[len(df)] = np.array(res, dtype = float)
    return df

## BifurcationKit related
def plot_branch(γ, ax, vars = ("param", "x"), branchlabel = None):
    try:
        p = getattr(γ.branch, vars[0])
        x = getattr(γ.branch, vars[1])
    except:
        raise ValueError(f"Invalid vars: {vars}. To see available vars, use: jl.propertynames(γ.branch)")

    stable = γ.branch.stable

    points = np.array([p, x]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mpl.collections.LineCollection(segments, linewidths=np.array(stable) + 1, linestyles = ['-' if b else ':' for b in stable], colors='k', label=branchlabel)
    ax.add_collection(lc)
    ax.autoscale()

    colors = {"bp": "tab:blue", "hopf": "tab:red", "fold": "tab:green", "nd": "tab:purple", "ns": "tab:brown", "pd": "tab:olive"} # Codim 1 special points
    def get_color(sp_type):
        if sp_type in colors:
            return colors[sp_type]
        elif sp_type in ["bt", "gh", "cusp", "zh", "hh"]: # Codim 2 equilibrium special points
            return "tab:orange"
        else:
            return "tab:gray"
        
    get_marker = lambda status: "o" if str(status) == "converged" else "x"
    labeled = dict() # add label once per special points
    for sp in γ.specialpoint:
        if str(sp.type) == "endpoint":
            continue
        if labeled.keys().__contains__(str(sp.type)):
            label = None 
        else: 
            labeled[str(sp.type)] = len(labeled)
            label = str(sp.type)

        # ax.scatter(getattr(γ[sp.idx], vars[0]), np.linalg.norm(getattr(γ[sp.idx], vars[1]), 1), c=get_color(str(sp.type)), alpha = 0.8, marker=get_marker(sp.status), zorder=10, label = label)
        ax.scatter(getattr(γ[sp.idx], vars[0]), getattr(γ[sp.idx], vars[1]), c=get_color(str(sp.type)), alpha = 0.8, marker=get_marker(sp.status), zorder=10, label = label)

    # ax.set_xlabel(f"{jl.BK.get_lens_symbol(γ.prob.lens)}")

# We need to unpack each parameter value from array to scalar for BifurcationProblem
def convert_to_scalar_floats(named_tuple):
    return named_tuple._replace(**{field: getattr(named_tuple, field).item() for field in named_tuple._fields})