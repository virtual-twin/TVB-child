# from tvb_fit.base.prior import Parameters
from tvb_fit.base.observation_models import ObservationModel
from tvb.simulator.simulator import Simulator
from tvb_autodiff.jax import JaxBackend
from tvb_fit.base.parameter import Parameter, Parameters
from tvb.simulator.lab import *

import numpy as np
import difflib

identity = lambda x: x
class GenerativeModel:
    # 
    def __init__(
            self,
            sim: Simulator,
            params: Parameters,
            kernel = None,
            simulation_length = None,
            initial_conditions = None,
            observation_model: ObservationModel = None,
            noise = None,
            preprocess = identity
    ):
        self.sim = sim
        self.params = params
        self.kernel = kernel
        self.initial_conditions = initial_conditions
        self.observation_model = observation_model
        self.preprocess = preprocess # This function is applied to the parameters before simulation
        self.noise_generator = None
        self.noise = noise
        if simulation_length is not None:
            self.simulation_length = simulation_length
        else:
            self.simulation_length = self.sim.simulation_length

    def __call__(self, params = None, ics = None):
        if params is not None:
            return self.run_with_params(params, ics)
        else:
            return self.run()

    def run(self):
        return self.run_with_params(self.params)

    def run_with_params(self, params, ics = None, noise = None):
        if ics is None:
            ics = self.initial_conditions
        if noise is None:
            noise = self.noise
        result, new_ics = self.kernel(self.preprocess(params), ics, noise = noise)
        self.initial_conditions = new_ics
        if self.observation_model is not None:
            return self.observation_model(result, self.preprocess(params))
        else:
            return result
        
    def __str__(self) -> str:
        # All model information from sim
        # params
        # behavior (dynamic ir static ICs, ...)
        description = f"""
        NMM:\t\t {self.sim.model.__class__.__name__}
        Coupling:\t {self.sim.coupling.__class__.__name__}
        Nodes:\t\t {self.sim.connectivity.weights.shape[0]}
        """
        return description


class GenerativeModelBuilder:

    def __init__(
            self,
            sim: Simulator,
            backend = JaxBackend(enable_x64=True),
            observation_model: ObservationModel = ObservationModel(),
            simulation_length = None,
    ):
        # Add all required fields
        backend._tvbo_enrich_fields(sim)
        self.sim = sim
        self.backend = backend
        self.observation_model = observation_model
        self._available_parameters = self.available_parameters() 
        if simulation_length is not None:
            self.simulation_length = simulation_length
        else:
            self.simulation_length = self.sim.simulation_length        
    
    def available_parameters(self):
        """
        Return a list of all available Parameters.
        """
        sim = self.sim
        has_delays = sim.connectivity.idelays.any()
        if has_delays:
            possible_names = ["connectivity_weights", "current_state", "history"]
        else:
            possible_names = ["connectivity_weights", "current_state"]
        
        # model 
        possible_names += [f"model_{par}" for par in sim.model.global_parameter_names + sim.model.spatial_parameter_names]
        
        # coupling
        possible_names += [f"coupling_{par}" for par in sim.coupling.parameter_names]

        # stimulus
        if sim.stimulus is not None:
            possible_names += [f"stimulus_{par}" for par in sim.stimulus.temporal.parameters.keys()]

        # monitors
        for i, monitor in enumerate(sim.monitors):
            if isinstance(monitor, (monitors.Raw, monitors.RawVoi, monitors.TemporalAverage, monitors.SubSample)):
                pass # nothing to add
            elif isinstance(monitor, monitors.Bold):
                possible_names += [f"monitor_{i}_Bold_{par}" for par in monitor.hrf_kernel.parameters.keys()]
            elif isinstance(monitor, monitors.Projection):
                # Observation Noise might be an option later
                possible_names += [f"monitor_{i}_{monitor.__class__.__name__}_gain"]

        return possible_names
    
    def select_parameters(self, names):
        """
        Select parameters for Inversion
        """
        if isinstance(names, str):
            names = [names]
        params = []
        for name in names:
            if not (name in self._available_parameters):
                raise ValueError(f"Parameter {name} is not part of the simulator, maybe you meant:{difflib.get_close_matches(name, self._available_parameters, cutoff=0.6)}. Valid parameters are: {self._available_parameters}")
            params.append(self._path_to_parameter(name))  
        params = Parameters(params)    
        self.params = params
        return params

    def _path_to_parameter(self, path): 
        if "model_" in path:
            p_name = path.split("model_")[-1]
            p = self._instance_to_parameter(path, p_name, self.sim.model)
            pass
        elif "coupling_" in path:
            p_name = path.split("coupling_")[-1]
            p = self._instance_to_parameter(path, p_name, self.sim.coupling)
            pass
        elif "stimulus_" in path:
            p_name = path.split("stimulus_")[-1]
            if p_name == "spatial_pattern":
                value = self.sim.stimulus._spatial_pattern
            else:
                value = np.array(self.sim.stimulus.temporal.parameters[p_name])
            # p = Parameter(path, value, shape = value.shape)
            p = Parameter(path, value)
        
        elif "monitor_" in path:
            imon = int(path.split("_")[1])
            mon = self.sim.monitors[imon]
            mon_type = path.split("_")[2]
            p_name = path.split(f"monitor_{imon}_{mon_type}_")[-1]
            if mon_type == "Bold":
                value = np.array(mon.hrf_kernel.parameters[p_name])
            elif mon_type in ["EEG", "MEG", "SEEG"] and p_name == "gain":
                value = np.array(mon.gain)   
            # p = Parameter(path, value, shape = value.shape)
            p = Parameter(path, value)

        # elif path in ["weights", "current_state", "history"]#, "noise"]:
        elif "connectivity_weights" in path:
            value = self.sim.connectivity.weights
            # p = Parameter(path, value, shape = value.shape)
            p = Parameter(path, value)

        elif "current_state" in path:
            # Jax backend specific
            value = self.sim.current_state[:,:,0]
            # p = Parameter(path, value, shape = value.shape)
            p = Parameter(path, value)

        elif "history" in path:
            # Jax backend specific
            has_delays = self.sim.connectivity.idelays.any()
            buf = self.sim.history.buffer[...,0]
            rbuf = np.concatenate((buf[0:1], buf[1:][::-1]), axis=0)
            history = np.transpose(rbuf, (1, 0, 2)).astype('f')
            n_hist_var, _, n_node = history.shape
            value = history.reshape((n_hist_var, self.sim.connectivity.horizon, n_node))
            if not has_delays:
                value = self.sim.current_state[:,:,0]
            # p = Parameter(path, value, shape = value.shape)
            p = Parameter(path, value)

        else:
            raise ValueError(f"Parameter {path} is not valid")
        return p

    def _instance_to_parameter(self, path, p_name, obj):
        # Take value from sim and metadata from class
        value = obj.__dict__[p_name]
        neo_obj = None
        try:
            neo_obj = obj.__class__.__dict__[p_name]
        except:
            pass
        if not neo_obj == None:
            try:
                # p = Parameter(path, value, shape = value.shape, low = neo_obj.domain.lo, high = neo_obj.domain.hi, inds = None, doc = neo_obj.doc)
                p = Parameter(path, value, low = neo_obj.domain.lo, high = neo_obj.domain.hi, inds = None, doc = neo_obj.doc)
            except:
                # p = Parameter(path, value, shape = value.shape)
                p = Parameter(path, value)

        else:
            # p = Parameter(path, value, shape = value.shape)
            p = Parameter(path, value)

        return p

    def build(self, update_ics = True, simulation_length = None, **kwargs):
        """
        Build the generative model as specified.

        * `update_ics`: Update the initial conditions of the model after each call to the kernel, similar behavior as TVBs `.run()` function.
        * `simulation_length`: Length of the simulation. If not specified, the length of the simulator is used.
        * `kwargs`: Further keyword arguments to be passed to the backend `.build()` function, e.g. `replace_temporal_averaging`, `small_dt`, `use_tvbo`, `print_source`, etc.
        """
        if simulation_length is None:
            simulation_length = self.simulation_length
        # Create a kernel function that takes parameters as input and returns the simulation result + the new initial conditions.
        kernel_raw, initial_params, constructors = self.backend.build(self.sim, simulation_length=simulation_length, return_constructors = True, return_new_ics=True, **kwargs)

        def names_and_idxs(pd):
            names = list(pd.keys())
            idxs = [v.name for v in pd.values()]
            return names, idxs
        # model
        p_names_dfun, p_index_dfun = names_and_idxs(self.params.get_model_params())
        # coupling 
        p_names_cfun, p_index_cfun = names_and_idxs(self.params.get_coupling_params())
        # stimulus
        p_names_stim, p_index_stim = names_and_idxs(self.params.get_stimulus_params())
        # monitors
        p_names_mon, p_index_mon = [], []
        for imon, _ in enumerate(self.sim.monitors):
            _p_names_mon, _p_index_mon = names_and_idxs(self.params.get_monitor_params(id = imon))
            p_names_mon.append(_p_names_mon)
            p_index_mon.append(_p_index_mon)

        # ic parameters
        p_names_ics = [n for n in ["current_state", "history"] if n in self.params._fields]
        if len(p_names_ics) > 0 and update_ics:
            print("Warning: initial conditions are part of the parameter set and are updated by the inversion method and not by the simulation.")
            update_ics = False

        # main parameters
        p_names = [n for n in ["connectivity_weights"] if n in self.params._fields]
        if update_ics:
            set_ics = lambda parameters, p_names_ics, ics: ics
            ics = constructors[1]()
        else:
            set_ics = lambda parameters, p_names_ics, ics: constructors[1](**dict(zip(p_names_ics, [getattr(parameters, name).value for name in p_names_ics])))
            ics = constructors[1]()

        def kernel(parameters, ics, noise = None):
            params, initial_conditions, params_dfun, params_cfun, params_stim, params_integrate, params_monitors_constructors = constructors
            params_dfun_pred = params_dfun(**dict(zip(p_names_dfun, [getattr(parameters, i).value for i in p_index_dfun])))
            params_cfun_pred = params_cfun(**dict(zip(p_names_cfun, [getattr(parameters, i).value for i in p_index_cfun])))
            params_stim_pred = params_stim(**dict(zip(p_names_stim, [getattr(parameters, i).value for i in p_index_stim])))
            params_integrate_pred = params_integrate(params_dfun=params_dfun_pred, params_cfun=params_cfun_pred, params_stimulus=params_stim_pred)
            params_monitors_pred = []
            for imon, mon in enumerate(params_monitors_constructors):
                params_monitors_pred.append(mon(**dict(zip(p_names_mon[imon], [getattr(parameters, i).value for i in p_index_mon[imon]]))))

            ics = set_ics(parameters, p_names_ics, ics)
            params_pred = params(initial_conditions=ics, noise = noise, **dict(zip(p_names, [getattr(parameters, i).value for i in p_names])), params_integrate=params_integrate_pred, params_monitors=params_monitors_pred)
            return kernel_raw(*params_pred)
            
        if self.observation_model is not None:
            self.observation_model.build()

        return GenerativeModel(self.sim, self.params, kernel = kernel, simulation_length = self.simulation_length, initial_conditions=ics, observation_model = self.observation_model, noise = initial_params.noise)

