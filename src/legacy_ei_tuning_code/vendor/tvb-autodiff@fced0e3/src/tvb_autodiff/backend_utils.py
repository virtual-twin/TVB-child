import numpy as np

from tvb.simulator.lab import *

from collections import namedtuple
    
class BackendUtils:
    
    # Integrators and monitors require custom implementations so we define them explicitly
    _supported_integrators = None
    _supported_monitors = None

    # Models and couplings require a set of fields which we check 
    _model_required_fields = ["coupling_terms", "state_variable_dfuns", "parameter_names", "state_variables"]
    _coupling_required_fields = ["parameter_names", "pre_expr", "post_expr"]

    def _check_field_present(self, obj, field):
        """
        Check that a field is present in an object and not None.
        """
        try:
            getattr(obj, field)
        except AttributeError:
            raise NotImplementedError(f"Missing required field: {field} for simulator component: {obj.__class__.__name__}")
        if getattr(obj, field) is None:
            raise NotImplementedError(f"Required field: {field} is None for simulator component: {obj.__class__.__name__}") 
        
    def _check_choices( self, val, choices, allow_subclass=False):
        if not isinstance(val, choices):
            raise NotImplementedError("Unsupported simulator component. Given: {}\nExpected one of: {}".format(val, choices))

    def _check_model(self, model):
        for field in self._model_required_fields:
            self._check_field_present(model, field)

    def _check_coupling(self, coupling):
        for field in self._coupling_required_fields:
            self._check_field_present(coupling, field)
        
    def check_compatibility(self,sim):
        """
        Check that the simulator is compatible with the backends.
        """
        # monitors
        for monitor in sim.monitors:
            self._check_choices(monitor, self._supported_monitors)
        # integrators
        self._check_choices(sim.integrator, self._supported_integrators)
    
        # models 
        self._check_model(sim.model)
        
        # coupling
        self._check_coupling(sim.coupling)

        # surface
        if sim.surface is not None:
            raise NotImplementedError("Surface simulation not supported.")

    def collect_state(self, sim, simulation_length=None, return_constructors=False):
        """
        Collect all state (parameters, initial conditions, etc) that defines the brain network model, which take the structure, where all fields are always present and are None if not used:
        
        params: namedtuple\n
        ├─  initial_conditions: namedtuple\n
        │   ├─  current_state: ArrayLike[float]\n
        │   ├─  history: ArrayLike[float]\n
        ├─  weights: ArrayLike[float]\n
        ├─  delay_indices: ArrayLike[int]\n
        ├─  dt: float\n
        ├─  nt: int\n
        ├─  noise: ArrayLike[float]\n
        ├─  params_integrate: namedtuple\n
        │   ├─  params_dfun: namedtuple\n
        │   ├─  params_cfun: namedtuple\n
        │   └─  params_stimulus: namedtuple\n
        └─ params_monitors: List\n
            ├─ params_monitor_0: namedtuple\n
            ├─ ...\n
            └─ params_monitor_x: namedtuple\n
        
        If return_constructors = True returns also the constructors for the named tuples:

        params, initial_conditions, params_dfun, params_cfun, params_stim, params_integrate, params_monitors_constructors
        """

        self.check_compatibility(sim) # maybe skip this as state is independent from backends

        # Simulation Length
        assert simulation_length is not None or sim.simulation_length is not None
        if simulation_length is None:
            simulation_length = sim.simulation_length
        nt = int(np.ceil(simulation_length/sim.integrator.dt))
        
        n_node = sim.connectivity.number_of_regions
        n_modes = sim.model.number_of_modes
        n_svar = sim.model.nvar
        
        # History - in no-delay case, the same as current_state
        has_delays = sim.connectivity.idelays.any()
        if has_delays:
            # buf = np.copy(sim.history.buffer[...,0])
            buf = np.copy(sim.history.buffer)
            # history has to go from oldest to newest
            # last insert in buffer is at current_step

            n_roll = -(sim.current_step + 1  % sim.history.n_time)
            rbuf = np.roll(buf, n_roll, axis=0)
            # history = np.transpose(rbuf, (1, 0, 2)).astype('f')
            history = np.transpose(rbuf, (1, 0, 2, 3)).astype('f')
            n_hist_var, _, n_node, n_mode = history.shape
            history = history.reshape((n_hist_var, sim.connectivity.horizon, n_node, n_mode))
        else:
            # _history = np.copy(sim.current_state[:,:,0])
            _history = np.copy(sim.current_state)
            history = _history
        
        # Initial Conditions
        # current_state = np.copy(sim.current_state[:, :, 0])
        current_state = np.copy(sim.current_state)
        ics = namedtuple("initial_conditions", ["current_state", "history"], defaults=[current_state, history])

        # Time Step
        dt = sim.integrator.dt 

        # Connectivity
        weights = sim.connectivity.weights.copy()
        dn = np.arange(n_node) * np.ones((n_node, n_node)).astype(int)
        if sim.connectivity.has_delays:
            di = -1 * sim.connectivity.idelays -1
            delay_indices = (di, dn)
        else:
            delay_indices = (None, dn)
        
        # Noise
        if isinstance(sim.integrator, integrators.IntegratorStochastic):
            noise =  self._evaluate_noise(sim.integrator.noise, nt, n_svar, n_node, n_modes = n_modes)
        else:
            noise = None
        
        # dfun
        params_dfun = self._collect_dfun_params(sim) 
        # cfun
        params_cfun = self._collect_cfun_params(sim)
        # stimulus
        if sim.stimulus is None:
            params_stim = namedtuple("params_stimulus", ["params_stim",], defaults=[None,])
        else:
            params_stim = namedtuple("params_stimulus", list(sim.stimulus.temporal.parameters.keys()) + ["spatial_pattern"], defaults=list(sim.stimulus.temporal.parameters.values()) + [np.squeeze(sim.stimulus._spatial_pattern)])
        params_integrate = namedtuple('params_integrate', ["params_dfun", "params_cfun", "params_stimulus"], defaults=(params_dfun(), params_cfun(), params_stim()))

        # monitors
        params_monitors_constructors = []
        params_monitors = []
        for i, monitor in enumerate(sim.monitors):
            if isinstance(monitor, (monitors.Raw, monitors.RawVoi, monitors.TemporalAverage, monitors.SubSample)):
                params = namedtuple(f"params_monitor_{i}", [f"params_monitor_{i}",], defaults=(None,))
            elif isinstance(monitor, monitors.Bold):
                # Bold monitor has state aka stock
                # stock = np.flip(np.copy(monitor._stock[:, :, :, 0]), axis = 0)
                stock = np.flip(np.copy(monitor._stock), axis = 0)
                params = namedtuple(f"params_monitor_{i}", list(monitor.hrf_kernel.parameters.keys()) + ["stock"], defaults=list(monitor.hrf_kernel.parameters.values()) + [stock])
            elif isinstance(monitor, monitors.Projection):
                mon_noise = None
                if monitor.obsnoise is not None:
                    mon_noise = self._evaluate_noise(monitor.obsnoise, int(nt / (monitor.period / dt)), len(monitor.voi), monitor.gain.shape[0], apply_gfun=False)
                if isinstance(monitor, monitors.EEG):
                    params = namedtuple(f"params_monitor_{i}", ["gain", "ref_vec", "obsnoise"], defaults=(np.array(monitor.gain), np.array(monitor._ref_vec[:, np.newaxis]), mon_noise))
                else:
                    params = namedtuple(f"params_monitor_{i}", ["gain", "obsnoise"], defaults=(np.array(monitor.gain), mon_noise))
                
            
            params_monitors_constructors.append(params)
            params_monitors.append(params())
                
        # The final concrete realization of all parameters
        params = namedtuple('params', ["initial_conditions", "weights", "delay_indices", "dt", "nt", "noise", "params_integrate", "params_monitors"], defaults=(ics(), weights, delay_indices, dt, nt, noise, params_integrate(), params_monitors))
        if return_constructors:
            return params, ics, params_dfun, params_cfun, params_stim, params_integrate, params_monitors_constructors
        else:
            return params()

    def _evaluate_noise(self, noise, length, n_states, n_regions, apply_gfun=True, n_modes = 1):
        nsig = noise.nsig

        # Get the random state from TVB Noise and duplicate it
        ct = noise.random_stream.get_state()
        rng = np.random.RandomState()
        rng.set_state(ct)

        # Sample
        dWt = np.sqrt(noise.dt) * rng.normal(size = (length, n_states, n_regions, n_modes))

        if apply_gfun:
            D = np.sqrt(2 * nsig)
            if len(D.shape) > 1: # if nsig is set per svar TVB expands shape to (n_svar, 1, 1), one short for multiply
                D = D[:,:,:,None]
                return np.einsum('ijkl,jilk->jilk', D, dWt)
            return np.einsum('i,jilk->jilk', D, dWt)

            # return np.transpose(D * np.transpose(dWt, (1, 0, 2, 3)), (1, 0, 2, 3)) # assert right shape for heterogenous nsig
        else:
            return dWt
        
    def get_noise_generator(self, sim):
        """
        Build a function that generates a valid new noise instance for the simulator object. 
        """
        assert sim.integrator.noise is not None, "Simulator must have noise defined"
        length = nt = int(np.ceil(sim.simulation_length/sim.integrator.dt))
        n_nodes = sim.connectivity.number_of_regions
        n_modes = sim.model.number_of_modes
        n_svar = sim.model.nvar
        dt = sim.integrator.noise.dt

        # Get the random state from TVB Noise and duplicate it
        ct = sim.integrator.noise.random_stream.get_state()
        rng = np.random.RandomState()
        rng.set_state(ct)
        nsig = sim.integrator.noise.nsig

        def _noise_generator(length = length, nsig = nsig, dt = dt, n_svar = n_svar, n_nodes = n_nodes, n_modes = n_modes, rng: np.random.RandomState = rng):
            dWt = np.sqrt(dt) * rng.normal(size = (length, n_svar, n_nodes, n_modes))
            
            D = np.sqrt(2 * nsig)
            if len(D.shape) > 1: # if nsig is set per svar TVB expands shape to (n_svar, 1, 1), one short for multiply
                return np.einsum('ijk,jilk->jilk', D, dWt)
            return np.einsum('i,jilk->jilk', D, dWt)
        
        return _noise_generator

        
    def _collect_dfun_params(self, sim):
        return namedtuple("params_dfun", 
                    [f"{par}" for par in sim.model.global_parameter_names + sim.model.spatial_parameter_names],
                    defaults=([getattr(sim.model, par) for par in sim.model.global_parameter_names + sim.model.spatial_parameter_names]))
                    # defaults=([getattr(sim.model, par)[0] for par in sim.model.global_parameter_names] + [np.array(sim.model.spatial_parameter_matrix[i]) for i, _ in enumerate(sim.model.spatial_parameter_names)]))

    def _collect_cfun_params(self, sim):
        return namedtuple("params_cfun", [f"{par}" for par in sim.coupling.parameter_names], 
                         defaults=[getattr(sim.coupling, par)[0] for par in sim.coupling.parameter_names])
    

    def build(self, sim, simulation_length=None, save_at=None, print_source=False, replace_temporal_averaging=False, return_constructors=False, return_new_ics=False, use_tvbo=True, template = None, small_dt = False):
        """
        Build a simulation kernel from a simulator object.

        * `sim`: is a simulator object.
        * `simulation_length`: is the length of time to simulate, default is sim.simulation_length.
        * `save_at`: Specify a path where the simulation will be saved as .py file. 
        * `print_source`: prints the generated source code to stdout.
        * `replace_temporal_averaging`: if False the simulator will use the TemporalAverage monitor in the Bold and EEG monitor as TVB does. If True the simulator will use the SubSample monitor instead, which is faster and gives very similar results. 
        * `return_constructors`: See below
        * `return_new_ics`: if True, the kernel function will return updated initial conditions (current state + history) along with the simulation output. This is of interest if you want continue simulations. The output of kernel will change from result to [result, (current_state, history)]
        * `use_tvbo`: if True, TVBO will be used to add missing fields to the model and coupling classes if possible. 

        Returns a tuple of (kernel, params):
        
        * kernel is the main simulator function.
        * params is a tuple of arguments containing the state of the simulator:
            
        current_state, weights, delay_indices, history, dt, nt, noise, params_integrate, params_monitor

        If return_constructors = True returns also the constructors for the named tuples:

        (kernel, params(), [params, params_dfun, params_cfun, params_stim, params_integrate, params_monitors_constructors])
        """

        try:
            self.check_compatibility(sim)
        except NotImplementedError as e:
            if use_tvbo:
                self._tvbo_enrich_fields(sim)
            
        # update simulation_length - this should not have sideffects on sim
        if simulation_length is not None: 
            sim.simulation_length = simulation_length

        content = dict(sim=sim, np=np, replace_temporal_averaging=replace_temporal_averaging, return_new_ics=return_new_ics, small_dt=small_dt)
        kernel = self.build_py_func(template, content, name="kernel", print_source=print_source, fname=save_at)
        if return_constructors:
            constructors = self.collect_state(sim, simulation_length=simulation_length, return_constructors=True)
            return (kernel, constructors[0](), constructors)
        else:
            params = self.collect_state(sim, simulation_length=simulation_length)
            return (kernel, params)
        
        
    def run(self, sim, simulation_length=None, print_source=False, replace_temporal_averaging=False):
        kernel, args = self.build(sim, simulation_length=simulation_length, print_source=print_source, replace_temporal_averaging=replace_temporal_averaging)
        return kernel(*args)

    def extract_entity(self, sim, entity, simulation_length=None, print_source=False, replace_temporal_averaging=False, template=None, use_tvbo=True):
        """
        Extract intermediate functions from a simulator object, like dfun, cfun, monitors, etc. Useful for testing.
        """
        try:
            self.check_compatibility(sim)
        except NotImplementedError as e:
            if use_tvbo:
                self._tvbo_enrich_fields(sim)
                
        # update simulation_length
        if simulation_length is not None: 
            sim.simulation_length = simulation_length
                
        content = dict(sim=sim, np=np, replace_temporal_averaging=replace_temporal_averaging)
        
        return self.build_py_func(template, content, name=entity, print_source=print_source)
    
    def _tvbo_enrich_fields(self, sim):
        try:
            from tvbo.export import templater
            self.add_fields(sim, templater)
            self.check_compatibility(sim)
        except ModuleNotFoundError as e2:
            raise ModuleNotFoundError(f"Module tvbo not found to automatically add necessary fields. Please install it with 'pip install tvbo'.")
        except AttributeError as e3:
            raise e3
        
    def _add_fields_model(self, model, templater, model_name=None, depth=1):
        if model_name is None:
            model_name = model.__class__.__name__
        try:
            tvbo_model = templater.model2class(model_name, split_nonintegrated_variables=True)
            setattr(model, "coupling_terms", tvbo_model.coupling_terms)
            setattr(model, "state_variable_dfuns", tvbo_model.state_variable_dfuns)
            setattr(model, "parameter_names", tvbo_model.parameter_names)
            # setattr(model, "cvar", tvbo_model.cvar)
        except AttributeError as e:
            # Case where model is not part of tvbo. We first check if the parent model is part of tvbo to enable the typical pattern class ExtendedXYZModel(XYZModel):.
            if depth > 0:
                try:
                    super_model_name = model.__class__.__bases__[0].__name__
                    self._add_fields_model(model, templater, model_name = super_model_name, depth=depth-1)
                except AttributeError as e4:
                    raise AttributeError(f"Model {model.__class__.__name__} is not supported by TVB-O, neither is it parent model {model.__class__.__bases__[0].__name__}.")
                # raise AttributeError(f"Model {model_name} is not supported by TVB-O.") from e
        return model

    def _add_fields_coupling(self, coupling, templater, coupling_name=None):
        if coupling_name is None:
            coupling_name = coupling.__class__.__name__
        try:
            tvbo_coupling = templater.coupling2class(coupling_name)
            setattr(coupling, "parameter_names", tvbo_coupling.parameter_names)
            setattr(coupling, "pre_expr", tvbo_coupling.pre_expr)
            setattr(coupling, "post_expr", tvbo_coupling.post_expr)
        except ValueError as e:
            raise ValueError(f"Coupling {coupling_name} is not supported by TVB-O.") from e
        return coupling

    def add_fields(self, sim, templater, depth=1):
        """
        Add fields containing the necessary information for the templates to the model and coupling classes from TVB-O.

        Adds:
        - model.coupling_terms
        - model.state_variable_dfuns
        - model.parameter_names
        - coupling.parameter_names
        - coupling.pre_expr
        - coupling.post_expr
        """
        try:
            self._check_model(sim.model)
        except NotImplementedError as e:
            print(f"Info: Adding TVB-O fields to Model {sim.model.__class__.__name__}")
            self._add_fields_model(sim.model, templater, depth = depth)

        try:
            self._check_coupling(sim.coupling)
        except NotImplementedError as e:
            print(f"Info: Adding TVB-O fields to Coupling {sim.coupling.__class__.__name__}")
            self._add_fields_coupling(sim.coupling, templater)
        return sim