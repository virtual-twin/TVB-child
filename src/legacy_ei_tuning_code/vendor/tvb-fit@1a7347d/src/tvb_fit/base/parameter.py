from typing import List, Union, NamedTuple
from collections import namedtuple
import numpy as np
from abc import ABC, abstractmethod
import itertools


from jax.tree_util import register_pytree_node_class

class Value(ABC):
    
    @property
    @abstractmethod
    def name(self):
        pass
        
    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass


@register_pytree_node_class
class Parameter(Value):
    def __init__(self, 
                name: str, 
                value, 
                low=None,
                high=None,
                inds=None,
                doc: str = None):  # TODO: clarify possible formats for inds
        self._name = name

        self._value = value # promote to array, so that shape is always defined
        # self._value = np.asarray(value) # promote to array, so that shape is always defined

        self.inds =  inds # defined as selection which parameters are free
        # value + shape None -> scaler
        # scaler value + shape -> vector of shape
        # vector value + shape -> vector of shape + check
        # inds not none + shape -> masked vector

        # TODO: promote shapes if not None    
        self.low = low
        self.high = high
        self.doc = doc
        # self._shape = value.shape
    
    def __str__(self):
        return f"Parameter(name={self._name}, value={self._value}, low={self.low}, high={self.high}, inds={self.inds}, doc={self.doc})"

    def __repr__(self):
        return self.__str__()
        
    @property
    def name(self):
        return self._name
    
    @property
    def value(self):
        return self._value

    @property
    def shape(self):
        return self.value.shape

    def set_shape(self, shape):
        """
        Broadcasts the value to the desired shape if possible. Useful to turn a global parameter local. 
        """
        if self.shape == shape:
            pass
        else:
            try:
                self._value = np.broadcast_to(self.value, shape)
            except:
                raise ValueError(f"Parameter shape {shape} does not match value shape {self.shape} and can not be broadcasted automatically.")

    def set_value(self, value):
        self._value = value

    def tree_flatten(self):
        children = (self._value,)
        aux_data = ((self._name,), (self.low, self.high, self.inds, self.doc))
        # aux_data = ((self._name,), (self.low, self.high, self.inds, self.doc), (self._value.shape,))
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data[0], *children, *aux_data[1])

    def assert_shapes(self):
        if self.inds is not None:
            pass
            # TODO: find out if the indices shape corresponds to the shape
        # TODO: similarly for value, low and high, the shapes should be either the same or compatible

        
# @register_pytree_node_class
# class JaxParameter(Parameter):
#     def __init__(self, superclass_instance=None):
#         if superclass_instance is not None:
#             # Copy the state from the superclass instance
#             for attr, value in vars(superclass_instance).items():
#                 setattr(self, attr, value)
        
#     def __str__(self):
#         return f"JaxParameter(name={self.name}, shape={self.shape}, value={self.value}, low={self.low}, high={self.high}, inds={self.inds}, doc={self.doc})"

#     def tree_flatten(self):
#         children = (self.value,)
#         aux_data = ((self.name, self.shape), (self.low, self.high, self.inds, self.doc))
#         return (children, aux_data)

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*aux_data[0], *children, *aux_data[1])

@register_pytree_node_class
class Parameters(ABC):
    def __init__(self, parameters: Union[List[Parameter], NamedTuple, Parameter]):
        if isinstance(parameters, Parameter):
            parameters = [parameters]
        if isinstance(parameters, list):
            parameters = namedtuple(f'{self.__class__.__name__}', [p.name for p in parameters])(*parameters)
        self.items = parameters

    def __str__(self):
        sl = '\n'.join([f'{"└─" if i == len(self.items) -1  else "├─"} {name}:\t {value.value} ' for i, (name, value) in enumerate(self.items._asdict().items())])
        return f"{len(self.items)} {self.__class__.__name__ if len(self.items) > 1 else self.__class__.__name__[:-1]}\n{sl}"
    
    def __repr__(self):
        return self.__str__()
    
    def __getstate__(self):
        return self.items._asdict().copy()

    def __setstate__(self, state):
        self.items = namedtuple(f'{self.__class__.__name__}', state.keys())(*state.values())
    
    def tree_flatten(self):
        # this should be a 1D array?
        p_flats = [p.tree_flatten() for p in self.items]
        children = tuple(zip(*p_flats))[0]
        # children need to be a single flat vector
        
        aux_data = tuple(zip(*p_flats))[1]
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        items = [Parameter.tree_unflatten(aux_data[i], children[i]) for i in range(len(aux_data))]
        return cls(items)
    
    def __getattr__(self, name):
        return getattr(self.items, name)

    def __setattr__(self, name, value):
        if name == "items":
            super().__setattr__(name, value)
            return None

        if name in self.items._fields:
            update_dict = {name: value}
            assert type(value) == type(getattr(self.items, name)), f"Value needs to be of type: {type(getattr(self.items, name))}"
            assert name == value.name, f"Value needs to have the same name as field: {name} has: {value.name}. Maybe you want to create new Parameters?"
            self.items = self.items._replace(**update_dict)
        else:
            super().__setattr__(name, value)

        return None
    
    def as_salib_problem(self):
        problem = dict()
        problem['num_vars'] = len(self.items)
        names = []
        bounds = []
        for _p in self.items:
            if np.prod(_p.shape) > 1:
                raise ValueError("Currently only scalar Parameters are supported for SALib sampling.")
            if _p.low is None or _p.high is None:
                raise ValueError("Bounds low and high need to be defined.")
            names.append(_p.name)
            bounds.append([_p.low, _p.high])
            
        problem['names'] = names
        problem['bounds'] = bounds
        return problem

    def copy(self   ):
        return copy.deepcopy(self)

    def __getitem__(self, idx):
        return self.items[idx]
    
    def __len__(self):
        return len(self.items)
    
    def append_parameter(self, parameter: Parameter):
        ext_params = namedtuple('Parameters', self.items._fields + (parameter.name,))
        self.items = ext_params(*self.items, **{parameter.name: parameter})

    def append_parameters(self, parameters):
        ext_params = namedtuple('Parameters', self.items._fields + parameters._fields)
        self.items = ext_params(*self.items, *parameters)

    def append(self, parameter):
        if isinstance(parameter, Parameter):
            self.append_parameter(parameter)
        elif isinstance(parameter, Parameters):
            self.append_parameters(parameter)
        else:
            raise ValueError("Argument parameter is neither of class Parameter nor of Parameters!")

    def get_params_from_path(self, param_type):
        return {pname.split(param_type)[-1]: pval for pname, pval in self.items._asdict().items() if param_type in pname}

    def get_model_params(self):
        return self.get_params_from_path("model_")

    def get_coupling_params(self):
        return self.get_params_from_path("coupling_")

    def get_stimulus_params(self):
        return self.get_params_from_path("stimulus_")
    
    # def get_integrator_params(self):
        # return self.get_params_from_path("integrator")

    def get_monitor_params(self, id=0):
        raw = self.get_params_from_path(f"monitor_{id}_")
        # Remove Bold and other suffixes
        return {"_".join(pname.split("_")[1:]) : pval for pname, pval in raw.items()}

    def get_observation_model_params(self):
        return self.get_params_from_path("observation_")

    # def get_main_params(self):
    #     main_names = 

import copy
def update_sim(sim, params):
    """
    Return a new simulator with updated model and coupling parameters
    """
    sim_new = copy.deepcopy(sim)
    def save_set(entity, name, value):
        try:
            setattr(entity, name, np.asarray(value))
        except Exception as e:
            print(f"Warning: Failed to update parameter {name} with value {value} due to {e}")
    # Model
    p_model = params.get_model_params()
    for i, (name, p) in enumerate(p_model.items()):
        save_set(sim_new.model, name, p.value)

    # Coupling
    p_coupling = params.get_coupling_params()
    for i, (name, p) in enumerate(p_coupling.items()):
        save_set(sim_new.coupling, name, p.value)

    sim_new.configure()
    return sim_new

class ParameterSpace():
    def __init__(self, parameter: Union[Parameter, Parameters], n = 1):
        if isinstance(parameter, Parameter):
            parameter = [parameter]
        self.parameters = Parameters(parameter)

class GridParameterSpace(ParameterSpace):
    """
    An iterator over a Parameter or Parameters on a grid. The grid bounds are defined by `low` and `high` for each parameter.

    * `parameters`: Parameter or Parameters
    * `n`: number of steps for all parameters if scalar (total number of combinations = n**len(parameters)) or a collection len(n) == len(parameters)
    """ 
    def __init__(self, parameter: Union[Parameter, Parameters], n = 1):
        super().__init__(parameter)
        # if n is scalar
        if isinstance(n, int):
            n = [n] * len(parameter)
        else:
            assert len(n) == len(parameter) , f"len(n) {len(n)} must be equal to len(parameter) {len(parameter)}"
        self.n = n
        self.N = np.prod(self.n)
        self.axes = []
        for p, _n in zip(self.parameters.items, self.n):
            assert hasattr(p, "low") and p.low != None and hasattr(p, "high") and p.high != None, f"Parameter {p.name} must have attributes 'low' and 'high', that are not None"
            if _n > 1:
                self.axes.append(np.linspace(p.low, p.high, _n))
            else:
                self.axes.append([p.value])
        values, aux = self.parameters.tree_flatten()
        self.aux = aux
        self.values = values

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        sl = '\n'.join([f'{"└─" if i == len(self.parameters.items) -1  else "├─"} {name}:\t low = {value.low}, high = {value.high}, n = {self.n[i]} ' for i, (name, value) in enumerate(self.parameters.items._asdict().items())])
        return f"{self.__class__.__name__} with {self.N} combinations\n{sl}"

    def __len__(self):
        return self.N

    def __iter__(self):
        self.iter = itertools.product(*self.axes)
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < self.N:
            self.i += 1
            next_values = self.iter.__next__()
            new_payload = [(np.full_like(p[0], v), ) for p, v in zip(self.values, next_values)]
            return self.parameters.tree_unflatten(self.aux, new_payload)
        else:
            raise StopIteration

from SALib.sample import sobol
class SALibParameterSpace(ParameterSpace):
    """
    Use sampler from SALib to generate low discrepancy samples for Parameters, based on the bounds defined by low and high values.

    * `parameters`: Parameter or Parameters
    * `n`: The number of samples to generate. Ideally a power of 2. The total number of parameters will be much higher depending on the sampler, e.g. for sobol N = n(D+1) 
    * `sampler`: the sampler from SALib, further kwargs can be passed to the sampler, see https://salib.readthedocs.io/en/latest/index.html for more information on samplers.
    """
    def __init__(self, parameters: Union[Parameter, Parameters], n = 1, sampler = sobol, **kwargs):
        super().__init__(parameters)
        self.n = n
        _, aux = self.parameters.tree_flatten()
        self.aux = aux
        self.problem = parameters.as_salib_problem()
        self.values = sampler.sample(self.problem, n, **kwargs)
        self.N = self.values.shape[0]
    
    def __repr__(self):
            return self.__str__()

    def __str__(self):
        sl = '\n'.join([f'{"└─" if i == len(self.parameters.items) -1  else "├─"} {name}:\t low = {value.low}, high = {value.high}, n = {self.n[i]} ' for i, (name, value) in enumerate(self.parameters.items._asdict().items())])
        return f"{self.__class__.__name__} with {self.N} combinations\n{sl}"

    def __len__(self):
        return self.N
    
    def __iter__(self):
        self.i = 0
        return self
        
    def __next__(self):
        if self.i < self.N:
            new_payload = ([(np.array([d]), ) for d in self.values[self.i,:]])
            self.i += 1
            return self.parameters.tree_unflatten(self.aux, new_payload)
        else:
            raise StopIteration

class SampleUniformParameterSpace(ParameterSpace):
    def __init__(self, parameters: Union[Parameter, Parameters], n = 1, seed = 0):
        super().__init__(parameters)
        self.n = n
        self.seed = seed
        self.seeds = [seed + i for i in range(len(self.parameters))]
        values, aux = self.parameters.tree_flatten()
        self.aux = aux
        self.values = values
        self.highs = [p.high for p in self.parameters]
        self.lows = [p.low for p in self.parameters]
        self.shapes = [value[0].shape for value in values]

    def __len__(self):
        return self.n
    
    def __iter__(self):
        self.rngs = [np.random.default_rng(seed=seed) for seed in self.seeds]

        # self.data =  [np.reshape(r * sob.random(self.n) + l, (self.n,) + shape) for sob, l, r, shape in zip(self.sobols, self.lows, self.ranges, self.shapes)]
        self.i = 0
        return self
        
    def __next__(self):
        if self.i < self.n:
            new_payload = ([(rng.uniform(low, high, shape), ) for rng, low, high, shape in zip(self.rngs, self.lows, self.highs, self.shapes)])
            self.i += 1
            return self.parameters.tree_unflatten(self.aux, new_payload)
        else:
            raise StopIteration

    
    pass

import jax
from tvb_fit.base.prior import NumPyroPriors
class PriorParameterSpace(ParameterSpace):
    def __init__(self, parameters: Union[Parameter, Parameters], n = 1, seed = 0, informative = False):
        super().__init__(parameters)
        self.n = n
        self.seed = seed
        values, aux = self.parameters.tree_flatten()
        self.aux = aux
        self.values = values
        self.priors = NumPyroPriors.from_parameters(self.parameters, informative = informative)

    def __len__(self):
        return self.n
    
    def __iter__(self):
        if type(self.seed) == jax.random.PRNGKey:
            self.key = self.seed
        else:
            self.key = jax.random.PRNGKey(self.seed)
        self.i = 0
        return self
        
    def __next__(self):
        if self.i < self.n:
            self.key, subkey = jax.random.split(self.key)
            new_payload = self.priors.sample(num_samples=1, rng_key=subkey)
            self.i += 1
            return self.parameters.tree_unflatten(self.aux, new_payload)
        else:
            raise StopIteration

    
    pass