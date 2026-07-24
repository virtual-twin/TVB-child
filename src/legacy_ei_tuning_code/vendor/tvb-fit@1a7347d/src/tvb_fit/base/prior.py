from typing import List, Union, NamedTuple
from numpy.typing import ArrayLike

from tvb_fit.base.parameter import Value

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpyro.distributions as ndist
import numpyro

import pymc as pm

from tvb_fit.base.parameter import Parameter, Parameters

class AbstractPrior(Value):
    def __init__(self, name: str, dist= None, inds=None, doc: str = None):
        self._name = name
        self.dist = dist
        self.inds = inds
        self.doc = doc
        
    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, dist={self.dist.__class__.__name__}, inds={self.inds})"
    
    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        return self._name
    
    @property
    def value(self):
        return self.sample(1)

    @property
    def shape(self):
        raise NotImplemented

    def sample(self, num_samples: int):
        raise NotImplemented


    def sample_to_numpy(self, num_samples: int):
        raise NotImplemented


    def sample_parameter(self):
        return Parameter(self.name, self.value, shape = self.shape, doc = self.doc)
        
class PriorBuilder():
    """
    * TruncatedNormal: AbstractPrior("name", 0, 1, shape = (2,3), informative=True, low = -1, high = 1) 
    
    * Normal: AbstractPrior("name", 0, 1, shape = (2,3), informative=True) 
    
    * Uniform: AbstractPrior("name", 0, 1, shape = (2,3), informative=False, low = -1, high = 1) 
    
    * Uniform: AbstractPrior("name", 0, 1, shape = (2,3), informative=False) 
    """
    def __init__(self, 
                    name: str,
                    loc: Union[float, int, ArrayLike],
                    scale: Union[float, int, ArrayLike],
                    shape: tuple = None,
                    informative: bool = False,
                    low = None,
                    high = None,
                    inds = None,
                    doc: str = None):
        self.name = name
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.informative = informative
        self.low = low
        self.high = high
        self.inds = inds
        self.doc = doc
        self.configure()
                    
    def configure(self):
        # Assure loc and scale have a shape
        if isinstance(self.loc, list):
            self.loc = np.array(self.loc)
        if isinstance(self.scale, list):
            self.scale = np.array(self.scale)
        if isinstance(self.loc, (float, int)):
            self.loc = np.array([self.loc])
        if isinstance(self.scale, (float, int)):
            self.scale = np.array([self.scale])
        
        if self.scale is None:
            self.scale = np.copy(self.loc)
        
        if self.shape is None:
            # take the maximum shape from loc and scale
            self.shape = max(self.loc.shape, self.scale.shape)
            self.loc = self.loc
            self.scale = self.scale
        else:
            if self.shape > max(self.loc.shape, self.scale.shape):
                print(self.shape)
                try:
                    self.scale = np.broadcast_to(self.scale, self.shape)
                    self.loc = np.broadcast_to(self.loc, self.shape)
                except ValueError:
                    raise ValueError(f" Could not broadcast {self.shape} to {self.loc.shape} and {self.scale.shape}")
            else:
                self.shape = max(self.loc.shape, self.scale.shape)

    @classmethod
    def from_parameter(cls, parameter: Parameter, scale = 1.0, informative = False):
        return cls(parameter.name, parameter.value, scale,
                shape = parameter.shape,
                informative = informative, 
                low = parameter.low,
                high = parameter.high,
                inds = parameter.inds,
                doc = parameter.doc)

    def prepare_low_high(self):
        if self.low is not None and self.high is not None:
            low, high = self.low, self.high
        elif self.low is None and self.high is not None:
            low = self.high - self.scale
            low, high = low, self.high
        elif self.low is not None and self.high is None:
            high = self.low + self.scale
            low, high = self.low, high
        else:
            low, high = self.loc - self.scale / 2, self.loc + self.scale / 2
        try:
            low = np.broadcast_to(low, self.shape)
            high = np.broadcast_to(high, self.shape)
            return low, high
        except ValueError:
            raise ValueError(f" Could not broadcast {self.shape} to {self.low.shape} and {self.high.shape}")
    
    def as_Prior(self, type):
        if type.__name__ == NumPyroPrior.__name__:
            return self.as_NumPyroPrior()
        elif type.__name__ == PyMCPrior.__name__:
            return self.as_PyMCPrior()
        else:
            raise ValueError(f"Type {type} not supported!")
                
    def as_NumPyroPrior(self):
        if self.informative:
            # Returns Normal automatically if low and high are None
            d = ndist.TruncatedNormal(loc=self.loc, scale=self.scale, low=self.low, high=self.high)
        else:
            low, high = self.prepare_low_high()
            d = ndist.Uniform(low=low, high=high)
        return NumPyroPrior(self.name, d, self.inds)
        pass

    def as_PyMCPrior(self):
        if self.informative:
            _d = pm.Normal.dist(mu=self.loc, sigma=self.scale)
            if self.low is None and self.high is None:
                d = _d
            else:
                d = pm.Truncated.dist(_d, lower=self.low, upper=self.high)
        else:
            low, high = self.prepare_low_high()
            d = pm.Uniform.dist(lower=low, upper=high)
        return PyMCPrior(self.name, d, self.inds)

class NumPyroPrior(AbstractPrior):

    @property
    def value(self, rng_key = PRNGKey(42)):
        return self.sample(num_samples = 1, rng_key=rng_key)
    
    @property
    def shape(self):
        return self.dist.shape()

    def sample(self, num_samples: int = 1, rng_key = PRNGKey(42), **kwargs):
        return numpyro.sample(self.name, self.dist, rng_key=rng_key, sample_shape=(num_samples,), **kwargs)          

    def plot(self, n = 2024, n_scale = 5):
        if self.dist.shape() == () or self.dist.shape() == (1,):
            if isinstance(self.dist, (ndist.TwoSidedTruncatedDistribution, ndist.LeftTruncatedDistribution, ndist.RightTruncatedDistribution)):
                if hasattr(self.dist, "high") and hasattr(self.dist, "low"):
                    x = jnp.linspace(1.3 * self.dist.low, 1.3 * self.dist.high, n)
                elif hasattr(self.dist, "low"):
                    x = jnp.linspace(n_scale * self.dist.low, -n_scale * self.dist.low, n)
                elif hasattr(self.dist, "high"):
                    x = jnp.linspace(-1.1 * self.dist.high, 1.1 * self.dist.high, n)
                else:
                    x = jnp.linspace(-1.1 * self.dist.mean, 1.1 * self.dist.mean, n)
                    
                pdf_values = jnp.exp(jnp.squeeze(self.dist.log_prob(x)))
            else:
                x = jnp.linspace(- n_scale * self.dist.variance + self.dist.mean, n_scale * self.dist.variance + self.dist.mean, n)
                # x = jnp.linspace(1, 10, n)
                pdf_values = jnp.exp(jnp.squeeze(self.dist.log_prob(x)))
            dx = x[1] - x[0]
            nrm = jnp.sum(dx * pdf_values[np.where(~np.isnan(pdf_values))])
            f, ax = plt.subplots(figsize=(4, 4));
            ax.plot(x, pdf_values, label=f'\u2211 PDF: {nrm:.2f}');
            ax.set_yticks([]);
            ax.set_title(f'PDF of {self.dist.__class__.__name__} Distribution');
            ax.set_xlabel(f'{self.name}');
            ax.set_ylabel('Probability Density');
            # ax.legend();
            f.tight_layout();
            return f
        else:
            print(f"Not implemented for multivariate distributions. Shape = {self.dist.shape()}")

class PyMCPrior(AbstractPrior):
    pass
            
class TorchPrior(AbstractPrior):
    pass


class Priors(Parameters):
    def __str__(self):
        sl = '\n'.join([f'{"└─" if i == len(self.items) -1  else "├─"} {name}:\t {value.dist.__class__.__name__} ' for i, (name, value) in enumerate(self.items._asdict().items())])
        return f"{len(self.items)} {self.items[0].__class__.__name__}{'s' if len(self.items) > 1 else ''}\n{sl}"
            
    @classmethod
    def priors_from_parameters(cls, parameters, type = NumPyroPrior, informative = False):
        priors = []
        for param in parameters:
            pb = PriorBuilder.from_parameter(param, informative = informative)
            prior = pb.as_Prior(type)
            priors.append(prior)
        return priors
    
    @classmethod
    def from_parameters(cls, parameters, type = NumPyroPrior, informative = False):
        return cls(Priors.priors_from_parameters(parameters, type = type, informative = informative))

    def sample(self, num_samples: int):
        return [p.sample(num_samples) for p in self.items]

    def sample_to_numpy(self, num_samples: int):
        return np.array(self.sample(num_samples))

class NumPyroPriors(Parameters):   
    @classmethod
    def from_parameters(cls, parameters, informative = False):
        return cls(Priors.priors_from_parameters(parameters, type = NumPyroPrior, informative = informative))

    def sample(self, num_samples: int, rng_key = PRNGKey(42)):
        res = []
        for p in self.items:
            rng_key, subkey = jax.random.split(rng_key)
            res.append(p.sample(num_samples, rng_key = subkey))
        return res

    def sample_to_numpy(self, num_samples: int):
        return np.array(self.sample(num_samples))