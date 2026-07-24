import numpy 
identity = lambda x, y: x

class ObservationModel:
    """
    Observation model (simulation, parameters) -> observation
    """
    def __init__(self, transforms = [], metadata = None):
        self.transforms = transforms
        self.metadata = metadata

    def __call__(self, sim_result, parameters):
        # chain transforms
        return self.with_metadata(sim_result, parameters, self.metadata)
    
    def with_metadata(self, prediction, parameters, metadata):
        return self.operation(prediction, parameters, metadata)
    
    def build(self):
        def operation(*input):
            for transform in self.transforms:
                input = transform(*input)
            return input
        self.operation = operation
        return operation


    def validate(self, time, state, regions, monitors, parameters):
        return True

# becomes multi observation?
class SplitCombine():
    """
    Apply multiple cost functions over (trace, parameters) and aggregate the weighted results.
    """
    def __init__(self, cost_functions, aggregation_function, weights = None):
        self.cost_functions = cost_functions
        self.aggregation_function = aggregation_function
        if weights is not None:
            assert len(weights) == len(cost_functions), f"Cost functions (length {len(cost_functions)}) and weights (length {len(weigths)}) must have same length"
            self.weights = weights
        else:
            self.weights = jnp.ones(len(cost_functions))
    def __call__(self, prediction, parameters):
        results = []
        for fun, w in zip(self.cost_functions, self.weights):
            result = w * fun(prediction, parameters)
            results.append(result)
        return self.aggregation_function(jnp.array(results))
 
# base.transforms
class AbstractTransform:
    def __init__(self, np = numpy):
        self.np = np
    def __call__(self, prediction, parameters, metadata):
        return prediction, parameters, metadata
        
class SelectMonitor(AbstractTransform):
    """
    Selects monitor from a simulation result with multiple monitors. Only necessary for multiple monitors.
    """
    def __init__(self, mon_idx = 0):
        self.mon_idx = mon_idx
    def __call__(self, prediction, parameters, metadata):
        prediction = prediction[self.mon_idx]
        return prediction, parameters, metadata
    
class SelectTrace(AbstractTransform):
    """ 
    Selects trace from timeseries tuple (time, trace). You might want to use SelectMonitor first. 
    """   
    def __call__(self, prediction, parameters, metadata):
        prediction = prediction.trace
        return prediction, parameters, metadata

class Subset(AbstractTransform):
    """
    Take a subset from the trace of the prediction. You want to use SelectTrace first.
    """
    def __init__(self, idx_time = None, idx_state = None, idx_region = None, idx_mode = None, np = numpy):
        self.np = np
        if idx_time is None:
            idx_time = self.np.s_[:]
        if idx_state is None:
            idx_state = self.np.s_[:]
        if idx_region is None:
            idx_region = self.np.s_[:]
        if idx_mode is None:
            idx_mode = self.np.s_[:]
        self.idx_time = idx_time
        self.idx_state = idx_state
        self.idx_region = idx_region
        self.idx_mode = idx_mode
    def __call__(self, prediction, parameters, metadata):
        prediction = prediction[self.idx_time, self.idx_state, self.idx_region, self.idx_mode]
        return prediction, parameters, metadata

class TransformPrediction(AbstractTransform):
    """
    Perform a transformation on the trace.
    """
    def __init__(self, fun):
        self.fun = fun
    def __call__(self, prediction, parameters, metadata):
        prediction = self.fun(prediction)
        return prediction, parameters, metadata

class TransformParameters(AbstractTransform):
    """
    Perform a transformation on the parameters.
    """
    def __init__(self, fun):
        self.fun = fun
    def __call__(self, prediction, parameters, metadata):
        parameters = self.fun(parameters)
        return prediction, parameters, metadata

class Transform(AbstractTransform):
    """
    Perform a transformation on prediction, parameters and metadata.
    """
    def __init__(self, fun):
        self.fun = fun
    def __call__(self, prediction, parameters, metadata):
        prediction, parameters, metadata = self.fun(prediction, parameters, metadata)
        return prediction, parameters, metadata
    
class Aggregate(AbstractTransform):
    """
    Aggregate prediction, parameters and metadata to a single scalar value.
    """
    def __init__(self, fun):
        self.fun = fun
    def __call__(self, prediction, parameters, metadata):
        agg = self.fun(prediction, parameters, metadata)
        return agg

