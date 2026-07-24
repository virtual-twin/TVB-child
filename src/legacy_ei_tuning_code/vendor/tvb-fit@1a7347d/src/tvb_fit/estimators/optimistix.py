import jax
import optimistix as optx

from tvb_fit.estimators.estimator import PointEstimator

class OptimistixPointEstimator(PointEstimator):

    def fit(self, iter_max = 1, initial_parameters = None):
        """
        Todo and test
        """
        @jax.jit
        def simulate(parameters, ics, metadata):
            result, ics_new = self.gm.kernel(self.gm.preprocess(parameters), ics)
            prediction = self.gm.observation_model.with_metadata(result, parameters, metadata)
            return prediction, ics_new

        @jax.jit
        def loss(parameters, args):
            print(parameters)
            ics, metadata = args
            prediction, ics_new = simulate(parameters, ics, metadata)
            loss_value = self.metric(prediction, self.observation, parameters, self.gm.params)
            return (loss_value, (prediction, ics_new))
        
        if initial_parameters is None:
            parameters = self.gm.params
        else:
            parameters = initial_parameters

        ics = self.gm.initial_conditions
        metadata = self.gm.observation_model.metadata

        return optx.minimise(loss, self.optimizer, parameters, args = (ics, metadata), has_aux = True, max_steps = iter_max)
