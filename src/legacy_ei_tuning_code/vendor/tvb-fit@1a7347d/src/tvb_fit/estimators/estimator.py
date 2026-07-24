from tvb_fit.base.gen_model import GenerativeModel

class Estimator:

    def __init__(
            self,
            gm: GenerativeModel,
            observation = None
    ):
        self.gm = gm
        self.observation = observation
    
    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def validate(self):
        pred, _ = self.gm.run()
        assert self.observation.shape == pred.shape, f"Observation shape {self.observation.shape} does not match prediction shape {pred.shape}" 
        return None

    def fit(self):
        pass

class PointEstimator(Estimator):

    def __init__(self, 
                 gm: GenerativeModel, 
                 observation = None, 
                 metric = None,
                 callback = None,
                 optimizer = None
                 ):
        super().__init__(gm, observation)
        self.metric = metric
        self.callback = callback
        self.optimizer = optimizer

    def validate(self):
        pred, params = self.gm.run()
        assert isinstance(self.metric(self.observation, pred, params, params), float), f"Metric must return a scalar float"        
        return 
    
    def fit(self):
        pass