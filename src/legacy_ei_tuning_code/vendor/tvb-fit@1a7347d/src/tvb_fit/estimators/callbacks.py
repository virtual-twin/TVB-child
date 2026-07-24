import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# stop, parameters, ics, metadata = self.callback(i, parameters, ics, metadata, fitting_data, self.gm, prediction, loss_value, grads)

class AbstractCallback:
    def __init__(self, every=1) -> None:
        self.every = every
    def __call__(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        if i % self.every == 0:
            return self.do(i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads)
        return False, parameters, ics, metadata
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        return False, parameters, ics, metadata

# class UpdateObservationModelCallback(AbstractCallback):
#     def __init__(self, om_builder, every=1) -> None:
#         super().__init__(every)
#         self.om_builder = om_builder
#     def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
#         om, ics_new = self.om_builder(gm, parameters, ics)
#         gm.observation_model = om
#         def simulate(parameters, ics):
#             result, ics_new = gm.kernel(parameters, ics)
#             prediction = om.operation(result, parameters)
#             return prediction, ics_new
#         return False, parameters, ics_new, metadata

class UpdateICsAndMetadataCallback(AbstractCallback):
    def __init__(self, updater, every=1) -> None:
        super().__init__(every)
        self.updater = updater
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        metadata_new, ics_new = self.updater(gm, parameters, ics)
        return False, parameters, ics_new, metadata_new
            
class SavingCallback(AbstractCallback):
    def __init__(self, every=1, key="", save_fun=lambda *args: None) -> None:
        super().__init__(every)
        self.key = key
        self.save_fun = save_fun

    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        if self.key not in fitting_data:
            fitting_data[self.key] = pd.DataFrame(columns=["step", "save"])

        fitting_data[self.key].loc[len(fitting_data[self.key])] = [i, self.save_fun(i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads)] 
        return False, parameters, ics, metadata
    
class SaveBestSeenCallback(AbstractCallback):
    def __init__(self, every=1, key="best", minimization=True) -> None:
        super().__init__(every)
        self.key = key
        self.minimization = minimization

    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        _loss_value = loss_value
        if not self.minimization:
            _loss_value *= -1
        if self.key not in fitting_data:
            fitting_data[self.key] = (loss_value, i, parameters, ics, metadata)
        elif _loss_value < fitting_data[self.key][0]:
            fitting_data[self.key] = (loss_value, i, parameters, ics, metadata)
        return False, parameters, ics, metadata
                

class DefaultPrintCallback(AbstractCallback):
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        print(f"Step {i}: {loss_value:6f}")
        return False, parameters, ics, metadata

class PrintParametersCallback(AbstractCallback):
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        print(f"Parameters at Step {i}: {loss_value:6f}")
        for p in parameters:
            print(f"{p}")
        return False, parameters, ics, metadata

class PrintGlobalParametersCallback(AbstractCallback):
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        print(f"Parameters at Step {i}: {loss_value:6f}")
        for p in parameters:
            if np.prod(p.shape) == 1: 
                print(f"{p}")
        return False, parameters, ics, metadata    

class PrintGradsCallback(AbstractCallback):
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        print(f"Grads at Step {i}: {loss_value:6f}")
        for g in grads:
            print(f"{g}")
        return False, parameters, ics, metadata  
        
class StopLossCallback(AbstractCallback):
    def __init__(self, every=1, stop_loss=0) -> None:
        super().__init__(every)
        self.stop_loss = stop_loss
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        if loss_value < self.stop_loss:
            print(f"Stopped at step {i} with loss {loss_value:6f}")
            return True, parameters, ics, metadata
        return False, parameters, ics, metadata

class StopConvergenceCallback(AbstractCallback):
    """
    Stop fitting if no improvement was seen for `patience` number of iterations. Improvement is defined by loss_new < loss_best - `min_delta`.
    """
    def __init__(self, every=1, patience = 10, min_delta = 10e-4) -> None:
        super().__init__(every)
        self.patience = patience
        self.min_delta = min_delta
        self.patience_count = 0
        self.best_loss = np.Inf

    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        if i == 0: # Reset on first iteration
            self.patience_count = 0
            self.best_loss = np.Inf

        if loss_value < self.best_loss - self.min_delta:
            self.best_loss = loss_value
            self.patience_count = 0  
        else:
            self.patience_count += 1
        
        converged = self.patience_count > self.patience
        if converged:
            print(f"Stopped at step {i} with loss {loss_value:6f} due to no improvement after {self.patience} steps")

        return converged, parameters, ics, metadata

class PlotObservationCallback(AbstractCallback):
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        plt.figure()
        plt.plot(prediction[0].T, color = 'k', alpha = 0.25)
        # plt.imshow(prediction[0].T)
        plt.show()

        return False, parameters, ics, metadata
    
class PlotTSCallback(AbstractCallback):
    """
    Plots the time series of the simulator without applying the observation model
    """
    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        # could be forwarded from inference to save double computing
        res = gm.kernel(gm.preprocess(parameters), ics)[0]
        has_multi_monitors = not hasattr(res, "time")
        if has_multi_monitors:
            n_mon = len(res)
            n_svar = res[0].trace.shape[1]
        else:
            n_mon = 1
            n_svar = res.trace.shape[1]

        _, axs = plt.subplots(n_svar, n_mon)
        try:
            axs_flat = axs.flatten()
        except:
            axs_flat = [axs]
        for m in range(n_mon):
            for s in range(n_svar):
                ax = axs_flat[m+(s*(n_mon))]
                if has_multi_monitors:
                    ax.plot(res[m].time, res[m].trace[:, s, :, 0], alpha = 0.1)
                else:
                    ax.plot(res.time, res.trace[:, s, :, 0], alpha = 0.1)
                if s == 0:
                    ax.set_title(f'Monitor {m}')
                if m == 0:
                    ax.set_ylabel(f'state variable {s}')
                if s == n_svar-1:
                    ax.set_xlabel('time')
             
        plt.tight_layout()
        plt.show()
        return False, parameters, ics, metadata
    
# class PlotTSDiffCallback(AbstractCallback):
#     def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
#         res = gm(parameters)
#         _, ax = plt.subplots(1, 3, figsize=(20,4))
#         ax[0].plot(jnp.squeeze(target))
#         ax[0].set_title(f'Target')
#         ax[1].plot(jnp.squeeze(res))
#         ax[1].set_title(f'Prediction at step {i} with loss {loss_value:6f}')
#         ax[1].sharey(ax[0])
#         ax[2].plot(jnp.squeeze(target - res))
#         ax[2].set_title(f'Difference')
#         plt.tight_layout()
#         plt.show()
#         return False, parameters, ics, metadata
    
class MultiCallback(AbstractCallback):
    def __init__(self, callbacks, every = 1) -> None:
        self.callbacks = callbacks
        self.every = every

    def do(self, i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads):
        for callback in self.callbacks:
            test, parameters, ics, metadata = callback(i, parameters, ics, metadata, fitting_data, gm, prediction, loss_value, grads)
            if test:
                return True, parameters, ics, metadata
        return False, parameters, ics, metadata
