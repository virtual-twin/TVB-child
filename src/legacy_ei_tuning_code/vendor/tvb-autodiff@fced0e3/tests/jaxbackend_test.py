# -*- coding: utf-8 -*-

"""
Tests for the JAX backend.

.. moduleauthor:: Marius Pille <marius.pille@bih-charite.de>

"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import test_util

from collections import namedtuple

from tvb_autodiff.jax import JaxBackend
from tvb.simulator.coupling import Sigmoidal, Linear, Difference, SigmoidalJansenRit
from tvb.simulator.integrators import (EulerDeterministic, EulerStochastic,
    HeunDeterministic, HeunStochastic, IntegratorStochastic, 
    RungeKutta4thOrderDeterministic, Identity, IdentityStochastic,
    VODEStochastic)
from tvb.simulator.noise import Additive, Multiplicative
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw, RawVoi, TemporalAverage, SubSample, Bold, EEG, MEG
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.basic.neotraits.api import List
from tvb.simulator.lab import *

from tests.backendtestbase import (BaseTestSim, BaseTestCoupling, BaseTestDfun,
    BaseTestIntegrate, BaseTestMonitors, BaseIntegrationTest)

class TestJaxBackend(BaseIntegrationTest):
    def test_jax_backend(self):
        JB = JaxBackend()
        for model in JB._supported_models:
            for integrator in JB._supported_integrators:
                for coupling in JB._supported_couplings:
                    for monitor in JB._supported_monitors:
                        for delay in [True, False]:
                            for n_modes in [1, 2]:
                                with self.subTest(model=model, integrator=integrator, coupling=coupling, monitor=monitor, delay=delay, n_modes=n_modes):
                                    self._test_sim(model, integrator, coupling, monitor, delay, n_modes)

    def _test_sim(self, model, integrator, coup, monitor, delay, n_modes):
        # Broken combination
        broken_combinations = [
            set([coupling.Kuramoto]), # can cause spiking which ruins the test if one timestep of, generally works
            set([SigmoidalJansenRit, models.Kuramoto]), # not enough state vars
            set([SigmoidalJansenRit, models.Generic2dOscillator]), # not enough state vars
            set([SigmoidalJansenRit, models.Linear]), # not enough state vars
            set([SigmoidalJansenRit, models.ReducedWongWangExcInh]), # not enough state vars
            set([SigmoidalJansenRit, models.ReducedWongWang]), # not enough state vars
            set([SigmoidalJansenRit, models.LarterBreakspear]), # not enough state vars
            set([EulerStochastic, EEG]), # to much change for reasonable tests
            set([EulerStochastic, MEG]), # to much change for reasonable tests
            set([HeunStochastic, EEG]), # to much change for reasonable tests
            set([2, EEG]), # TVB EEG (Projection) does not work with n_modes != 1
            set([2, MEG]), # TVB MEG (Projection) does not work with n_modes != 1
            set([MEG]), # TVB MEG somehow fails on configure
            set([HeunStochastic, MEG]), # to much change for reasonable tests
            set([Identity]), # Not useful for DEs
            set([IdentityStochastic]), # Not useful for DEs
        ]
        is_broken = any([broken_combination.issubset(set([model, integrator, coup, monitor, delay, n_modes])) for broken_combination in broken_combinations])

        is_numba = issubclass(model, models.base.ModelNumbaDfun) # Numba dfuns fail for n_modes != 1
        
        if is_broken or (is_numba and n_modes != 1):
            return None
        
        sim = self._prep_sim(model, integrator, coup, monitor, delay = delay, n_modes = n_modes)
        kernel, params = JaxBackend().build(sim, print_source=True)
        (th, yh), = kernel(*params)
        (t, y), = sim.run()
        np.testing.assert_allclose(t, th)
        if isinstance(sim.integrator, IntegratorStochastic):
            self._check_match(y, yh, rtol = 1e-2, atol = 2e-2)
        else:
            self._check_match(y, yh)
            
    
    def _check_match(self, expected, actual, rtol=2e-4, atol=1e-4):
        # check we don't have numerical errors
        if not np.isfinite(actual).all():
            raise ValueError('Non-finite values in actual')
        # check tolerances
        maxtol = np.max(np.abs(actual[0] - expected[0]))
        print('maxtol 1st step:', maxtol)
        for t in range(0, len(actual)):
            print(t, 'tol:', np.max(np.abs(actual[t] - expected[t])))
            np.testing.assert_allclose(actual[t], expected[t], rtol*(t+1)*2, atol*(t+1)*2)

class TestJaxSim(BaseTestSim):

    def _check_match(self, expected, actual):
        # check we don't have numerical errors
        if not np.isfinite(actual).all():
            raise ValueError('Non-finite values in actual')
        # check tolerances
        maxtol = np.max(np.abs(actual[0,0] - expected[0,:,:,0]))
        print('maxtol 1st step:', maxtol)
        for t in range(0, len(actual)):
        # for t in range(1, len(actual)):
            print(t, 'tol:', np.max(np.abs(actual[t] - expected[t,:,:,0])))
            np.testing.assert_allclose(actual[t, :],
                                        #    expected[t, :, :, 0], 2e-5*t*2, 1e-5*t*2)
                                       expected[t, :, :, 0], 4e-3*(t+1)*2, 2e-3*(t+1)*2)
        
    def _test_mpr(self, integrator, delays=False):
        sim = self._create_sim(
            integrator,
            inhom_mmpr=True,
            delays=delays,
            run=False
        )

        kernel, args = JaxBackend().build(sim, print_source=True)
        (th, yh), = kernel(*args)
        (t, y), = sim.run()

        self._check_match(y, yh)
        np.testing.assert_allclose(t, th)

        # Check gradients
        state, weights, *resargs = args
        kernel_weights = lambda weights: kernel(state, weights, *resargs)

        test_util.check_grads(kernel_weights, (weights,), order=1, modes=['fwd', 'rev'], eps = 0.8 )


    def _test_mvar(self, integrator):
        pass # TODO

    def _test_integrator(self, Integrator, delays=False, nsig=False):
        dt = 0.01
        if issubclass(Integrator, IntegratorStochastic):
            if nsig:
                integrator = Integrator(dt=dt, noise=Additive(nsig= 0.001 * np.r_[dt,dt*2]))
            else:
                integrator = Integrator(dt=dt, noise=Additive(nsig=0.001 * np.r_[dt]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=dt)
        if isinstance(integrator, (Identity, IdentityStochastic)):
            self._test_mvar(integrator, delays=delays)
        else:
            self._test_mpr(integrator, delays=delays)


    # TODO move to BaseTestSim to avoid duplicating all the methods
    def test_euler(self): self._test_integrator(EulerDeterministic)
    def test_eulers(self): self._test_integrator(EulerStochastic)
    def test_eulersn(self): self._test_integrator(EulerStochastic, nsig=True)
    def test_heun(self): self._test_integrator(HeunDeterministic)
    def test_heuns(self): self._test_integrator(HeunStochastic)
    def test_rk4(self): self._test_integrator(RungeKutta4thOrderDeterministic)
    # def test_id(self): self._test_integrator(Identity)
    # def test_ids(self): self._test_integrator(IdentityStochastic)

    def test_deuler(self): self._test_integrator(EulerDeterministic, delays=True)
    def test_deulers(self): self._test_integrator(EulerStochastic, delays=True)
    def test_dheun(self): self._test_integrator(HeunDeterministic, delays=True)
    def test_dheuns(self): self._test_integrator(HeunStochastic, delays=True)
    def test_drk4(self): self._test_integrator(RungeKutta4thOrderDeterministic, delays=True)
    # def test_did(self): self._test_integrator(Identity,
    #                                           delays=True)
    #
    # def test_dids(self): self._test_integrator(IdentityStochastic,
    #                                            delays=True)

    def test_float32(self):
        sim = self._create_sim(EulerDeterministic(), run=False)
        sim.configure()
        kernel, args = JaxBackend(enable_x64=False).build(sim, print_source=True)
        (th, yh), = kernel(*args)
        assert yh.dtype == jnp.float32
    
    def test_float64(self):
        sim = self._create_sim(EulerDeterministic(), run=False)
        sim.configure()
        kernel, args = JaxBackend(enable_x64=True).build(sim, print_source=True)
        (th, yh), = kernel(*args)
        assert yh.dtype == jnp.float64

    def test_scipy_int_notimpl(self):
        with self.assertRaises(NotImplementedError):
            self._test_integrator(VODEStochastic)

    def test_multnoise_notimpl(self):
        dt = 0.01
        integrator = HeunStochastic(dt=dt, noise=Multiplicative(nsig=np.r_[dt]))
        with self.assertRaises(NotImplementedError):
            self._test_mpr(integrator)

    def test_voi_order(self):
        sim = self._create_sim(EulerDeterministic(), run=False)
        sim.model.variables_of_interest = ("V", "r") # reverse to default
        sim.configure()
        kernel, args = JaxBackend().build(sim, print_source=True)
        (th, yh), = kernel(*args)
        (t, y), = sim.run()
        self._check_match(y, yh)

    def test_voi_selection(self):
        sim = self._create_sim(EulerDeterministic(), run=False)
        sim.model.variables_of_interest = ("V") # Only one svar is of interest
        sim.configure()
        kernel, args = JaxBackend().build(sim, print_source=True)
        (th, yh), = kernel(*args)
        (t, y), = sim.run()
        self._check_match(y, yh)

    def test_voi_derived(self):
        sim = self._create_sim(EulerDeterministic(), run=False)
        class ExtendedMontbrioPazoRoxin(MontbrioPazoRoxin):
            variables_of_interest = List(
            of=str,
            label="Variables or quantities available to Monitors",
            choices=("r", "V", "r-V"),
            default=("r", "V", "r-V"),
            doc="Add a derived quantity.",
        )
        sim.model = ExtendedMontbrioPazoRoxin()
        sim.configure()
        kernel, args = JaxBackend().build(sim, print_source=True)
        (th, yh), = kernel(*args)
        (t, y), = sim.run()
        self._check_match(y, yh)



class TestJaxCoupling(BaseTestCoupling):

    def _test_cfun(self, cfun):
        "Test a Python cfun template."
        sim = self._prep_sim(cfun)
        # prep & invoke kernel
        template = f'''<%include file="jax-coupling.py.mako"/>'''
        kernel = JaxBackend().build_py_func(template, dict(sim=sim, np=np), 
            name='cfun', print_source=True)
        params_cfun = JaxBackend()._collect_cfun_params(sim)
        fill = np.r_[:sim.history.buffer.size]
        fill = np.reshape(fill, sim.history.buffer.shape[:-1])
        sim.history.buffer[..., 0] = fill
        sim.current_state[:] = fill[0,:,:,None]
        buf = sim.history.buffer[...,0]
        # kernel has history in reverse order except 1st element 🤕
        rbuf = np.concatenate((buf[0:1], buf[1:][::-1]), axis=0)
        state = np.transpose(rbuf, (1, 0, 2)).astype('f')
        weights = sim.connectivity.weights.astype('f')
        # cX = jnp.zeros_like(state[:,0])
        cX_kernel = kernel(weights, state, state[0,:,:], params_cfun(), sim.connectivity.delay_indices)
        # do comparison
        (t, y), = sim.run()
        np.testing.assert_allclose(cX_kernel, y[0,:,:,0], 1e-5, 1e-6)


    def test_jax_linear(self): self._test_cfun(Linear())
    def test_jax_sigmoidal(self): self._test_cfun(Sigmoidal())
    def test_jax_difference(self): self._test_cfun(Difference()) # Added as simplest coupling containing pre()


class TestJaxDfun(BaseTestDfun):

    def _test_dfun(self, model_):
        "Test a Python cfun template."
        class sim:  # dummy sim
            model = model_
        template = '''<%include file="jax-dfuns.py.mako"/>'''
        kernel = JaxBackend().build_py_func(template, dict(sim=sim, np=np),
            name='dfun', print_source=True)
        params_dfun = JaxBackend()._collect_dfun_params(sim)
        # key = jax.random.PRNGKey(0)  # Create a random key
        # cX = jax.random.uniform(key, shape = (2, 128))
        cX = np.random.rand(2, 128)
        # dX = jnp.zeros_like(cX)
        # state = jax.random.uniform(key, shape= (2, 1, 128))
        state = np.random.rand(2, 1, 128)
        # parmat = jnp.array(sim.model.spatial_parameter_matrix)
        # parmat = sim.model.spatial_parameter_matrix
        dX_jax = kernel(state, cX, params_dfun()) # JAX does not work inplace
        np.testing.assert_allclose(dX_jax, sim.model.dfun(state, cX)[:, 0], rtol=5e-05)

    def test_py_mpr_symmetric(self):
        "Test symmetric MPR model"
        self._test_dfun(self._prep_model())

    def test_py_mpr_spatial1(self):
        "Test MPR w/ 1 spatial parameter."
        self._test_dfun(self._prep_model(1))

    def test_py_mpr_spatial2(self):
        "Test MPR w/ 2 spatial parameters."
        self._test_dfun(self._prep_model(2))


class TestJaxIntegrate(BaseTestIntegrate):

    def _test_dfun(self, state, cX, lc):
        return -state*cX**2/state.shape[1]

    def _eval_cg(self, integrator_, state, weights_):
        class sim:
            integrator = integrator_
            connectivity = Connectivity.from_file()
            stimulus = None
            class model:
                state_variables = 'foo', 'bar'
                nintvar = 2
                nvar = 2
                cvar = np.array([0])
        sim.connectivity.speed = np.r_[np.inf]
        sim.connectivity.configure()
        sim.integrator.configure()
        sim.connectivity.set_idelays(sim.integrator.dt)
        template = '''
import numpy as np
import jax.numpy as jnp
def cfun(weights, history, current_state, params_cfun, delay_indices): 
    cX = weights.dot(current_state.T).T
    return cX 
def dfun(state, cX, params_dfun):
    dX = -state*cX**2/state.shape[1]
    return dX
<%include file="jax-integrate.py.mako" />
'''
        integrate = JaxBackend().build_py_func(template, dict(sim=sim, np=np),
            name='integrate', print_source=True)
        params_ = namedtuple("params", ["params_dfun", "params_cfun", "params_stimulus"], defaults = [0,0,0])
        
        np.random.seed(42)
        args = ((state, state), weights_, params_())
        if isinstance(sim.integrator, IntegratorStochastic):
            args = args + (1, (1, sim.integrator.noise.nsig))
        else:
            args = args + (1, 1)
        return [integrate, args]

    def _test_integrator(self, Integrator):
        if issubclass(Integrator, IntegratorStochastic):
            integrator = Integrator(dt=0.1, noise=Additive(nsig=np.r_[0.0001]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=0.1)
        nn = 76
        state_tvb = np.random.randn(2, 1, nn)
        weights = np.random.randn(nn, nn)
        cx = weights.dot(state_tvb[:,0].T).T
        assert cx.shape == (2, nn)
        expected = integrator.scheme(state_tvb[:,0], self._test_dfun, cx, 0, 0)

        state = np.squeeze(state_tvb)
        integrate, args = self._eval_cg(integrator, state, weights)
        actual, _ = integrate(*args)
        actual = actual[1]
        # JAX and numpy produce different random numbers, so we set the tolerance to the noise level
        if issubclass(Integrator, IntegratorStochastic):
            np.testing.assert_allclose(actual, expected, atol = np.sqrt(2 * integrator.noise.nsig)[0])
        else:
            np.testing.assert_allclose(actual, expected, rtol = 1e-6)

        # # Test gradient can be pulled from the integrator
        # state, weights, params_integrate, *resargs = args
        # integrate_weights = lambda weights: integrate(state, weights, params_integrate, *resargs)
        # test_util.check_grads(integrate_weights, (weights,), order=1, modes=['fwd', 'rev'], eps = 0.5, rtol = 0.1)

    def test_euler(self): self._test_integrator(EulerDeterministic)
    def test_eulers(self): self._test_integrator(EulerStochastic)
    def test_heun(self): self._test_integrator(HeunDeterministic)
    def test_heuns(self): self._test_integrator(HeunStochastic)
    def test_rk4(self): self._test_integrator(RungeKutta4thOrderDeterministic)
    def test_id(self): self._test_integrator(Identity)
    def test_ids(self): self._test_integrator(IdentityStochastic)


class TestJaxMonitors(BaseTestMonitors):
    def prepare_monitor(self, monitor, name, istep = None):
        monitor.istep = istep
        class sim_:
            monitors = [monitor]
            model_ = namedtuple("model", ["variables_of_interest", "number_of_modes"], defaults = [np.array([0, 1]), 1])
            model = model_()
            number_of_nodes = 76
            integrator_ = namedtuple("integrator", ["dt"], defaults = [0.1,])
            integrator = integrator_()
        sim = sim_()
        sim.monitors[0]._config_vois(sim)
        if isinstance(monitor, Bold):
            sim.monitors[0].config_for_sim(sim)
        template = '''
## <%include file="jax-monitors.py.mako"/>
<%namespace name="monitors" file="jax-monitors.py.mako"/>
from collections import namedtuple
import jax.numpy as jnp
import jax
import jax.scipy.signal as sig
exp, sin, sqrt = jnp.exp, jnp.sin, jnp.sqrt
timeseries = namedtuple("timeseries", ["time", "trace"])

${monitors.create_monitor(0, sim.monitors[0])}
            '''
        monitor_fun = JaxBackend().build_py_func(template, dict(sim=sim, np=np),
            name=name, print_source=True)
        return monitor_fun

    def test_Raw(self):
        monitor_fun = self.prepare_monitor(Raw(), "monitor_raw_0")
        trace = np.random.randn(100, 2, 76)
        time_steps = np.arange(1, 100)
        dt = 0.1
        t, y = monitor_fun(time_steps, trace, None)
        np.testing.assert_allclose(t, time_steps * dt)
        np.testing.assert_allclose(y, trace)
        
    def test_RawVoi(self):
        monitor_fun = self.prepare_monitor(RawVoi(variables_of_interest = np.array([1,])), "monitor_raw_voi_0")
        trace = np.random.randn(100, 2, 76)
        time_steps = np.arange(1, 100)
        dt = 0.1
        t, y = monitor_fun(time_steps, trace, None)
        np.testing.assert_allclose(t, time_steps * dt)
        np.testing.assert_equal(y.shape, (100, 1, 76)) 
        np.testing.assert_allclose(y, trace[:, [1], :])

    def test_TemporalAverage(self):
        period = 1.0
        trace = np.random.randn(100, 2, 76)
        time_steps = np.arange(1, 100)
        dt = 0.1
        istep = np.round(period / dt).astype(np.int32)
        monitor_fun = self.prepare_monitor(TemporalAverage(period = period), "monitor_temporal_average_0", istep = istep)
        t, y = monitor_fun(time_steps, trace, None)
        np.testing.assert_allclose(t, ((time_steps[::istep]-1) + istep / 2) * dt)
        np.testing.assert_equal(y.shape, ((trace.shape[0] // istep), 2, 76))
        # np.testing.assert_allclose(y, trace[::10, :, :])
    
    def test_SubSample(self):
        trace = np.random.randn(100, 2, 76)
        time_steps = np.arange(1, 100)
        dt = 0.1
        istep = 10
        monitor_fun = self.prepare_monitor(SubSample(period = 1), "monitor_subsample_0", istep=istep)
        t, y = monitor_fun(time_steps, trace, None)
        idxs = jnp.arange(istep-1, time_steps.shape[0], istep)
        np.testing.assert_allclose(t, time_steps[idxs] * dt)
        np.testing.assert_equal(y.shape, (9, 2, 76))
        np.testing.assert_allclose(y, trace[idxs, :, :])
    
    def test_Bold(self):
        # Default period of Bold() is 2s so we can expect a single value as return
        monitor_fun = self.prepare_monitor(Bold(), "monitor_bold_0")
        trace = np.random.randn(20_001, 2, 76)
        time_steps = np.arange(1, 20_001)
        dt = 0.1
        t, y = monitor_fun(time_steps, trace, (0.8, 0.4, 5.6, 0.02))
        np.testing.assert_allclose(t, [2000.0])
        np.testing.assert_equal(y.shape, (1, 2, 76))

# TODO surface support

# TODO stimulus support

# TODO bounds/clamp support

