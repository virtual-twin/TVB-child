# -*- coding: utf-8 -*-

"""
Tests for the pytensor backend.



"""

import numpy as np
import pytensor
from pytensor import tensor as pyt

from collections import namedtuple

from tvb_autodiff.pytensor import PytensorBackend
from tvb.simulator.coupling import Sigmoidal, Linear, Difference
from tvb.simulator.integrators import (
    EulerDeterministic, EulerStochastic,
    HeunDeterministic, HeunStochastic,
    IntegratorStochastic, RungeKutta4thOrderDeterministic,
    Identity, IdentityStochastic,
    VODEStochastic)
from tvb.simulator.noise import Additive, Multiplicative
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw, RawVoi, TemporalAverage, SubSample, Bold
from tvb.simulator.models.oscillator import Generic2dOscillator

from tests.backendtestbase import BaseTestDfun, BaseTestCoupling, BaseTestIntegrate, BaseTestSim


class TestPytensorSim(BaseTestSim):

    def _test_mpr(self, integrator, delays=False):
        sim, state_numpy, t, y = self._create_sim(
            integrator,
            inhom_mmpr=True,
            delays=delays
        )
        template = '<%include file="pytensor-sim.py.mako"/>'
        content = dict(sim=sim, mparams={}, cparams={}, np=np, pyt=pyt)
        kernel, default_noise = PytensorBackend().build_py_func(template, content, name="kernel,default_noise", print_source=True)

        state = pyt.as_tensor_variable(state_numpy, name="state")
        dX = state.copy()
        n_svar, _, n_node = state.eval().shape

        if not delays:
            self.assertEqual(sim.connectivity.horizon, 1)  # for now

        state = state.reshape((n_svar, sim.connectivity.horizon, n_node))

        weights_numpy = sim.connectivity.weights.copy()
        weights = pyt.as_tensor_variable(weights_numpy, name="weights")

        yh_numpy = np.zeros((len(t),) + state.eval()[:, 0].shape)
        yh = pyt.as_tensor_variable(yh_numpy, name="yh")

        parmat = sim.model.spatial_parameter_matrix
        self.assertEqual(parmat.shape[0], 1)
        self.assertEqual(parmat.shape[1], weights.eval().shape[1])
        np.random.seed(42)

        args = state, weights, yh, parmat
        if isinstance(integrator, IntegratorStochastic):
            args = args + (default_noise(sim.integrator.noise.nsig),)
        if delays:
            args = args + (sim.connectivity.delay_indices,)

        yh = kernel(*args)
        yh = yh.eval()[:, sim.model.cvar, 0, :].reshape(y.shape)
        self._check_match(y, yh)

    def _test_mvar(self, integrator):
        pass  # TODO

    def _test_osc(self, integrator, delays=False):
        sim, state_numpy, t, y = self._create_osc_sim(
            integrator,
            delays=delays
        )
        template = '<%include file="pytensor-sim.py.mako"/>'
        content = dict(sim=sim, mparams={}, cparams={}, np=np, pyt=pyt)
        kernel, default_noise = PytensorBackend().build_py_func(template, content, name="kernel,default_noise", print_source=True)

        state = pyt.as_tensor_variable(state_numpy, name="state")
        dX = state.copy()
        n_svar, _, n_node = state.eval().shape

        if not delays:
            self.assertEqual(sim.connectivity.horizon, 1)  # for now

        state = state.reshape((n_svar, sim.connectivity.horizon, n_node))

        weights_numpy = sim.connectivity.weights.copy()
        weights = pyt.as_tensor_variable(weights_numpy, name="weights")

        yh_numpy = np.zeros((len(t),) + state.eval()[:, 0].shape)
        yh = pyt.as_tensor_variable(yh_numpy, name="yh")

        parmat = sim.model.spatial_parameter_matrix
        self.assertEqual(parmat.shape[0], 0)
        np.random.seed(42)

        args = state, weights, yh, parmat
        if isinstance(integrator, IntegratorStochastic):
            args = args + (default_noise(sim.integrator.noise.nsig),)
        if delays:
            dn = np.arange(sim.connectivity.weights.shape[0]) * np.ones((sim.connectivity.weights.shape[0], sim.connectivity.weights.shape[0])).astype(int)
            di = -1 * sim.connectivity.idelays - 1
            delay_indices = (di, dn)
            args = args + (delay_indices,)
            # args = args + (sim.connectivity.delay_indices,)

        yh = kernel(*args)
        yh = yh.trace.eval()[:, sim.model.cvar, 0, :].reshape(y.shape)
        self._check_match(y, yh)

    def _test_integrator(self, Integrator, delays=False):
        if issubclass(Integrator, IntegratorStochastic):
            integrator = Integrator(dt=0.01, noise=Additive(nsig=np.r_[0.01]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=0.01)
        if isinstance(integrator, (Identity, IdentityStochastic)):
            self._test_mvar(integrator, delays=delays)
        else:
            # self._test_mpr(integrator, delays=delays)
            self._test_osc(integrator, delays=delays)

    # TODO move to BaseTestSim to avoid duplicating all the methods

    def test_euler(self):
        self._test_integrator(EulerDeterministic)

    def test_eulers(self):
        self._test_integrator(EulerStochastic)

    def test_heun(self):
        self._test_integrator(HeunDeterministic)

    def test_heuns(self):
        self._test_integrator(HeunStochastic)

    def test_rk4(self):
        self._test_integrator(RungeKutta4thOrderDeterministic)

    def test_deuler(self):
        self._test_integrator(EulerDeterministic, delays=True)

    def test_deulers(self):
        self._test_integrator(EulerStochastic, delays=True)

    def test_dheun(self):
        self._test_integrator(HeunDeterministic, delays=True)

    def test_dheuns(self):
        self._test_integrator(HeunStochastic, delays=True)

    def test_drk4(self):
        self._test_integrator(RungeKutta4thOrderDeterministic, delays=True)


class TestPytensorCoupling(BaseTestCoupling):

    def _test_cfun(self, cfun, **cparams):
        """Test a Python cfun template."""

        sim = self._prep_sim(cfun)

        # prep & invoke kernel
        template = f'''<%include file="pytensor-coupling.py.mako"/>'''
        kernel = PytensorBackend().build_py_func(template, dict(sim=sim, np=np), name='cfun', print_source=True)
        params_cfun = PytensorBackend()._collect_cfun_params(sim)

        fill = np.r_[:sim.history.buffer.size]
        fill = np.reshape(fill, sim.history.buffer.shape[:-1])
        sim.history.buffer[..., 0] = fill
        sim.current_state[:] = fill[0, :, :, None]
        buf = sim.history.buffer[..., 0]
        # kernel has history in reverse order except 1st element 🤕
        rbuf = np.concatenate((buf[0:1], buf[1:][::-1]), axis=0)

        state_numpy = np.transpose(rbuf, (1, 0, 2)).astype('f')
        state = pyt.as_tensor_variable(state_numpy, name="state")

        weights_numpy = sim.connectivity.weights.astype('f')
        weights = pyt.as_tensor_variable(weights_numpy, name="weights")

        cX_numpy = np.zeros_like(state_numpy[:, 0])
        cX = pyt.as_tensor_variable(cX_numpy, name="cX")

        dn = np.arange(weights_numpy.shape[0]) * np.ones((weights_numpy.shape[0], weights_numpy.shape[0])).astype(int)
        if sim.connectivity.idelays.any():
            di = -1 * sim.connectivity.idelays - 1
            delay_indices = (di, dn)
        else:
            delay_indices = (None, dn)

        cX = kernel(cX, weights, state, state[:, 0, :], params_cfun(), delay_indices)
        # do comparison
        (t, y), = sim.run()
        np.testing.assert_allclose(cX.eval(), y[0, :, :, 0], 1e-5, 1e-6)

    def test_linear(self):
        self._test_cfun(Linear())

    def test_difference(self):
        self._test_cfun(Difference())

    def test_sigmoidal(self):
        self._test_cfun(Sigmoidal())


class TestPytensorDfun(BaseTestDfun):

    def _test_dfun(self, model_):
        """Test a Python dfun template."""

        class sim:  # dummy sim
            model = model_

        template = '''<%include file="pytensor-dfuns.py.mako"/>'''
        kernel = PytensorBackend().build_py_func(template, dict(sim=sim, np=np), name="dfuns", print_source=True)
        params_dfun = PytensorBackend()._collect_dfun_params(sim)

        cX_numpy = np.random.rand(2, 128, 1)
        cX = pyt.as_tensor_variable(cX_numpy, name="cX")

        dX_numpy = pyt.zeros(shape=(2, 128, 1))
        dX = pyt.as_tensor_variable(dX_numpy, name="dX")

        state_numpy = np.random.rand(2, 128, 1)
        state = pyt.as_tensor_variable(state_numpy, name="state")

        dX = kernel(dX, state, cX, params_dfun())
        np.testing.assert_allclose(dX.eval(),
                                   sim.model.dfun(state_numpy, cX_numpy))

    def test_oscillator(self):
        """Test Generic2dOscillator model"""
        oscillator_model = Generic2dOscillator()
        self._test_dfun(oscillator_model)

    def test_py_mpr_symmetric(self):
        """Test symmetric MPR model"""
        self._test_dfun(self._prep_model())


class TestPytensorIntegrate(BaseTestIntegrate):

    def _test_dfun(self, state, cX, lc):
        return -state * cX ** 2 / state.shape[1]

    def _eval_cg(self, integrator_, state, weights_):
        class sim:
            integrator = integrator_
            connectivity = Connectivity.from_file(source_file="/home/ldap_users/emiliusrichter/tvb/connectivity_76.zip")

            class model:
                state_variables = "foo", "bar"

        sim.connectivity.speed = np.r_[np.inf]
        sim.connectivity.configure()
        sim.integrator.configure()
        sim.connectivity.set_idelays(sim.integrator.dt)
        template = '''
import numpy as np
import pytensor
from pytensor import tensor as pyt
def cfun(cX, weights, history, current_state, params_cfun, delay_indices): 
    cX = pyt.set_subtensor(cX[:], weights.dot(current_state.T).T)
    return cX
def dfun(dX, state, cX, params_dfun):
    dX = pyt.set_subtensor(dX[:], -state*cX**2/state.shape[1])
    return dX
<%include file="pytensor-integrate.py.mako" />
'''
        integrate = PytensorBackend().build_py_func(template, dict(sim=sim, np=np, pyt=pyt), name='integrate', print_source=True)
        params_ = namedtuple("params", ["params_dfun", "params_cfun", "params_stimulus"], defaults = [0,0,0])

        dX = pyt.zeros(shape=(integrator_.n_dx,) + state[:, 0].eval().shape)
        cX = pyt.zeros_like(state[:, 0])
        np.random.seed(42)
        args = ((state, state[:, 0]), weights_, dX, cX, params_(), (1, 1))

        if isinstance(sim.integrator, IntegratorStochastic):
            ## TODO handle multiplicative noise
            if isinstance(sim.integrator.noise, Additive):
                sim.integrator.noise.reset_random_stream()
                z_t = sim.integrator.noise.generate(dX.eval().shape)*sim.integrator.noise.gfun(dX.eval())
                z_t = z_t[0, :, :]
            else:
                raise NotImplementedError
            args = args + (z_t,)
        state = integrate(*args)
        return state

    def _test_integrator(self, Integrator):
        if issubclass(Integrator, IntegratorStochastic):
            integrator = Integrator(dt=0.1, noise=Additive(nsig=np.r_[0.01]))
            integrator.noise.dt = integrator.dt
        else:
            integrator = Integrator(dt=0.1)
        nn = 76
        state_numpy = np.random.randn(2, 1, nn)
        state = pyt.as_tensor_variable(state_numpy, name="state")

        weights_numpy = np.random.randn(nn, nn)
        weights = pyt.as_tensor_variable(weights_numpy, name="weights")

        cx_numpy = weights_numpy.dot(state_numpy[:, 0].T).T
        cx = weights.dot(state[:, 0].T).T

        assert cx_numpy.shape == (2, nn)
        expected = integrator.scheme(state_numpy[:, 0], self._test_dfun, cx_numpy, 0, 0)
        # actual = state
        np.random.seed(42)
        actual = self._eval_cg(integrator, state, weights)
        actual = actual[1]
        np.testing.assert_allclose(actual.eval(), expected)

    def test_euler(self):
        self._test_integrator(EulerDeterministic)

    def test_eulers(self):
        self._test_integrator(EulerStochastic)

    def test_heun(self):
        self._test_integrator(HeunDeterministic)

    def test_heuns(self):
        self._test_integrator(HeunStochastic)

    def test_rk4(self):
        self._test_integrator(RungeKutta4thOrderDeterministic)

    def test_id(self):
        self._test_integrator(Identity)

    def test_ids(self):
        self._test_integrator(IdentityStochastic)


class TestPytensorMonitors(BaseTestCoupling):
    def prepare_monitor(self, monitor, name):
        
        class sim_:
            monitors = [monitor]
            model_ = namedtuple("model", ["variables_of_interest", "number_of_modes", "nvar"], defaults = [np.array([0, 1]), 1, 2])
            model = model_()
            number_of_nodes = 76
            integrator_ = namedtuple("integrator", ["dt"], defaults = [0.1,])
            integrator = integrator_()
        sim = sim_()
        sim.monitors[0]._config_vois(sim)
        if isinstance(monitor, Bold):
            sim.monitors[0].config_for_sim(sim)
        template = '''
<%namespace name="monitors" file="pytensor-monitors.py.mako"/>
from collections import namedtuple
import numpy as np
import pytensor
from pytensor import tensor as pyt
import jax

timeseries = namedtuple("timeseries", ["time", "trace"])

${monitors.create_monitor(0, sim.monitors[0])}
            '''
        monitor_fun = PytensorBackend().build_py_func(template, dict(sim=sim, np=np),
            name=name, print_source=True)
        return monitor_fun

    def test_Raw(self):
        monitor_fun = self.prepare_monitor(Raw(), "monitor_raw_0")
        
        trace_numpy = np.random.randn(100, 2, 76)
        trace = pyt.as_tensor_variable(trace_numpy, "trace")

        time_steps_numpy = np.arange(1, 100)
        time_steps = pyt.as_tensor_variable(time_steps_numpy, "time_steps")

        dt = 0.1
        t, y = monitor_fun(time_steps, trace)
        np.testing.assert_allclose(t.eval(), time_steps_numpy * dt)
        np.testing.assert_allclose(y.eval(), trace_numpy)
        
    def test_RawVoi(self):
        monitor_fun = self.prepare_monitor(RawVoi(variables_of_interest = np.array([1,])), "monitor_raw_voi_0")
        
        trace_numpy = np.random.randn(100, 2, 76)
        trace = pyt.as_tensor_variable(trace_numpy, "trace")

        time_steps_numpy = np.arange(1, 100)
        time_steps = pyt.as_tensor_variable(time_steps_numpy, "time_steps")

        dt = 0.1
        t, y = monitor_fun(time_steps, trace)
        np.testing.assert_allclose(t.eval(), time_steps_numpy * dt)
        np.testing.assert_equal(y.eval().shape, (100, 1, 76)) 
        np.testing.assert_allclose(y.eval(), trace_numpy[:, [1], :])

    def test_TemporalAverage(self):
        period = 1
        monitor_fun = self.prepare_monitor(TemporalAverage(period=period), "monitor_temporal_average_0")

        trace_numpy = np.random.randn(100, 2, 76)
        trace = pyt.as_tensor_variable(trace_numpy, "trace")

        time_steps_numpy = np.arange(1, 100)
        time_steps = pyt.as_tensor_variable(time_steps_numpy, "time_steps")
        
        dt = 0.1
        isteps = np.round(period / dt).astype(np.int32)
        t, y = monitor_fun(time_steps, trace)
        np.testing.assert_allclose(t.eval(), (time_steps_numpy[::isteps] + isteps / 2) * dt)
        np.testing.assert_equal(y.eval().shape, ((trace_numpy.shape[0] // isteps), 2, 76))
        # np.testing.assert_allclose(y, trace[::10, :, :])
    
    def test_SubSample(self):
        monitor_fun = self.prepare_monitor(SubSample(period = 1), "monitor_subsample_0")
        
        trace_numpy = np.random.randn(100, 2, 76)
        trace = pyt.as_tensor_variable(trace_numpy, "trace")

        time_steps_numpy = np.arange(1, 100)
        time_steps = pyt.as_tensor_variable(time_steps_numpy, "time_steps")
        
        dt = 0.1
        t, y = monitor_fun(time_steps, trace)
        np.testing.assert_allclose(t.eval(), time_steps_numpy[::10] * dt)
        np.testing.assert_equal(y.eval().shape, (10, 2, 76))
        np.testing.assert_allclose(y.eval(), trace_numpy[::10, :, :])
    
    def test_Bold(self):
        # Default period of Bold() is 2s so we can expect a single value as return
        monitor_fun = self.prepare_monitor(Bold(), "monitor_bold_0")

        trace_numpy = np.random.randn(20_001, 2, 76)
        trace = pyt.as_tensor_variable(trace_numpy, "trace")

        time_steps_numpy = np.arange(1, 20_001)
        time_steps = pyt.as_tensor_variable(time_steps_numpy, "time_steps")
        
        dt = 0.1
        t, y = monitor_fun(time_steps, trace)
        np.testing.assert_allclose(t.eval(), [2000.0])
        np.testing.assert_equal(y.eval().shape, (1, 2, 76))