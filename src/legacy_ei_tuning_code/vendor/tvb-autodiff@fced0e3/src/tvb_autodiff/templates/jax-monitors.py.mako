# -*- coding: utf-8 -*-

from collections import namedtuple

<%namespace name="utils" file="jax-utils.py.mako"/>

<%def name = "apply_monitor(i, monitor)" filter = "trim">
    <%
        from tvb.simulator.monitors import Raw, RawVoi, TemporalAverage, SubSample, Bold, EEG, Projection, AfferentCoupling
    %>
    % if isinstance(monitor, AfferentCoupling):
    monitor_afferent_coupling_${i}(time_steps, node_coupling, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, RawVoi):
    monitor_raw_voi_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, Raw):
    monitor_raw_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, TemporalAverage):
    monitor_temporal_average_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, SubSample):
    monitor_subsample_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, Bold):
    monitor_bold_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, EEG):
    monitor_eeg_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % elif isinstance(monitor, Projection):
    monitor_projection_${i}(time_steps, trace, params_monitors[${i}], t_offset = t_offset),
    % endif
</%def>

<%def name = "create_monitor(i, monitor)" filter = "trim">
<%
    from tvb.simulator.monitors import Raw, RawVoi, TemporalAverage, SubSample, Bold, EEG, Projection, AfferentCoupling
    from tvb.simulator.lab import equations

    import jax.numpy as jnp
%>

### Raw monitor
% if isinstance(monitor, Raw) & (not isinstance(monitor, AfferentCoupling)):
def monitor_raw_${i}(time_steps, trace, params, t_offset = 0):
    dt = ${sim.integrator.dt}
    return timeseries(time=(time_steps + t_offset) * dt, trace=trace)
% endif

### Raw VOI monitor
% if isinstance(monitor, RawVoi):
def monitor_raw_voi_${i}(time_steps, trace, params, t_offset = 0):
    dt = ${sim.integrator.dt}
    voi = ${utils.array_input(monitor.voi)}
    return timeseries(time=(time_steps + t_offset) * dt, trace=trace[:, voi, :])
% endif

### Afferent Coupling monitor
## There is an offset of -1 to the TVB affrent monitor...
% if isinstance(monitor, AfferentCoupling):
def monitor_afferent_coupling_${i}(time_steps, node_coupling, params, t_offset = 0):
    return monitor_raw_voi_${i}(time_steps, node_coupling, params, t_offset = t_offset)
% endif

### Temporal Average monitor
## Bold and Projection based monitors reuse the temporal average monitor in non replace_temporal_averaging mode
% if isinstance(monitor, TemporalAverage) or (not replace_temporal_averaging and isinstance(monitor, (Bold, Projection))):
def monitor_temporal_average_${i}(time_steps, trace, params, t_offset = 0):
    dt = ${sim.integrator.dt}
    voi = ${utils.array_input(monitor.voi)}
    % if isinstance(monitor, TemporalAverage):
    istep = ${monitor.istep}
    t_map = time_steps[::istep] - 1
    % elif isinstance(monitor, Projection):
    istep = ${monitor._period_in_steps}
    t_map = jnp.arange(0, time_steps.shape[0]-istep, istep)
    % else: ## Bold
    istep = ${monitor._interim_istep}
    t_map = time_steps[::istep] - 1
    % endif

    def op(ts): 
        start_indices = (ts,) + (0,) * (trace.ndim - 1)
        slice_sizes = (istep,) + voi.shape + trace.shape[2:]
        return jnp.mean(jax.lax.dynamic_slice(trace[:, voi, :], start_indices, slice_sizes), axis=0)
    trace_out = jax.lax.map(op, t_map)

    idxs = jnp.arange(((istep - 2) // 2), time_steps.shape[0], istep)
    return timeseries(time=(time_steps[idxs]) * dt, trace=trace_out[0:idxs.shape[0], :, :])
% endif

### SubSample monitor
## Bold and Projection based monitors reuse the temporal average monitor in replace_temporal_averaging mode
% if isinstance(monitor, SubSample) or (replace_temporal_averaging and isinstance(monitor, (Bold, Projection))):
def monitor_subsample_${i}(time_steps, trace, params, t_offset = 0):
    dt = ${sim.integrator.dt}
    voi = ${utils.array_input(monitor.voi)}
    period = ${monitor.period} # sampling period in ms

    % if isinstance(monitor, SubSample):
    istep = ${monitor.istep}
    % elif isinstance(monitor, Projection):
    istep = ${monitor._period_in_steps}
    % else: ## Bold
    istep = ${monitor._interim_istep}
    % endif
    
    idxs = jnp.arange(istep-1, time_steps.shape[0], istep)
    return timeseries(time=(time_steps[idxs] + t_offset) * dt, trace=trace[idxs[:, None], voi[None, :], :])
% endif

### BOLD monitor
% if isinstance(monitor, Bold):
import jax.scipy.signal as sig
exp, sin, sqrt = jnp.exp, jnp.sin, jnp.sqrt

def monitor_bold_${i}(time_steps, trace, params, t_offset = 0):
    # downsampling via temporal average / subsample
    dt = ${sim.integrator.dt}
    voi = ${utils.array_input(monitor.voi)}
    period = ${monitor.period} # sampling period in ms
    period_int = ${monitor._interim_period} # sampling period in steps
    istep_int = ${monitor._interim_istep}
    istep = ${jnp.round(monitor.period / sim.integrator.dt).astype(jnp.int32)}
    final_istep = ${(jnp.round(monitor.period / sim.integrator.dt) / monitor._interim_istep).astype(jnp.int32)}
    
    % if replace_temporal_averaging:
    time_steps_i, trace_new = monitor_subsample_${i}(time_steps, trace, None)
    %else:
    time_steps_i, trace_new = monitor_temporal_average_${i}(time_steps, trace, None)
    % endif

    time_steps_new = time_steps[jnp.arange(istep-1, time_steps.shape[0], istep)]
    # hemodynamic response function
        \
% for par, _ in monitor.hrf_kernel.parameters.items():
${par}, \
% endfor
stock = params
    trace_new = jnp.vstack([stock, trace_new])

    op = lambda var: ${monitor.hrf_kernel.equation}
    stock_steps = ${jnp.ceil(monitor._stock_sample_rate * monitor.hrf_length).astype(jnp.int32)}
    stock_time_max = ${monitor.hrf_length / 1000.0} # stock time has to be in seconds
    stock_time_step = stock_time_max / stock_steps
    stock_time = jnp.arange(0.0, stock_time_max, stock_time_step)
    hrf = op(stock_time)##[::-1]

    # Convolution along time axis
    ## op1 = lambda x: sig.convolve(x, hrf, mode="full") ## much slower
    op1 = lambda x: sig.fftconvolve(x, hrf, mode="valid")
    op2 = lambda x: jax.vmap(op1, in_axes=(1), out_axes=(1))(x)
    op3 = lambda x: jax.vmap(op2, in_axes=(1), out_axes=(1))(x)
    bold = jax.vmap(op3, in_axes=(3), out_axes=(3))(trace_new)

    %if isinstance(monitor.hrf_kernel, equations.FirstOrderVolterra):
    bold = k_1 * V_0 * (bold - 1.0)
    %endif

    ## return timeseries(time=time_steps_new * dt, trace=bold[jnp.arange(final_istep-1, time_steps_i.shape[0], final_istep), :, :])
    bold_idx = jnp.arange(final_istep-2, time_steps_i.shape[0], final_istep)[0:time_steps_new.shape[0]] + 1
    return timeseries(time=(time_steps_new + t_offset) * dt, trace=bold[bold_idx, :, :])
    ## return timeseries(time=(time_steps_new + t_offset) * dt, trace=bold)
% endif

### Projection monitor
% if isinstance(monitor, Projection):
def monitor_projection_${i}(time_steps, trace, params, t_offset = 0):
    voi = ${utils.array_input(monitor.voi)}
    dt = ${sim.integrator.dt}

    gain = params.gain
    ## projection = jnp.matmul(trace[:, voi, :], gain.T)
    op = lambda x: jnp.matmul(x, gain.T)
    projection = jax.vmap(op, in_axes=(3), out_axes=(3))(trace)
    istep = ${monitor._period_in_steps}

     % if replace_temporal_averaging:
    t, proj_avg =  monitor_subsample_${i}(time_steps, projection, None) 
    %else:
    t, proj_avg = monitor_temporal_average_${i}(time_steps, projection, None) 

    ## Temporal Average
    ## istep = ${monitor._period_in_steps}
    ## def op(ts): 
    ##     start_indices = (ts,) + (0,) * (trace.ndim - 1)
    ##     slice_sizes = (istep,) + voi.shape + projection.shape[2:]
    ##     return jnp.mean(jax.lax.dynamic_slice(projection[:, voi, :], start_indices, slice_sizes), axis=0)
    ## proj_avg = jax.lax.map(op, (time_steps[::istep] - 1))

    ## idxs = jnp.arange(((istep - 2) // 2), time_steps.shape[0], istep)
    ## t = (time_steps[idxs] + 0.5) * dt
    t += 0.5 * dt
    % endif

    % if monitor.obsnoise is not None:
    proj_avg += params.obsnoise
    return timeseries(time=t, trace=proj_avg)
    %else:
    return timeseries(time=t, trace=proj_avg)
    % endif
% endif

### EEG monitor
% if isinstance(monitor, EEG):
def monitor_eeg_${i}(time_steps, trace, params, t_offset = 0):
    % if monitor.reference:
    t, _eeg = monitor_projection_${i}(time_steps, trace, params, t_offset = t_offset)
    ref_vec = params.ref_vec
    return timeseries(time=t, trace=_eeg - jnp.matmul(_eeg, ref_vec))
    %else:
    return monitor_projection_${i}(time_steps, trace, params, t_offset = t_offset)
    % endif
% endif

</%def>