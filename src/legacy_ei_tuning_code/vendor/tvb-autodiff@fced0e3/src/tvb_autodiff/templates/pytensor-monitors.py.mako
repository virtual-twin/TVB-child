# -*- coding: utf-8 -*-

from collections import namedtuple

<%def name = "array_input(array)" filter="trim">
        np.array(${np.array2string(array, separator=",")})
</%def>

<%def name = "apply_monitor(i, monitor)" filter = "trim">
    <%
        from tvb.simulator.monitors import Raw, RawVoi, TemporalAverage, SubSample, Bold, EEG, Projection
    %>
    % if isinstance(monitor, RawVoi):
    monitor_raw_voi_${i}(time_steps, trace, params_monitors[${i}]),
    % elif isinstance(monitor, Raw):
    monitor_raw_${i}(time_steps, trace, params_monitors[${i}]),
    % elif isinstance(monitor, TemporalAverage):
    monitor_temporal_average_${i}(time_steps, trace, params_monitors[${i}]),
    % elif isinstance(monitor, SubSample):
    monitor_subsample_${i}(time_steps, trace, params_monitors[${i}]),
    % elif isinstance(monitor, Bold):
    monitor_bold_${i}(time_steps, trace, params_monitors[${i}]),
    % elif isinstance(monitor, EEG):
    monitor_eeg_${i}(time_steps, trace, params_monitors[${i}]),
    % elif isinstance(monitor, Projection):
    monitor_projection_${i}(time_steps, trace, params_monitors[${i}]),
    % endif
</%def>

<%def name = "create_monitor(i, monitor)" filter = "trim">
<%
    from tvb.simulator.monitors import Raw, RawVoi, TemporalAverage, SubSample, Bold, Projection, EEG
    from tvb.simulator.lab import equations
%>

### Raw monitor
% if isinstance(monitor, Raw):
def monitor_raw_${i}(time_steps, trace, params):
    dt = ${sim.integrator.dt}
    return timeseries(time=time_steps * dt, trace=trace)
% endif

### Raw VOI monitor
% if isinstance(monitor, RawVoi):
def monitor_raw_voi_${i}(time_steps, trace, params):
    dt = ${sim.integrator.dt}
    voi = ${array_input(monitor.voi)}
    return timeseries(time=time_steps * dt, trace=trace[:, voi, :])
% endif

### Temporal Average monitor
## Bold and Projection based monitors reuse the temporal average monitor in non replace_temporal_averaging mode
% if isinstance(monitor, TemporalAverage) or (not replace_temporal_averaging and isinstance(monitor, (Bold, Projection))):
def monitor_temporal_average_${i}(time_steps, trace, params):
    dt = ${sim.integrator.dt}
    n_node = ${sim.number_of_nodes}
    voi = ${array_input(monitor.voi)}
    % if isinstance(monitor, TemporalAverage):
    istep = ${np.round(monitor.period / sim.integrator.dt).astype(np.int32)}
    % else:
    istep = ${monitor._interim_istep}
    % endif
    
    def op(ts): 
        return pyt.mean(trace[(ts-istep):ts, voi, :], axis=0)
    
    time_steps_i = time_steps
    trace_oute = pyt.mean(trace[time_steps_i[::istep][-1]:, voi, :], axis=0)
    trace_oute = pyt.shape_padleft(trace_oute)
    trace_out, _ = pytensor.scan(fn=op, outputs_info=None, sequences=time_steps_i[::istep][1:])
    
    return timeseries(time=(time_steps[::istep] + istep / 2.0) * dt, trace=pyt.concatenate([trace_out, trace_oute]))
% endif

### SubSample monitor
## Bold and Projection based monitors reuse the temporal average monitor in replace_temporal_averaging mode
% if isinstance(monitor, SubSample) or (replace_temporal_averaging and isinstance(monitor, (Bold, Projection))):
def monitor_subsample_${i}(time_steps, trace, params):
    dt = ${sim.integrator.dt}
    voi = ${array_input(monitor.voi)}
    period = ${monitor.period} # sampling period in ms
    
    % if isinstance(monitor, SubSample):
    istep = ${monitor.istep}
    % elif isinstance(monitor, Projection):
    istep = ${monitor._period_in_steps}
    % else: ## Bold
    istep = ${monitor._interim_istep}
    % endif
    
    trace_out = pyt.zeros_like(trace[:, :len(voi), :])
    for index, vindex in enumerate(voi):
        trace_out = pyt.set_subtensor(trace_out[:, index, :], trace[:, vindex, :])

    return timeseries(time=time_steps[::istep] * dt, trace=trace_out[::istep, :, :])
% endif

### Bold monitor
% if isinstance(monitor, Bold):
import scipy.signal as sig
exp, sin, sqrt = pyt.exp, pyt.sin, pyt.sqrt
## params_bold_${i} = namedtuple("params_bold_${i}", [\
## % for par, _ in monitor.hrf_kernel.parameters.items():
## "${par}", \
## % endfor
## ], defaults=[
## % for _, value in monitor.hrf_kernel.parameters.items():
## ${value}, 
## % endfor
## ])

def monitor_bold_${i}(time_steps, trace, params):
    # downsampling via temporal average / subsample
    dt = ${sim.integrator.dt}
    n_svar = ${sim.model.nvar}
    n_node = ${sim.number_of_nodes}
    nt = ${int(sim.simulation_length/sim.integrator.dt)}
    voi = ${array_input(monitor.voi)}
    period = ${monitor.period} # sampling period in ms
    period_int = ${monitor._interim_period} # sampling period in ms
    istep_int = ${monitor._interim_istep}
    istep = ${np.round(monitor.period / sim.integrator.dt).astype(np.int32)}
    final_istep = ${(np.round(monitor.period / sim.integrator.dt) / monitor._interim_istep).astype(np.int64)}
    
    % if replace_temporal_averaging:
    time_steps_i, trace_new = monitor_subsample_${i}(time_steps, trace, None)
    %else:
    time_steps_i, trace_new = monitor_temporal_average_${i}(time_steps, trace, None)
    % endif
    len_i = int((nt/period_int)*dt)

    time_steps_new = time_steps[np.arange(istep-1, nt, istep)]
    len_new = len(np.arange(istep-1, nt, istep))

    # hemodynamic response function
       \
% for par, _ in monitor.hrf_kernel.parameters.items():
${par}, \
% endfor
= params

    op = lambda var: ${monitor.hrf_kernel.equation}
    stock_steps = ${np.ceil(monitor._stock_sample_rate * monitor.hrf_length).astype(np.int64)}
    stock_time_max = ${monitor.hrf_length / 1000.0} # stock time has to be in seconds
    stock_time_step = stock_time_max / stock_steps
    stock_time = np.arange(0.0, stock_time_max, stock_time_step)
    hrf = op(stock_time)

    # Convolution along time axis
    def op1(x):
        return sig.fftconvolve(x, hrf, mode="full")
    def op2(x):
        return pytensor.map(op1, sequences=[x])[0]
    bold, _ = pytensor.map(op2, sequences=[trace_new.transpose(1, 2, 0)])  ## for loop over nsvar and nnode: trace_new.shape (t, nsvar, nnode) --> (nsvar, nnode, t)
    bold = bold.transpose(2, 0, 1)
    
    %if isinstance(monitor.hrf_kernel, equations.FirstOrderVolterra):
    bold = k_1 * V_0 * (bold - 1.0)
    %endif
    
    bold_idx = np.arange(final_istep-2, len_i, final_istep)[:len_new]
    return timeseries(time=time_steps_new * dt, trace=bold[bold_idx, :, :])
% endif

### Projection monitor
% if isinstance(monitor, Projection):
def monitor_projection_${i}(time_steps, trace, params):
    voi = ${array_input(monitor.voi)}
    dt = ${sim.integrator.dt}

    gain = params.gain
    projection = np.matmul(trace[:, voi, :], gain.T)
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
def monitor_eeg_${i}(time_steps, trace, params):
    % if monitor.reference:
    t, _eeg = monitor_projection_${i}(time_steps, trace, params)
    ref_vec = params.ref_vec
    return timeseries(time=t, trace=_eeg - np.matmul(_eeg, ref_vec))
    %else:
    return monitor_projection_${i}(time_steps, trace, params)
    % endif
% endif

</%def>