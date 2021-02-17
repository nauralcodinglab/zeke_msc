import numpy as np
import numpy.random as npr
import sys
from cell_params import params

# MUST CHANGE '3' IN CELL cell_params3 TO THE APPROPRIATE NUMBER FOR THE DESIRED FIGURE

# Functions ------------------------------------------------------------------------------------------------------------


def exp_link(array, loc=0.0, tau=1.0, alpha=1.0):
    """
    Link function used for SRM model that produces rates from values that could be below zero. Element-wise operation.
    :param array: array, input
    :param loc: scalar, location on x-axis. Optional; default = 0
    :param tau: scalar, time constant. Optional; default = 1
    :param alpha: scalar, scaling factor. Optional; default = 1
    :return: array of rates
    """

    return 1/tau * np.exp(alpha * (array - loc))


def sigmoid(array, tau=1.0, alpha=1.0, loc=0.0):
    """
    Element-wise operation
    :param array: array, input array
    :param tau: scalar, time constant. Optional; default = 1
    :param alpha: scalar, scaling factor. Optional; default = 1
    :param loc: scalar, location of sigmoid. Optional; default = 0
    :return: array of outputs
    """

    return alpha/(1 + np.exp(-(array - loc)/tau))


def rel_ref(tau, dt=1, N=100, t=None):
    """
    Calculates relative refractory kernel, eta(t - t'), for SRM model
    :param tau: scalar, time constant with inverse units of t, dt
    :param dt: scalar, time step with units of t. If this is given N must be given. Default is 1
    :param N: scalar, length of kernel. If this is given dt must be given. Default is 100
    :param t: (1 x len(t)) array, vector of times with time of spike = 0. This can be given instead of dt, N
    :return: (1 x len(t)) array, which is rel ref kernel 'eta'
    """

    if t is None:
        t = np.arange(0, N, dt)

    return -np.exp(-t/tau)


def gen_signal(P, N, dt, c=5, num=1):
    """
    Generates an O-U process with expected value of zero
    :param P: scalar, total power of the signal (integral of power spectrum)
    :param N: scalar, length of signal, in time-steps
    :param dt: scalar, width of time-step
    :param c: relaxation constant
    :param num: scalar, gives the number of signals to generate
    :return: x: (num x N) array, of generated signals
    """

    x = np.sqrt(P * (1 - np.exp(-2 * c * dt))) * npr.randn(num, N)
    scale = np.exp(-c * dt)

    for i in range(1, N):
        x[:, i] = scale * x[:, i-1] + x[:, i]

    return x


def calc_isi(eb_view, hist_max, n_cells, burn, n_eta, sum=True):
    """
    calculate ISI for use in methods4 (in burst multiplexing paradigm)
    :param eb_view:
    :param hist_max:
    :param n_cells:
    :param burn:
    :param n_eta:
    :param sum: if True then sum isis over all cells; if False just calculate isi of first cell
    :return:
    """
    if sum:
        n_loop = n_cells
    else:
        n_loop = 1

    # Store only histogram of ISIs to save space
    bins = np.linspace(1, hist_max, 1001)
    isi = np.zeros((1000, 2))
    for i in range(n_loop):
        e_times = np.nonzero(eb_view[0, i, burn:-n_eta])[0]  # Get times for e,b separately
        b_times = np.nonzero(eb_view[1, i, burn:-n_eta])[0]
        index = np.concatenate((np.zeros(len(e_times)), np.ones(len(b_times))))  # Make index
        times = np.array([index, np.concatenate((e_times, b_times))])  # Time and index in one array
        times = np.transpose(times.transpose()[times[1].argsort()])  # Put spikes back in original time order
        diffs = np.diff(np.insert(times[1], 0, -10000))  # Calc isis
        isi[:, 0] += np.histogram(diffs[times[0] == 0][1:], bins)[0]
        isi[:, 1] += np.histogram(diffs[times[0] == 1], bins)[0]

    return isi


def calc_isi0(s_view, hist_max, n_cells, burn, n_eta, sum=True):
    """
    calculate ISI for use in methods4 (in burst multiplexing paradigm)
    :param s_view:
    :param hist_max:
    :param n_cells:
    :param burn:
    :param n_eta:
    :param sum: if True then sum isis over all cells; if False just calculate isi of first cell
    :return:
    """
    if sum:
        n_loop = n_cells
    else:
        n_loop = 1

    # Store only histogram of ISIs to save space
    bins = np.linspace(1, hist_max, 1001)
    isi = np.zeros(1000)
    for i in range(n_loop):
        s_times = np.nonzero(s_view[i, burn:-n_eta])[0]  # Get times for e,b separately
        isi += np.histogram(np.diff(s_times), bins)[0]

    return isi


def method4(seed, eb_view, n_cells, p_burst, p_event, rho_event, n_steps, n_eta, burst_scale, pre_b_ref, pre_e_ref, rho_eta, dt0, burn, n_burst=1, ISI=True, sum_isi=False):

    """
    evaluates my rate method with rel ref for generating spike trains. Outputs burst/event spike trains separately.
    NOTE: BURST SHAPE PARAMETER IS CURRENTLY FIXED AT 1.5
    :param seed:           integer: random seed
    :param eb_view:        array: numpy view storing event/burst spikes, separately, at each time-step
    :param n_cells:        length of second dimenion of eb_view
    :param n_steps:        length of last dimension of eb_view
    :param n_eta:          length of refractory kernel in timesteps
    :param burst_scale:    scale for event to burst ISI (ms),
    :param pre_b_ref:      burst abs. ref. period (ms),
    :param pre_e_ref:      event abs. ref. period (ms),
    :param p_burst:        burst signal (no units),
    :param p_event:        event signal (no units),
    :param rho_event:      event rate (kHz),
    :param rho_eta:        refractory kernel raised to e (unitless)
    :param dt0:            timestep (ms)
    :param burn:           number of timesteps at start of sim to throw out
    :param n_burst:        number of bursts is not used but is here so that method4, 6 have the same input
    :param ISI:            whether or not to calc ISI
    :param sum_isi:        if True then sum isis over all cells; if False just calculate isi of first cell
    :return: eb_view:      Need to return it to work, apparently...(?)
    :return: eb_isi:       ISIs sorted into event, burst based on which were truly intra-burst or not intra-burst
    """
    npr.seed(seed)
    sample_ix = 0  # Define s_view index
    p_eve = np.copy(p_event)
    bursting = np.zeros(n_cells, dtype='bool')
    unif1 = npr.rand(n_cells, n_steps)
    unif2 = npr.rand(n_cells, n_steps)

    while sample_ix < n_steps:                                                 # Loop through timesteps

        eb_view[0, :, sample_ix] = (p_eve[:, sample_ix] > unif1[:, sample_ix]) & (bursting == 0)          # Check if event occurs
        e_ix = eb_view[0, :, sample_ix] == 1
        if np.any(e_ix == 1):                                              # Event occurs

            b_ix = (p_burst[:, sample_ix] > unif2[:, sample_ix]) & e_ix
            if np.any(b_ix == 1):                       # Check if burst occurs
                x = npr.gamma(1.5, burst_scale, np.sum(b_ix)).astype('int') + pre_b_ref   # Burst occurs
                eb_view[1, b_ix, sample_ix + x] = 1
                bursting[b_ix] = 1

                p_eve[e_ix ^ b_ix, sample_ix:sample_ix + n_eta] = 1 - np.exp(
                    -rho_eta * rho_event[sample_ix:sample_ix + n_eta] * dt0)
                p_eve[e_ix ^ b_ix, sample_ix:sample_ix + pre_e_ref] = 0
            else:
                p_eve[e_ix, sample_ix:sample_ix + n_eta] = 1 - np.exp(
                    -rho_eta * rho_event[sample_ix:sample_ix + n_eta] * dt0)
                p_eve[e_ix, sample_ix:sample_ix + pre_e_ref] = 0

        reset = eb_view[1, :, sample_ix - pre_e_ref] == 1
        if np.any(reset):
            bursting[reset] = 0
            p_eve[reset, sample_ix - pre_e_ref:sample_ix - pre_e_ref + n_eta] = 1 - np.exp(
                -rho_eta * rho_event[sample_ix - pre_e_ref:sample_ix - pre_e_ref + n_eta] * dt0)

        sample_ix += 1

    # Calc isi
    if ISI:
        hist_max = 1000  # Store only histogram of ISIs to save space
        isi = calc_isi(eb_view, hist_max, n_cells, burn, n_eta, sum=sum_isi)

        return eb_view, isi
    else:
        return eb_view


def method6(seed, eb_view, n_cells, p_burst, p_event, rho_event, n_steps, n_eta, burst_scale, pre_b_ref, pre_e_ref, rho_eta, dt0, burn, n_burst, ISI=True, sum_isi=False):

    """
    evaluates my rate method with rel ref for generating spike trains. Outputs burst/event spike trains separately.
    NOTE: BURST SHAPE PARAMETER IS CURRENTLY FIXED AT 1.5
    :param seed:           integer: random seed
    :param eb_view:        array: numpy view storing event, intra_burst and burst spikes, separately, at each time-step
    :param n_cells:        length of second dimenion of eb_view
    :param n_steps:        length of last dimension of eb_view
    :param n_eta:          length of refractory kernel in timesteps
    :param burst_scale:    scale for event to burst ISI (ms),
    :param pre_b_ref:      burst abs. ref. period (ms),
    :param pre_e_ref:      event abs. ref. period (ms),
    :param p_burst:        burst signal (no units),
    :param p_event:        event signal (no units),
    :param rho_event:      event rate (kHz),
    :param rho_eta:        refractory kernel raised to e (unitless)
    :param dt0:            timestep (ms)
    :param burn:           number of timesteps at start of sim to throw out
    :param n_burst:        number of spikes in a burst
    :param ISI:            whether or not to calc ISI
    :param sum_isi:        if True then sum isis over all cells; if False just calculate isi of first cell
    :return: eb_view:      Need to return it to work, apparently...(?)
    :return: eb_isi:       ISIs sorted into event, burst based on which were truly intra-burst or not intra-burst
    """
    npr.seed(seed)
    sample_ix = 0  # Define s_view index
    p_eve = np.copy(p_event)
    bursting = np.zeros(n_cells, dtype='bool')
    unif1 = npr.rand(n_cells, n_steps)
    unif2 = npr.rand(n_cells, n_steps)
    c_ind = np.arange(n_cells)

    while sample_ix < n_steps:                                                 # Loop through timesteps

        eb_view[0, :, sample_ix] = (p_eve[:, sample_ix] > unif1[:, sample_ix]) & (bursting == 0)          # Check if event occurs
        e_ix = eb_view[0, :, sample_ix] == 1
        if np.any(e_ix == 1):                                              # Event occurs

            b_ix = (p_burst[:, sample_ix] > unif2[:, sample_ix]) & e_ix
            if np.any(b_ix == 1):                       # Check if burst occurs
                x = np.cumsum(npr.gamma(1.5, burst_scale, (np.sum(b_ix), n_burst)).astype('int') + pre_b_ref, axis=1)   # Burst occurs
                eb_view[1, np.tile(c_ind[b_ix], (n_burst, 1)).transpose(), sample_ix + x] = 1
                eb_view[2, b_ix, sample_ix + x[:, -1]] = 1
                bursting[b_ix] = 1

                p_eve[e_ix ^ b_ix, sample_ix:sample_ix + n_eta] = 1 - np.exp(
                    -rho_eta * rho_event[sample_ix:sample_ix + n_eta] * dt0)
                p_eve[e_ix ^ b_ix, sample_ix:sample_ix + pre_e_ref] = 0
            else:
                p_eve[e_ix, sample_ix:sample_ix + n_eta] = 1 - np.exp(
                    -rho_eta * rho_event[sample_ix:sample_ix + n_eta] * dt0)
                p_eve[e_ix, sample_ix:sample_ix + pre_e_ref] = 0

        reset = eb_view[2, :, sample_ix - pre_e_ref] == 1
        if np.any(reset):
            bursting[reset] = 0
            p_eve[reset, sample_ix - pre_e_ref:sample_ix - pre_e_ref + n_eta] = 1 - np.exp(
                -rho_eta * rho_event[sample_ix - pre_e_ref:sample_ix - pre_e_ref + n_eta] * dt0)

        sample_ix += 1

    # Calc isi
    if ISI:
        hist_max = 1000  # Store only histogram of ISIs to save space
        isi = calc_isi(eb_view[:2, :, :], hist_max, n_cells, burn, n_eta, sum=sum_isi)

        return eb_view, isi
    else:
        return eb_view


def method0(seed, s_view, n_cells, p_event, rho_event, n_steps, n_eta, rho_eta, dt0, burn, ISI=True, sum_isi=False):

    """
    evaluates my rate method with rel ref for generating spike trains. Outputs burst/event spike trains separately.
    NOTE: BURST SHAPE PARAMETER IS CURRENTLY FIXED AT 1.5
    :param seed:           integer: random seed
    :param s_view:         array: numpy view storing spikes at each time-step
    :param n_cells:        length of second dimenion of eb_view
    :param p_event:        event signal (no units),
    :param rho_event:      event rate (kHz),
    :param n_steps:        length of last dimension of eb_view
    :param n_eta:          length of refractory kernel in timesteps
    :param pre_e_ref:      event abs. ref. period (ms),
    :param rho_eta:        refractory kernel raised to e (unitless); INCLUDES ABS. REF PERIOD
    :param dt0:            timestep (ms)
    :param burn:           number of timesteps at start of sim to throw out
    :param ISI:            whether or not to calc ISI
    :param sum_isi:        if True then sum isis over all cells; if False just calculate isi of first cell
    :return: s_view:
    :return: isi:       ISIs sorted into event, burst based on which were truly intra-burst or not intra-burst
    """
    npr.seed(seed)
    sample_ix = 0  # Define s_view index
    p_eve = np.copy(p_event)
    unif1 = npr.rand(n_cells, n_steps)

    while sample_ix < n_steps:                                                 # Loop through timesteps

        s_view[:, sample_ix] = (p_eve[:, sample_ix] > unif1[:, sample_ix])          # Check if event occurs
        e_ix = s_view[:, sample_ix] == 1
        if np.any(e_ix == 1):                                              # Event occurs

            p_eve[e_ix, sample_ix:sample_ix + n_eta] = 1 - np.exp(
                -rho_eta * rho_event[sample_ix:sample_ix + n_eta] * dt0)

        sample_ix += 1

    # Calc isi
    if ISI:
        hist_max = 1000  # Store only histogram of ISIs to save space
        isi = calc_isi0(s_view, hist_max, n_cells, burn, n_eta, sum=sum_isi)

        return s_view, isi
    else:
        return s_view


def srm(params, n_cells, mean_rate, tau_ref, T, seed=None, ISI=True, SUMISI=False, return_input=False, P=None, C=None):
    """
    simulates one srm model cell using method0 with perscribed mean_rate, relative refractory time constant and params
    given in param dictionary. Input is OU process
    :param params:
    :param n_cells
    :param mean_rate:
    :param tau_ref:
    :param T: number of timesteps to simulate in milliseconds
    :param ISI: whether or not to calculate ISI
    :param seed: if seed is None then seed is uniformly selected from integers on 0, 999999
    :param sum_isi: if True then sum isis over all cells; if False just calculate isi of first cell
    :param return_input:
    :return: 1 x (T + burn + n_eta spiketrain), isi 1000 length isi histogram
    """

    if seed is None:
        seed = int(npr.uniform(0, 999999))

    if P is None:
        P_e = params['P_e']
    else:
        P_e = P

    if C is None:
        c_e = params['c_e']
    else:
        c_e = C

    n_eta = params['n_eta']
    dt = params['dt']
    burn = params['burn']
    a_eta = params['a_eta']
    a_rho = params['a_rho']
    th_rho = a_rho * P_e / 2 - np.log(mean_rate) / a_rho
    pre_e_ref = params['Del_e']
    n_steps = T + burn

    if tau_ref > 0:
        rho_eta = np.exp(a_eta * rel_ref(tau_ref, dt, n_eta))
    else:
        rho_eta = np.ones(n_eta)
    if pre_e_ref > 0:
        rho_eta = np.insert(rho_eta, 0, np.zeros(pre_e_ref))[:-pre_e_ref]    # predefine arrays and initialize signals

    event_sig = np.tile(gen_signal(P_e, n_steps + n_eta, dt, c_e, num=1), (n_cells, 1))
    rho_event = exp_link(event_sig, loc=th_rho, alpha=a_rho)
    p_event = 1 - np.exp(-rho_event * dt)
    rho_event = rho_event[0, :]
    s_view = np.zeros((n_cells, n_steps + n_eta))

    if return_input:
        return method0(seed, s_view, n_cells, p_event, rho_event, n_steps, n_eta, rho_eta, dt, burn, ISI=ISI,
                       sum_isi=SUMISI), event_sig[0, burn:-n_eta]
    else:
        return method0(seed, s_view, n_cells, p_event, rho_event, n_steps, n_eta, rho_eta, dt, burn, ISI=ISI,
                       sum_isi=SUMISI)


def srm02(params, n_cells, mean_rate, tau_ref, T, n_burst=1, seed=None, ISI=True, SUMISI=False, simulator=method4, return_input=False, P=None, C=None):
    """
    simulates one srm02 model cell using method4 or 6 with perscribed mean_rate, relative refractory time constant and
    params given in param dictionary. Input is OU process
    :param params:
    :param n_cells:
    :param mean_rate:
    :param tau_ref:
    :param T: number of timesteps to simulate in milliseconds
    :param n_burst number of intra-burst spikes. Default is 1. Obviously only used if method6 is simulator
    :param ISI: whether or not to calculate ISI
    :param seed: if seed is None then seed is uniformly selected from integers on 0, 999999
    :param SUMISI: if True then sum isis over all cells; if False just calculate isi of first cell
    :param simulator: which simulator to use (4, doublet, or 6, fixed num intra-burst spikes). Defaults to 4
    :param return_input:
    :param P: power of input. 1st element = e, 2nd = b
    :param C: time constants for input. 1st element = e, 2nd = b
    :return: 1 x (T + burn + n_eta) spiketrain, isi 1000 length isi histogram
    """

    if seed is None:
        seed = int(npr.uniform(0, 999999))

    if P is None:
        P_e = params['P_e']
        P_b = params['P_b']
    else:
        P_e = P[0]
        P_b = P[1]

    if C is None:
        c_e = params['c_e']
        c_b = params['c_b']
    else:
        c_e = C[0]
        c_b = C[1]

    n_eta = params['n_eta']
    dt = params['dt']
    burn = params['burn']
    a_eta = params['a_eta']
    a_rho = params['a_rho']
    th_rho = a_rho * P_e / 2 - np.log(mean_rate) / a_rho
    pre_e_ref = params['Del_e']
    ta_F = params['ta_F']
    th_F = params['th_F']
    pre_b_ref = params['Del_b']
    burst_scale = 1 / params['b_rate']
    n_steps = T + burn

    rho_eta = np.exp(a_eta * rel_ref(tau_ref, dt, n_eta))  # predefine arrays and initialize signals
    event_sig = np.tile(gen_signal(P_e, n_steps + n_eta, dt, c_e, num=1), (n_cells, 1))
    rho_event = exp_link(event_sig, loc=th_rho, alpha=a_rho)
    p_event = 1 - np.exp(-rho_event * dt)
    rho_event = rho_event[0, :]
    burst_sig = np.tile(gen_signal(P_b, n_steps + n_eta, dt, c_b, num=1), (n_cells, 1))
    p_burst = sigmoid(burst_sig, tau=ta_F, loc=th_F)

    if simulator == method4:
        eb_view = np.zeros((2, n_cells, n_steps + n_eta))
    elif simulator == method6:
        eb_view = np.zeros((3, n_cells, n_steps + n_eta))
    else:
        print('error, srm02: simulator not valid type')
        sys.exit()

    if return_input:
        return simulator(seed, eb_view, n_cells, p_burst, p_event, rho_event, n_steps, n_eta, burst_scale, pre_b_ref,
                         pre_e_ref, rho_eta, dt, burn, n_burst, ISI=ISI, sum_isi=SUMISI), (event_sig[0, burn:-n_eta], burst_sig[0, burn:-n_eta])
    else:
        return simulator(seed, eb_view, n_cells, p_burst, p_event, rho_event, n_steps, n_eta, burst_scale, pre_b_ref,
                         pre_e_ref, rho_eta, dt, burn, n_burst, ISI=ISI, sum_isi=SUMISI)


if __name__ == '__main__':

    # Define params
    n_cells = 3
    mean_rate = 0.01
    tau_ref = 8
    T = int(4e3) #int(6e6)
    seed = 42
    ISI = True
    burn = params['burn']
    n_eta = params['n_eta']
    SUMISI = False

    # SIMULATE SRM
    print('running sim 1...         ', end='\r')
    sim1, stim1 = srm(params, n_cells, mean_rate, tau_ref, T, seed=seed, ISI=ISI, SUMISI=False, return_input=True)
    S1, isi1 = sim1

    # SIMULATE SRM02 WITH DOUBLETS
    print('running sim 2...         ', end='\r')
    sim2, stim2 = srm02(params, n_cells, mean_rate, tau_ref, T, seed=seed, ISI=ISI, SUMISI=False, simulator=method4, return_input=True)
    S2, isi2 = sim2

    # SIMULATE SRM02 WITH MORE THAN ONE INTRA-BURST SPIKE
    print('running sim 3...         ', end='\r')
    sim3, stim3 = srm02(params, n_cells, mean_rate, tau_ref, T, n_burst=3, seed=seed, ISI=ISI, SUMISI=False, simulator=method6, return_input=True)
    S3, isi3 = sim3

    # Plot
    print('plotting...          ', end='\r')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs

    if T > 3000:
        len_x = 3000 # x-axis length in ms
    else:
        len_x = T
    x_axis = np.arange(0, len_x)
    n_bins = 1000
    bin_edges = np.arange(0, n_bins + 1, 1)
    bins = np.arange(0, n_bins, 1)
    b_c = (1, 0.5, 0.5)  # Define burst colour. '#D3084C'
    e_c = (0, 0.5, 1)  # Define event colour. '#08D368'
    grey = 'grey'

    fig = plt.figure(1, [12, 12], facecolor='white')
    gs0 = gs.GridSpec(2, 3, hspace=0.2, wspace=0.2)

    plot_spikes = (S1[0, burn:-n_eta], S2[:, 0, burn:-n_eta], S3[:2, 0, burn:-n_eta])
    plot_isis = (isi1, isi2, isi3)

    for ix in range(3):

        ax = fig.add_subplot(gs0[0, ix])
        ax1 = fig.add_subplot(gs0[1, ix])

        if ix == 0:
            ax.plot(x_axis, plot_spikes[ix][0:len_x], color=grey)
            ax1.hist(bins, bins=bin_edges, weights=plot_isis[ix], fill=True, color=grey, lw=2, density=True)
        elif ix > 0:
            ax.plot(x_axis, plot_spikes[ix][0, 0:len_x], color=e_c)
            ax.plot(x_axis, plot_spikes[ix][1, 0:len_x], color=b_c)
            ax1.hist(np.tile(bins, (2, 1)).transpose(), bins=bin_edges, weights=plot_isis[ix], stacked=True, fill=True, color=[e_c, b_c], lw=2, density=True)

    fig.show()

    print('Done.                        ')

