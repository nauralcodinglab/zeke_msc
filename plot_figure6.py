import os
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import gamma_renewal as gr
from cell_params import params as PARAMS
import cell_models as cmz
from scipy.signal import welch
import analyze_simulated as a_s

import matplotlib.gridspec as gs
import matplotlib.cm as cm


def get_stuff(location1, location2, location3):
    spikes1 = np.load(location1 + '/SPIKES.npy')
    isi1 = np.load(location1 + '/ISIs.npy')
    spikes2 = np.load(location2 + '/SPIKES.npy')
    isi2 = np.load(location2 + '/ISIs.npy')
    spikes3 = np.load(location3 + '/SPIKES.npy')
    isi3 = np.load(location3 + '/ISIs.npy')

    return spikes1, spikes2, spikes3, isi1, isi2, isi3


def gamma_stuff():

    # Gamma renewal stuff
    N_trials = 10
    mean_isis = 1 / np.linspace(0.001, 0.012, N_trials)
    shapes = 0.5
    rates = shapes / mean_isis
    N = 200000
    max_spike_length = 480000 * 10

    S_list = []
    for ix, val in enumerate(rates):
        S_list.append(gr.gamma_renewal(val, N, shapes, abs_ref=2, binary=False))

    s3, end_time = gr.binary_from_ts(S_list, use_min=True)
    s3 = s3.T

    if end_time > max_spike_length:
        s3 = s3[:max_spike_length, :]

    return s3, ts_from_binary(s3)


def ts_from_binary(array, data_type=np.float64):
    """
    converts binary array of spiketrains to nested list of time-stamps. Assumes that spiketrains are the last dimension
    and that dt=1
    :param array: 2d array with the last dimension being single, binary spiketrains
    :param data_type: type of data to use in output array
    :return: nested list of arrays of time-stamps
    """

    n1 = array.shape[0]

    out_list = [[] for _ in range(n1)]
    for i in range(n1):
        out_list[i] = (np.where(array[i, :])[0].astype(data_type))

    return out_list


def poisson(rate, N, binary=False, abs=0):
    """
    simulates homogeneous Poisson process
    :param rate: rate
    :param N: number of points to simulate
    :param abs: absolute refractory period in units of time
    :return:
    """

    X = npr.exponential(1 / rate, N) + abs
    ts = np.cumsum(X)

    if binary:
        ts = ts.astype(int)
        S = np.zeros(np.max(ts + 1))
        S[ts] = 1

        return ts, S

    else:
        return ts


def get_poisson_spiketrain(T1=int(5e6)):
    # Get Poisson spike-train for comparison
    pois_abs = 2
    pois_rate = 0.012
    N = 100000
    ts_poisson, S_poisson = poisson(pois_rate, N, binary=True, abs=pois_abs)
    if len(S_poisson) < T1:
        print('error, S_poisson is not long enough')
        sys.exit()

    return S_poisson[:T1], ts_poisson


def get_srm02(T1):

    rate = 0.01
    tau_ref1 = 6
    seed = 42
    if os.path.isfile(os.getcwd() + '/srm02_fR4.npy') is False:
        print('generating S_srm02...')
        S_srm02 = np.any(cmz.srm02(PARAMS, 1, rate, tau_ref1, T1, seed=seed, ISI=False), axis=0)[0]
        np.save(os.getcwd() + '/srm02_fR4.npy', S_srm02)
    else:
        S_srm02 = np.load(os.getcwd() + '/srm02_fR4.npy')

    return S_srm02


def isi_histograms(i_srm02, i_pois, srm_c, pois_c, font, fig1, gs0):

    x_axis = np.arange(0, 1000)
    y_srm02 = np.histogram(i_srm02, np.linspace(1, 1000, 1001), density=True)[0]
    y_compare = np.histogram(i_pois, np.linspace(1, 1000, 1001), density=True)[0]

    ax = fig1.add_subplot(gs0[0])
    ax.plot(x_axis, y_srm02, color=srm_c, lw=2)
    ax.plot(x_axis, y_compare, color=pois_c, ls='--', lw=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([])
    ax.set_xlabel('ISI (ms)', size=font)
    ax.set_ylabel('P(ISI)', size=font)
    ax.set_xlim(-20, 500)


def spike_autocorrs(ac_srm02, ac_poisson, srm_c, pois_c, font, fig1, gs0):

    n_time = 30
    YLIMS = (-0.0005, 0.001)
    axi = fig1.add_subplot(gs0[1])
    axi.plot(np.arange(0, n_time), np.real(ac_srm02[0:n_time]), color=srm_c, lw=2)
    axi.plot(np.arange(0, n_time), np.real(ac_poisson[0:n_time]), color=pois_c, ls='--', lw=2)
    axi.spines['right'].set_visible(False)
    axi.spines['top'].set_visible(False)
    axi.yaxis.set_ticks_position('left')
    axi.set_xlabel(r'$\tau$ (s)', size=font)
    axi.set_ylabel(r'$R_{xx}(\tau)$', size=font)
    axi.set_ylim(YLIMS)
    #axi.set_yticklabels([-5, 0, 5, 10])
    axi.ticklabel_format(axis='y', style='sci', scilimits=(0,0))




if __name__ == '__main__':

    file1 = os.getcwd() + '/figure6_1'
    file2 = os.getcwd() + '/figure6_2'
    file3 = os.getcwd() + '/figure6_3'

    s1, s2, s3, i1, i2, i3 = get_stuff(file1, file2, file3)
    s4, i4 = gamma_stuff(shapes=0.5)

    # SPLIT S2 INTO TWO SETS
    # s3 = s2[:, 10:]
    # i3 = s2[:, 10:]
    s2 = s2[:, :10]
    i2 = i2[:, :10]

    # Get poisson and srm02 spiketrains for histograms
    T1 = int(5e6)
    s_pois, i_pois = get_poisson_spiketrain(T1)
    s_srm02 = get_srm02(T1)
    s_srm02 = s_srm02[500:-650]
    i_srm02 = np.nonzero(s_srm02)[0]
    i_pois = i_pois[-len(i_srm02):]
    i_srm02 = np.diff(i_srm02)
    i_pois = np.diff(i_pois)

    # Autocorrelations
    sample_freq_poisson, pwr_poisson = welch(s_pois, fs=1, nperseg=int(2048 * 2), axis=-1, return_onesided=False)
    ac_poisson = np.fft.ihfft(pwr_poisson, axis=-1)
    sample_freq_srm02, pwr_srm02 = welch(s_srm02, fs=1, nperseg=int(2048 * 2), axis=-1, return_onesided=False)
    ac_srm02 = np.fft.ihfft(pwr_srm02, axis=-1)

    # CALC STATS
    mu1, sd1, _, _ = a_s.isi_stats(s1.T)
    mu2, sd2, _, _ = a_s.isi_stats(s2.T)
    mu3, sd3, _, _ = a_s.isi_stats(s3.T)
    mu4, sd4, _, _ = a_s.isi_stats(s4.T)
    cv1 = sd1 / mu1
    cv2 = sd2 / mu2
    cv3 = sd3 / mu3
    cv4 = sd4 / mu4

    CV2_1 = a_s.Cv2_from_spiketrains(s1.T)
    LV_1 = a_s.Lv_from_spiketrains(s1.T)
    CV2_2 = a_s.Cv2_from_spiketrains(s2.T)
    LV_2 = a_s.Lv_from_spiketrains(s2.T)
    CV2_3 = a_s.Cv2_from_spiketrains(s3.T)
    LV_3 = a_s.Lv_from_spiketrains(s3.T)
    CV2_4 = a_s.Cv2_from_spiketrains(s4.T)
    LV_4 = a_s.Lv_from_spiketrains(s4.T)

    # CALC RATES
    x_axis1 = s1.sum(0) / (480000 * 10) * 1000
    x_axis2 = s2.sum(0) / (480000 * 10) * 1000
    x_axis3 = s3.sum(0) / (480000 * 10) * 1000
    x_axis4 = s4.sum(0) / (480000 * 10) * 1000


    # Set up plots
    colour_samples = np.linspace(0.5, 0.98, 6)
    Purples = [cm.Purples(ix) for ix in colour_samples]
    Reds = [cm.Reds(ix) for ix in colour_samples]
    Blues = [cm.Blues(ix) for ix in colour_samples]
    grey = 'grey'
    red = 'red'
    srm_c = Purples[2]
    pois_c = red
    gamma_c = Blues[2]
    srm_reg_c = Reds[-1]
    srm_reg_c2 = Reds[-3]
    font = 20
    figbackground_c = 'white'
    # fig1 = plt.figure(1, [9.5, 7], facecolor=figbackground_c)
    fig1 = plt.figure(1, [6, 5], facecolor=figbackground_c)
    gsbase1 = gs.GridSpec(2, 1, hspace=0.7, wspace=0.3)
    gs0 = gsbase1[0].subgridspec(1, 2, wspace=0.4)
    #gs1 = gsbase1[2].subgridspec(1, 2, wspace=0.4)
    gs2 = gsbase1[1].subgridspec(1, 2, wspace=0.4)

    # Make histogram and autocorr plots
    isi_histograms(i_srm02, i_pois, srm_c, pois_c, font, fig1, gs0)
    spike_autocorrs(ac_srm02, ac_poisson, srm_c, pois_c, font, fig1, gs0)

    inx = 6
    nbins = np.linspace(0, 700, 120)
    xlims0 = (-10, 700)

    ax2 = fig1.add_subplot(gs2[0])
    line_size = 3
    # ax2.set_ylim((0.8, 1.4))
    # ax2.scatter(x_axis1, CV2_1, label='BSRM', lw=line_size)
    # ax2.scatter(x_axis2, CV2_2, label='SRM, low power', lw=line_size)
    # ax2.scatter(x_axis3, CV2_3, label='SRM, high power', lw=line_size)
    # ax2.scatter(x_axis4, CV2_4, label='Gamma, shape=0.5', lw=line_size)

    #ax2.scatter(x_axis5, CV2_5, label='Gamma, shape=0.5', lw=line_size)
    #ax2.legend()

    ax2.plot(CV2_1, x_axis1, label='BSRM', lw=line_size, color=srm_c)
    ax2.scatter(CV2_1, x_axis1, color=srm_c)
    ax2.plot(CV2_2, x_axis2, label='SRM, low power', lw=line_size, color=srm_reg_c)
    ax2.scatter(CV2_2, x_axis2, color=srm_reg_c)
    ax2.plot(CV2_3, x_axis3, label='SRM, high power', lw=line_size, color=srm_reg_c2)
    ax2.scatter(CV2_3, x_axis3, color=srm_reg_c2)
    ax2.plot(CV2_4, x_axis4, label='Gamma, shape=0.5', lw=line_size, color=gamma_c)
    ax2.scatter(CV2_4, x_axis4, color=gamma_c)
    ax2.axvspan(np.min(CV2_1), np.max(CV2_3), color=srm_reg_c2, alpha=0.1)
    ax2.axvspan(np.min(CV2_4), np.max(CV2_4), color=gamma_c, alpha=0.1)
    ax2.set_xlabel('CV2', size=font)
    ax2.set_ylabel('Rate (Hz)', size=font)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.set_yticks([1, 6, 11])
    ax2.set_xlim(0.97, 1.28)

    ax3 = fig1.add_subplot(gs2[1])
    line_size = 3
    # ax3.set_ylim((0.8, 1.6))
    # ax3.plot(x_axis1, LV_1, label='BSRM', lw=line_size)
    # ax3.plot(x_axis2, LV_2, label='SRM, low power', lw=line_size)
    # ax3.plot(x_axis3, LV_3, label='SRM, high power', lw=line_size)
    # ax3.plot(x_axis4, LV_4, label='Gamma, shape=0.5', lw=line_size)

    #ax3.plot(x_axis5, LV_5, label='Gamma, shape=0.5', lw=line_size)
    #ax3.legend()

    ax3.plot(LV_1, x_axis1, label='BSRM', lw=line_size, color=srm_c)
    ax3.scatter(LV_1, x_axis1, color=srm_c)
    ax3.plot(LV_2, x_axis2, label='SRM, low power', lw=line_size, color=srm_reg_c)
    ax3.scatter(LV_2, x_axis2, color=srm_reg_c)
    ax3.plot(LV_3, x_axis3, label='SRM, high power', lw=line_size, color=srm_reg_c2)
    ax3.scatter(LV_3, x_axis3, color=srm_reg_c2)
    ax3.plot(LV_4, x_axis4, label='Gamma, shape=0.5', lw=line_size, color=gamma_c)
    ax3.scatter(LV_4, x_axis4, color=gamma_c)
    ax3.axvspan(np.min(LV_3), np.max(LV_3), color=srm_reg_c2, alpha=0.1)
    ax3.axvspan(np.min(LV_4), np.max(LV_4), color=gamma_c, alpha=0.1)
    ax3.set_xlabel('LV', size=font)
    #ax3.set_ylabel('Rate (Hz)', size=font)

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.set_yticks([1, 6, 11])
    ax3.set_xlim(0.94, 1.53)

    fig1.savefig('/Users/macbookair/downloads/fR4_updated1.pdf', format='pdf')
    plt.show()



