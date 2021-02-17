import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.integrate import simps
from matplotlib import cm as cm
from scipy.ndimage import gaussian_filter1d


# This file plots the output of masters_iv.py

# Subfunctions ---------------------------------------------------------------------------------------------------------


def sigmoid0(x):
    """
    no explanation
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def get_i(iw, f):
    """
    calc corr theory mutual info given arrays of variance. No longer dividing by factor of two
    :param V:
    :param Vs:
    :param n_stim:
    :param f:
    :return:
    """

    return simps(iw[:, 1:], x=f[1:], even='avg', axis=-1)


def get_kernel2(beta, tau, dt=1.0, k='exponential', sig_tau=1):
    """
    calculates kernel vector
    :param beta: param
    :param tau: length of kernel (and spiketrain) in time units; tau = n_k*dt
    :param dt: time step (same units as above) for spiketrain and kernel
    :param k: kernel function. Can be 'exponential' or 'sigmoid'
    :param sig_tau: time constant for sigmoid kernel (hyperparameter)
    :param type: determines if decoding burst or event
    :return:
    """

    if k == 'exponential':
        return np.insert(np.exp(- beta * np.arange(0, tau, dt)), 0, 0)[:-1]
    elif k == 'sigmoid':
        return np.insert(sigmoid0((-(np.arange(0, tau, dt) - beta))/sig_tau), 0, 0)[:-1]
    else:
        print('error get_kernel2: invalid value for k')
        sys.exit()


def main_plots(info, tau_rel, tau_rel_plot, tau_sig, tau_sig_plot, c1, c2, fig, g, font, l_p0):
    """
    plot info as function tau_reaul_sig
    :param info:
    :param tau_rel:
    :param tau_rel_plot:
    :param tau_sig:
    :param tau_sig_plot:
    :param c1:
    :param c2:
    :param fig:
    :param g:
    :param font:
    :param lp_0: fR1 burst info (sharp thresh)
    :return:
    """

    normalize = 29.55010280205601
    ax = fig.add_subplot(g[0])  # Plot info as function of tau_rel for three tau_sigs
    ax.plot(tau_rel, l_p0 / normalize, color='black', lw=2)
    ax.scatter(tau_rel, l_p0 / normalize, color='black')

    for ix in range(3):
        ax.plot(tau_rel, info[tau_sig_plot[ix], :, 0] / normalize, color=c1[ix], lw=2)
        ax.scatter(tau_rel, info[tau_sig_plot[ix], :, 0] / normalize, color=c1[ix])
        #ax.plot(tau_rel, np.ones(len(tau_rel)) * 0.7, color='grey', ls='--')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.set_xlabel('Refractory Time Constant (ms)', size=font)
        ax.set_ylabel(r'$\mathbb{I}_{lb} / \mathbb{I}_{lb}^{*max}$ (bits/s)', size=font)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.5, 1])

    ax2 = fig.add_subplot(g[1])  # Plot info as function of tau_sig for three tau_rels

    for ix in range(3):
        ax2.plot(tau_sig, info[:, tau_rel_plot[ix], 0] / normalize, color=c2[ix], lw=2)
        ax2.scatter(tau_sig, info[:, tau_rel_plot[ix], 0] / normalize, color=c2[ix])
        #ax2.plot(tau_sig, np.ones(len(tau_sig)) * 0.7, color='grey', ls='--')
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.set_xlabel('Decoding Smoothness (ms)', size=font)
        ax2.set_ylabel(r'$\mathbb{I}_{lb} / \mathbb{I}_{lb}^{*max}$ (bits/s)', size=font)
        ax2.set_ylim(0.1, 1.2)
        ax2.set_yticks([0.1, 0.6, 1.1])
        ax2.scatter(0, l_p0[tau_rel_plot[ix]] / normalize, color='black')


def secondary_plots1(tau_sig, tau_sig_plot, c, g, font):
    """
    Plots weight three functions (currently burst ones)
    :param tau_sig:
    :param tau_sig_plot:
    :param c:
    :param font:
    :return:
    """

    ax = fig.add_subplot(g)

    # PARAMS
    a = 40
    beta = 6
    x_end = 12
    dt = 0.01
    x_axis = np.arange(0, x_end, dt)

    # GET KERNELS
    k = np.zeros((3, len(x_axis)))
    for ix in range(3):
        k[ix, :] = get_kernel2(beta, x_end, dt=dt, k='sigmoid', sig_tau=tau_sig[tau_sig_plot[ix]])
    w = sigmoid0(a * k - a / 2)

    # PLOT
    for ix in range(3):
        ax.plot(x_axis[1:], w[ix, 1:], color=c[ix], lw=2, alpha=0.9)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.set_xlabel('ISI (ms)', size=font)
        ax.set_ylabel('Weight', size=font)
        ax.set_yticks([0, 0.5, 1])


def secondary_plots2(isis, tau_rel_plot, c, g, s_filter=1, font=16):
    """
    Plots three distribution functions on one plot
    :param isis:
    :param tau_rel_plot:
    :param c:
    :param g:
    :param s_filter: filter width
    :param font:
    :return:
    """

    ax = fig.add_subplot(g)

    isis = np.sum(isis, axis=1)
    x_axis = np.arange(1, 1001, 1)

    ixs = np.flip(np.arange(3))
    for _, ix in enumerate(ixs):
        #ax.fill_between(x_axis, np.zeros(len(x_axis)), isis[tau_rel_plot[ix], 0, :] / np.sum(isis[tau_rel_plot[ix], 0, :]), color=c[ix], alpha=0.5)
        ax.plot(np.insert(x_axis, 0, 0), np.insert(gaussian_filter1d(isis[:, tau_rel_plot[ix]] / np.sum(isis[:, tau_rel_plot[ix]]), s_filter), 0, 0), color=c[ix], lw=2, alpha=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([])
        ax.set_xlabel('ISI (ms)', size=font)
        ax.set_ylabel('P(ISI)', size=font)
        ax.set_xlim(-12, 300)
        ax.set_ylim(0, 0.015)


def best_thresh(l_e, l_p):
    """
    gets best thresh from info arrays
    :param l_e: (n_sig, n_rel, n_thresh)
    :param l_p: (n_sig, n_rel, n_thresh)
    :return:
    """

    n_sig = l_e.shape[0]
    n_rel = l_e.shape[1]
    e = np.zeros((n_sig, n_rel, 2))
    b = np.zeros((n_sig, n_rel, 2))
    sum0 = l_e + l_p

    for ix in range(n_sig):  # USING SAME THRESH PROTOCOL
        for iy in range(n_rel):
            ind = np.argmax(sum0[ix, iy, :-1])
            e[ix, iy, 0] = l_e[ix, iy, ind]
            b[ix, iy, 0] = l_p[ix, iy, ind]

    e[:, :, 1] = l_e[:, :, -1]
    b[:, :, 1] = l_p[:, :, -1]

    return e, b


def get_stuff(load_folder, load_file):
    le = np.mean(np.load(load_folder + load_file + 'l_e.npy'), axis=1)
    lp = np.mean(np.load(load_folder + load_file + 'l_p.npy'), axis=1)
    lwe = np.mean(np.load(load_folder + load_file + 'lwe.npy'), axis=2)
    lwp = np.mean(np.load(load_folder + load_file + 'lwp.npy'), axis=2)
    isis = np.mean(np.load(load_folder + load_file + 'ISIs.npy'), axis=2)
    isis = np.expand_dims(isis, axis=1)
    sum0 = le + lp
    n_param = 18

    n_thresh = le.shape[0] - 1
    T = 8.192
    f = np.arange(0, 500 + 1 / T, 1 / T)
    n_f = len(f)

    # Organize data
    e = np.zeros(n_param)
    b = np.zeros(n_param)
    ew = np.zeros((n_param, n_f))
    bw = np.zeros((n_param, n_f))
    sum = np.zeros(n_param)

    # NOTE:
    # BELOW INDEXES ARE np.array([ 7.,  7.,  8.,  8.,  9., 10., 11., 12., 12., 13., 13., 14., 16., 17., 17., 18., 19., 19.])
    # Corresponding thresh values: np.array([12., 12., 13., 13., 14., 15., 16., 17., 17., 18., 18., 19., 21., 22., 22., 23., 24., 24.])
    for ix in range(n_param):  # USING SAME THRESH PROTOCOL
        ind = np.argmax(sum0[:-1, ix])
        e[ix] = le[ind, ix]
        b[ix] = lp[ind, ix]
        ew[ix, :] = lwe[ind, :, ix]
        bw[ix, :] = lwp[ind, :, ix]
        sum[ix] = sum0[ind, ix]

    et = le[-1, :]
    bt = lp[-1, :]
    sumt = sum0[-1, :]

    return e, b, ew, bw, sum, et, bt, sumt, isis, f


# Main script ----------------------------------------------------------------------------------------------------------

# DEFINE COLOURS
b_c = (1, 0.5, 0.5) # Define burst colour. '#D3084C'
e_c = (0, 0.5, 1) # Define event colour. '#08D368'
t_c = (0, 0, 0)  # Define total colour. '#000000'
gt_c = (0, 0, 1)  # Define ground truth colour. '#3a8fff'
b_light_c = (1, 0.8, 0.8)  # Define second burst colour. '#CB7390'
e_light_c = (0.3, 0.8, 1)  # Define second event colour. '#99D4B4'
grey = (0.8, 0.8, 0.8)  # Define grey. '#BEBEBE'
grey_dark = (0.5, 0.5, 0.5)
pink = (0.255, 0.192, 0.203)  # '#d142f4'
purple = (0.7, 0.3, 0.7)  # For threshold. '#5024e2'

colour_samples = np.linspace(0.3, 1, 3)
Blues = [cm.Blues(ix) for ix in colour_samples]
Oranges = [cm.Reds(ix) for ix in colour_samples]
color2 = [cm.Purples(ix) for ix in colour_samples]
font = 20

figbackground_c = 'white'

# DEFINE LOAD FOLDERS
load_folder = os.getcwd() + '/results/'
load_file = 'figure4' #'04_17/lag6/' #'1_23/' #'1_14/' #'01_12/lag6/'  # lb
load_file0 = 'figure3_1'

# DEFINE PARAMS
tau_rel = np.arange(1, 13, 1)
n_rel = len(tau_rel)
tau_sig = np.arange(1, 22, 2) #np.array([0.1, 0.5, 0.8]) #np.arange(1, 11, 1)
n_sig = len(tau_sig)
thresh_b = np.array([3., 3., 4., 5., 6., 7., 7., 8., 8., 8., 9., 10])  # OPTIMAL THRESHOLDS AS PER 02_13 SIM (BUT NEED TO DOUBLE CHECK)

tau_rel_plot = [4, 7, 11]
tau_sig_plot = [2, 5, 8]

T = 8.192 #16.384, 2.048
f = np.arange(0, 500 + 1 / T, 1 / T)
n_f = len(f)
FREQ_MAX_e = int(4097 * 0.3) #1228
FREQ_MAX_b = int(4097 * 0.10) #491
s_filter = 1.5

#n_t = 2

# LOAD FILES AND FORMAT
l_e = np.mean(np.load(load_folder + load_file + 'l_e.npy'), axis=-2)
l_p = np.mean(np.load(load_folder + load_file + 'l_p.npy'), axis=-2)
# l_e = np.zeros((n_sig, n_rel, n_t))
# l_p = np.zeros((n_sig, n_rel, n_t))
l_e = np.moveaxis(l_e, 0, -1)
l_p = np.moveaxis(l_p, 0, -1)

# for i in range(n_sig):
#     for j in range(n_rel):
#         l_e[i, j, :] = get_i(lw_e[i, j, :, :FREQ_MAX_e], f[:FREQ_MAX_e])
#         l_p[i, j, :] = get_i(lw_p[i, j, :, :FREQ_MAX_b], f[:FREQ_MAX_b])

l_e, l_p = best_thresh(l_e, l_p)

isis = np.mean(np.load(load_folder + load_file + 'ISIs.npy'), axis=-2)

# ind = 0
# plt.plot(tau_rel, l_e[:, :, ind].transpose())
# plt.show()
# plt.plot(tau_rel, l_p[:, :, 0].transpose())
# plt.plot(tau_rel, l_p[:, :, 1].transpose())
# plt.show()

e0, b0, ew0, bw0, sum0, et0, bt0, sumt0, isis0, f0 = get_stuff(load_folder, load_file0)



# WRITE PLOTS ----------------------------------------------------------------------------------------------------------
fig = plt.figure(1, [8, 5.5], facecolor=figbackground_c)
gsbase = gs.GridSpec(1, 3, hspace=0, wspace=0.5)
gs00 = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsbase[0, 0], hspace=0.4, wspace=0.3)
gs01 = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsbase[0, 1:], hspace=0.4, wspace=0.3)
# gs10 = gs.GridSpecFromSubplotSpec(1, 1, subplot_spec=gsbase[1, 0], hspace=0.6, wspace=0.2)
# gs11 = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsbase[0, 1:], hspace=0.6, wspace=0.2)


# PLOT FIGURE 00
c = Oranges
secondary_plots1(tau_sig, tau_sig_plot, c, gs00[0], font)

c = color2
secondary_plots2(isis, tau_rel_plot, c, gs00[1], s_filter, font)


# PLOT FIGURE 01
c1 = Oranges
c2 = color2
main_plots(l_p, tau_rel, tau_rel_plot, tau_sig, tau_sig_plot, c1, c2, fig, gs01, font, b0[:12])


fig.show()

