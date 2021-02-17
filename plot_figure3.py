import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# This file plots the output if fR1_sim.py

# Subfunctions ---------------------------------------------------------------------------------------------------------


def adjust_log(isi, bins, new_bins):
    """
    Function takes in ISIs that have been binned a certain way (original BurstRenewal3 binning) and downsamples the
    bin size
    :param isi: (bins, 2) array, ISIs (dim 2 separates events, bursts)
    :param bins: original bin edges
    :param new_bins: new bin edges
    :return: ISIs binned according to new bins
    """
    isi_new = np.zeros((len(new_bins) - 1, 2))

    for ix in range(len(new_bins) - 1):
        index = ((bins >= new_bins[ix]) & (bins < new_bins[ix + 1]))[:-1]

        isi_new[ix, :] = np.sum(isi[index, :], axis=0)

    return isi_new


def get_stuff(load_folder, load_file):
    le = np.mean(np.load(load_folder + load_file + 'l_e.npy'), axis=1)
    lp = np.mean(np.load(load_folder + load_file + 'l_p.npy'), axis=1)
    lwe = np.mean(np.load(load_folder + load_file + 'lwe.npy'), axis=2)
    lwp = np.mean(np.load(load_folder + load_file + 'lwp.npy'), axis=2)
    isis = np.mean(np.load(load_folder + load_file + 'ISIs.npy'), axis=2)
    isis = np.expand_dims(isis, axis=1)
    sum0 = le + lp

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


def plot_main_subplot(gs_, b_, bt_, e_, et_, sum_, sumt_, unimodal_region_, ylabel=True, xlabel=True, letters=None):
    """
    :param gs: gridspec
    :param b:
    :param bt:
    :param e:
    :param et:
    :param sum:
    :param sumt:
    :return:
    """

    ax0 = fig.add_subplot(gs_[0])
    ax0.set_ylim(burst_limits)
    ax0.set_xlim(xlimits)
    ax0.plot(np.arange(0, 60, 1), np.ones(60), '--', color=grey_dark)
    ax0.plot(xaxis, b_ / np.max(bt_), '-o', color=bp_c)
    ax0.set_yticks([0, 0.5, 1])
    #ax0.set_yticks(np.arange(burst_limits[0], burst_limits[1], 0.25))
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.yaxis.set_ticks_position('left')
    #ax0.text(36 * 0.95, 0.5, letters[0], fontsize=fs1, weight='bold')

    ax = fig.add_subplot(gs_[1])
    ax.set_ylim(event_limits)
    ax.set_xlim(xlimits)
    ax.plot(np.arange(0, 60, 1), np.ones(60), '--', color=grey_dark)
    ax.plot(xaxis, e_ / np.max(et_), '-o', color=e_c)
    if ylabel:
        ax.set_ylabel('Normalized Info. Rate', size=fs2)
    #ax.set_yticks(np.arange(event_limits[0], event_limits[1], 0.25))
    ax.set_yticks([0, 0.5, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    #ax.text(36 * 0.95, 0.5, letters[1], fontsize=fs1, weight='bold')

    axi = fig.add_subplot(gs_[2])
    axi.set_ylim(total_limits)
    axi.set_xlim(xlimits)
    if xlabel:
        axi.set_xlabel(r'Refractory Time Constant, $\tau_{rel}$ (ms)', size=fs2)
    axi.plot(np.arange(0, 60, 1), np.ones(60), '--', color=grey_dark)
    axi.plot(xaxis, sum_ / np.max(sumt_), '-o', color=t_c)
    axi.plot(np.arange(unimodal_region_[0], unimodal_region_[1]),
             np.ones(len(np.arange(unimodal_region_[0], unimodal_region_[1]))) * 0.08, color=purple, lw=15)
    #axi.set_yticks(np.arange(total_limits[0], total_limits[1], 0.25))
    axi.set_yticks([0, 0.5, 1])
    axi.spines['right'].set_visible(False)
    axi.spines['top'].set_visible(False)
    #axi.text(36 * 0.95, 0.5, letters[2], fontsize=fs1, weight='bold')

    # -----  HIDE SPINES AND MAKE DIAGONAL THINGS FOR BROKEN AXIS 1-----
    # hide the spines between ax and ax2
    ax0.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax0.set_xticks([])
    ax0.tick_params(labeltop=False)  # don't put tick labels at the top

    # -----  HIDE SPINES AND MAKE DIAGONAL THINGS FOR BROKEN AXIS 2-----
    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    axi.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(labeltop=False)  # don't put tick labels at the top


# Main script ----------------------------------------------------------------------------------------------------------

# LOAD
xaxis = np.concatenate((np.arange(1, 13, 1), np.arange(16, 37, 4)))
zoom_index = [1, 6, 15]
n_param = len(xaxis)
FREQ_MAX_e = int(8192 // 2 * 0.4)
FREQ_MAX_b = int(8192 // 2 * 0.15)

# LOAD MAIN STUFF (RATE CORRECTED)
load_folder = os.getcwd() + '/results/'
load_file = 'figure3_1/'
e, b, ew, bw, sum, et, bt, sumt, isis, f = get_stuff(load_folder, load_file)

load_file2 = 'figure3_2/'
e2, b2, _, _, sum2, et2, bt2, sumt2, isis2, _ = get_stuff(load_folder, load_file2)



# PLOT MAIN RESULTS ----------------------------------------------------------------------------------------------------

# SET UP ERRYTHANG
cutoff_effective = 2
unimodal_region1 = (-1, 7) #in ms
unimodal_region2 = (-1, 9) #in ms
xlimits = (0, 37)

burst_limits = (0, 1.1)
event_limits = (0, 1.1)
total_limits = (0, 1.1)
zoom_labels = [str(xaxis[zoom_index[0]]) + ' ms', str(xaxis[zoom_index[1]]) + ' ms', str(xaxis[zoom_index[2]]) + ' ms' ]

letters1 = ['Ai', 'Aii', 'Aiii']
letters2 = ['Bi', 'Bii', 'Biii']
letters3 = ['Ci', 'Cii', 'Ciii']
letters4 = ['Di', 'Dii', 'Diii']

bp_c = (212/255, 0/255, 0/255) #'#C83737'
b_c = (255/255, 149/255, 0/255) # Define burst colour. '#D3084C'
e_c = (0, 0.5, 1) # Define event colour. '#08D368'
t_c = (0, 0, 0)  # Define total colour. '#000000'
gt_c = (0, 0, 1)  # Define ground truth colour. '#3a8fff'
b_light_c = (1, 0.8, 0.8)  # Define second burst colour. '#CB7390'
e_light_c = (0.3, 0.8, 1)  # Define second event colour. '#99D4B4'
grey = (0.8, 0.8, 0.8)  # Define grey. '#BEBEBE'
grey_dark = (0.5, 0.5, 0.5)
pink = (0.255, 0.192, 0.203)  # '#d142f4'
purple = (0.7, 0.3, 0.7)  # For threshold. '#5024e2'

figbackground_c = 'white'

# Hist bin params
n_bins = 1000
nl_bins = 24
bin_edges = np.arange(0, n_bins + 1, 1)
bins = np.arange(0, n_bins, 1)
bins = np.tile(bins, (2, 1)).transpose()
log_bin_edges = np.logspace(np.log10(2), 3, nl_bins + 1)
log_bins = np.logspace(np.log10(2), 3, nl_bins)
log_bins = np.tile(log_bins, (2, 1)).transpose()

# Fig setup
fig = plt.figure(1, [12, 10], facecolor=figbackground_c)
gsbase = gs.GridSpec(2, 1, hspace=0.4, wspace=0.2)
gs0i = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsbase[0], hspace=0.6, wspace=0.2)
gs0ii = gs.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsbase[1], hspace=0.2, wspace=0.2)
gs10 = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0i[0])
gs20 = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0i[1])
gs_b1 = gs.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0ii[0])
gs_b2 = gs.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0ii[1])
fs1 = 20
fs2 = 20


# Plot mutual information as function of tc_rel

plot_main_subplot(gs_b1, b, bt, e, et, sum, sumt, unimodal_region1, letters=letters3)
plot_main_subplot(gs_b2, b2, bt2, e2, et2, sum2, sumt2, unimodal_region2, ylabel=False, letters=letters4)


# Plot histograms and mutual info as function of freq
for ix, val in enumerate(zoom_index):
    isi = isis[:, 0, :, val]

    isi_log = adjust_log(isi, bin_edges, log_bin_edges)

    # isi = np.zeros(isi0.shape, dtype=float)
    # isi[:, 0] = np.copy(isi0[:, 1])
    # isi[:, 1] = np.copy(isi0[:, 0])

    ax2 = fig.add_subplot(gs10[ix])
    ax2.set_title(r'$\tau_{rel}=$ ' + zoom_labels[ix], size=fs2)
    ax2.set_ylim(0, 0.015)
    ax2.set_xlim(-5, 500)
    ax2.set_xlabel('ISI (ms)', size=fs2)
    ax2.hist(bins, bins=bin_edges, weights=isi, stacked=True, fill=True, color=[e_c, b_c], lw=2, density=True)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    #ax2.spines['left'].set_visible(False)
    ax2.set_yticks([0, 0.01])
    ax2_ins = inset_axes(ax2, width="50%", height="55%")
    ax2_ins.hist(log_bins, bins=log_bin_edges, weights=isi_log, stacked=True, fill=True, color=[e_c, b_c], lw=2, density=True)
    ax2_ins.set_xscale("log")
    ax2_ins.spines['right'].set_visible(False)
    ax2_ins.spines['top'].set_visible(False)
    #ax2_ins.spines['left'].set_visible(False)
    #ax2_ins.set_yticks([])
    #ax2.text((ax2.get_xlim()[0] - ax2.get_xlim()[1]) / 4, 0.014, letters1[ix], fontsize=fs1, weight='bold')
    ax2_ins.set_ylim(0, 0.021)
    if ix == 0:
        ax2.set_ylabel('P(ISI)', size=fs2)

    ax3 = fig.add_subplot(gs20[ix])
    if ix == 0:
        #ax3.set_ylabel(r'$\mathbb{I}_{lb}$ (bits)', size=fs2)
        ax3.set_ylabel('Info. (bits)', size=fs2)
    ax3.set_xlabel('Frequency (Hz)', size=fs2)
    # ax3.semilogx(f, iw_e[val, e_ix, :FREQ_MAX_e], color=e_c)
    # ax3.semilogx(f, iw_b[val, b_ix, :FREQ_MAX_b], color=b_c)
    ax3.plot(f[:FREQ_MAX_e], ew[val, :FREQ_MAX_e], color=e_c)
    ax3.plot(f[:FREQ_MAX_e], bw[val, :FREQ_MAX_e], color=bp_c)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_ylim([0, 3])
    #ax3.text((ax3.get_xlim()[0] - ax3.get_xlim()[1]) / 3.5, 2.7, letters2[ix], fontsize=fs1, weight='bold')
    #if ix > 0:
        #ax3.spines['left'].set_visible(False)
        #ax3.set_yticks([])

plt.show()

print(sum/np.max(sumt))
print(e/np.max(et))
print(b/np.max(bt))

print(sum)
print(e)
print(b)







