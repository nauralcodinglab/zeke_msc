import numpy as np
import numpy.random as npr
from scipy.stats import gamma as G
import analyze_simulated as a_s
import matplotlib.pyplot as plt


def plot_gamma_renewal(rate, N, shapes):
    """
    plot CV and LV for gamma renewal process with given rate, shapes. Also plot gamma distributions with same rate and
    given shapes
    :param rate:
    :param shape:
    :param N:
    :param shapes:
    :return:
    """


    CV2 = []
    LV = []
    for val in shapes:
        _, SPIKES = gamma_renewal(rate, N, val, abs_ref=2, binary=True)
        SPIKES = SPIKES[:, np.newaxis]

        CV2.append(a_s.Cv2_from_spiketrains(SPIKES.T))
        LV.append(a_s.Lv_from_spiketrains(SPIKES.T))

    plt.plot(shapes, CV2, lw=3, color='red', label='CV2')
    plt.plot(shapes, LV, lw=3, color='blue', label='LV')
    plt.legend()
    plt.show()

    shapes_plot = [0.1, 0.5, 1, 1.5, 2]
    x = np.arange(0, 400)
    alpha = [1, 1, 0.4, 0.4, 0.4]
    for ix, val in enumerate(shapes_plot):
        plt.plot(x, G.pdf(x=x, a=val, scale=1 / 0.01), lw=2.5, label=f'Shape: {val}', alpha=alpha[ix])
    plt.legend()
    plt.ylim(-0.0005, 0.04)
    plt.show()


def  gamma_renewal(rate, N, shape_, abs_ref=0, binary=False):
    """
    simulates homogeneous gamma renewal process
    :param rate: rate
    :param N: number of points to simulate
    :return:
    """

    X = npr.gamma(shape=shape_, scale=1 / rate, size=N)
    X += abs_ref
    ts = np.cumsum(X)

    if binary:
        ts = ts.astype(int)
        S = np.zeros(np.max(ts + 1))
        S[ts] = 1

        return ts, S

    else:
        return ts


def binary_from_ts(spike_list, use_min=True):
    """
    produces binary array from spike list of time stamps, where times must be in milliseconds
    :param spike_list:
    :return:
    """

    n_units = len(spike_list)
    max_list = [np.max(spike_list[ix]) for ix in range(n_units)]
    max_t = int(np.max(max_list))

    if use_min:
        end_t = int(np.min(max_list))
    else:
        end_t = max_t

    spikes = np.zeros((n_units, max_t + 1))

    for ix in range(n_units):
        index = np.asarray(spike_list[ix]).astype(int)
        spikes[ix, index] = 1

    return spikes[:, :end_t], end_t


if __name__ == '__main__':

    # Plot gamma renewal stuff
    N_trials = 10
    mean_isis = 1 / np.linspace(0.001, 0.012, N_trials)
    shapes = 0.5
    rates = shapes / mean_isis
    N = 200000
    max_spike_length = 480000 * 10

    S_list = []
    for ix, val in enumerate(rates):
        S_list.append(gamma_renewal(val, N, shapes, abs_ref=2, binary=False))

    S, end_time = binary_from_ts(S_list, use_min=True)
    S = S.T

    if end_time > max_spike_length:
        S = S[:max_spike_length, :]

    plot_stuff = False
    if plot_stuff:
        inx = 9
        CV2 = a_s.Cv2_from_spiketrains(S.T)
        LV = a_s.Lv_from_spiketrains(S.T)

        times = np.where(S[:, inx])[0]
        isi = np.diff(times)

        plt.hist(isi, bins=100)
        plt.show()

        print(f'CV2: {CV2}')
        print(f'LV: {LV}')
        print(f'Total # of spikes for neuron 1: {np.sum(S[:, inx])}')
        print(f'Approximate Rate of neuron 1: {np.sum((S[:, inx])) / max_spike_length * 1000}')

