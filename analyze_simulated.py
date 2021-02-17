import sys
import numpy as np
import scipy.stats as sps


def downsample(array, n):
    """
    downsamples array sequences over second dimension
    :param array: m x T array of input sequences
    :param n: number of bins to pool
    :return:
    """

    if len(array.shape) == 1:
        array = np.expand_dims(array, 0)
    elif len(array.shape) > 2:
        print('downsample: array dimension too large')
        sys.exit()

    m = array.shape[0]
    T = array.shape[1]
    bins = T // n

    array = array[:, :bins * n].reshape(bins * m, n)

    if m == 1:
        return np.sum(array, axis=1).reshape(m, bins)[0, :]
    else:
        return np.sum(array, axis=1).reshape(m, bins)


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


def Cv2(ts):
    """
    calculates Cv2 (Holt et al. 1996) from spike train of time stamps
    :param ts:
    :return:
    """

    isi = np.diff(ts)
    isis = np.array([isi[:-1], isi[1:]])
    num = np.abs(isis[0] - isis[1])
    denom = np.sum(isis, axis=0)

    return np.mean(2 * num / denom)


def Lv(ts):
    """
    calculates Lv (Shinomoto et al. 2003) from spike train of time stamps
    :param ts:
    :return:
    """

    isi = np.diff(ts)
    isis = np.array([isi[:-1], isi[1:]])
    num = (isis[0] - isis[1])**2
    denom = np.sum(isis, axis=0)**2

    return np.mean(3 * num / denom)


def Lv_list(ts):
    """
    calculates lv if ts is a list of arrays
    :param ts: len(ts) length list of time stamp arrays
    :return: len(ts) x 1 array of Lvs
    """
    Lvs = np.zeros(len(ts))
    for ix in range(len(ts)):
        Lvs[ix] = Lv(ts[ix])

    return Lvs


def Cv2_list(ts):
    """
    calculates Cv2 if ts is a list of arrays
    :param ts: len(ts) length list of time stamp arrays
    :return: len(ts) x 1 array of Lvs
    """
    Cv2s = np.zeros(len(ts))
    for ix in range(len(ts)):
        Cv2s[ix] = Cv2(ts[ix])

    return Cv2s


def Lv_from_spiketrains(S):
    """
    takes m x T array of spiketrains and calculates LV over rows
    :param S: m x T
    :return:
    """

    return Lv_list(ts_from_binary(S))


def Cv2_from_spiketrains(S):
    """
    takes m x T array of spiketrains and calculates Cv2 over rows
    :param S: m x T
    :return:
    """

    return Cv2_list(ts_from_binary(S))


def isi_stats(S):
    """
    takes binary spiketrain, calcs ISIs and performs stats on them
    :param S: n_units x time
    :return:
    """

    n = S.shape[0]
    mu = np.zeros(n)
    sd = np.zeros(n)
    kur = np.zeros(n)
    skew = np.zeros(n)

    for ix in range(n):
        ISI = np.diff(np.where(S[ix])[0])
        mu[ix] = np.mean(ISI)
        sd[ix] = np.std(ISI)
        kur[ix] = sps.kurtosis(ISI, fisher=True, bias=False)
        skew[ix] = sps.skew(ISI, bias=False)

    return mu, sd, kur, skew
