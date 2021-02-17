import os
import numpy as np
import numpy.random as npr
import main_functions as fns
import cell_models as cmz
from cell_params import params
from datetime import datetime
from multiprocessing import Pool

import time


def core(params, mean_rate, tau_ref, T, seed, ISI, SUMISI, burn, n_eta, n_cells, thresh1, thresh2, s_t, lag):
    """
    multithreads fR45 loop
    :param params:
    :param mean_rate:
    :param tau_ref:
    :param T:
    :param seed:
    :param ISI:
    :param SUMISI:
    :param burn:
    :param max_spike_length:
    :param n_eta:
    :param n_cells:
    :param thresh1: event
    :param thresh2: burst
    :param s_t:
    :param lag:
    :return:
    """

    # SIMULATE SRM02 WITH MORE THAN ONE INTRA-BURST SPIKE -- 3 x 1D grid search
    sim, stim = cmz.srm02(params, n_cells, mean_rate, tau_ref, T, seed=seed, ISI=ISI, SUMISI=SUMISI,
                          simulator=cmz.method6, return_input=True)
    S, isis = sim

    # LOWER BOUND ANALYSIS (MATCHED THRESH)
    pl2 = len(thresh)
    pl3 = len(s_t)
    l_e = np.zeros((pl2 + 1, pl3))
    l_p = np.zeros((pl2 + 1, pl3))
    #lwe = np.zeros((pl2 + 1, LF, pl3))
    #lwp = np.zeros((pl2 + 1, LF, pl3))

    for ix in range(len(s_t)):
        # l_e[:, ix], l_p[:, ix], lwe[:, :, ix], lwp[:, :, ix] = fns.lba(np.copy(S), np.copy(stim[0]), np.copy(stim[1]), burn, n_eta, n_cells, thresh1, thresh2=thresh2, s_t=s_t[ix], lag=lag, decoder=2)
        l_e[:, ix], l_p[:, ix], _, _ = fns.lba(np.copy(S), np.copy(stim[0]), np.copy(stim[1]), burn, n_eta, n_cells, thresh1, thresh2=thresh2, s_t=s_t[ix], lag=lag, decoder=2)
    return isis, l_e, l_p


def set_thresholds(threshes, a, b):
    """
    formats input array of thresholds
    :param thresh: threshes to use
    :param a: scale factor
    :param b: additive factor
    :return:
    """

    thresh_x = np.zeros((len(threshes), 3))
    thresh_x[:, 0] = a
    thresh_x[:, 2] = b

    thresh_x[:, 1] = threshes

    return thresh_x


if __name__ == '__main__':

    start_time = time.time()

    saving = True
    num_pools = 4 #os.cpu_count()

    # SIM PARAMS
    N_trials = 5
    LF = 4096 // 2 + 1
    n_cells = 200
    ISI = True
    burn = params['burn']
    n_eta = params['n_eta']
    SUMISI = False

    # DECODING PARAMS AND VARIABLES
    tau_ref = np.arange(1, 13, 1) #np.concatenate((np.arange(1, 13, 1), np.arange(16, 22, 4)))
    mean_rate = np.array([0.0094, 0.00971178, 0.00994279, 0.01018704, 0.0105, 0.01075, 0.0111, 0.0113, 0.01162305,
                          0.01195677, 0.0124, 0.01273885, 0.0143472 , 0.01633987])[:len(tau_ref)]
    seed = npr.uniform(0, 999999, (N_trials, len(tau_ref))).astype(int)
    lag = 9
    thresh = np.arange(5, 35)
    a = 40
    thresh_e = set_thresholds(thresh, -a, a/2)
    thresh_b = set_thresholds(thresh, a, -a/2)
    T = 4096 * 123 + lag #int(4e5) #int(6e6)
    s_t = np.arange(1, 22, 2)

    # PRE-ASSIGN ARRAYS
    pl1 = len(tau_ref)
    pl2 = len(thresh)
    pl3 = len(s_t)

    l_e = np.zeros((pl2 + 1, pl3, N_trials, pl1))
    l_p = np.zeros((pl2 + 1, pl3, N_trials, pl1))
    #lwe = np.zeros((pl2 + 1, LF, pl3, N_trials, pl1))
    #lwp = np.zeros((pl2 + 1, LF, pl3, N_trials, pl1))
    isis = np.zeros((1000, 2, N_trials, pl1))

    for iy in range(N_trials):

        print('running trial ' + str(iy + 1) + '...                    ')

        pool_in = [(params, mean_rate[ix], tau_ref[ix], T, seed[iy, ix], ISI, SUMISI, burn, n_eta, n_cells, thresh_e, thresh_b, s_t, lag) for ix in range(pl1)]
        with Pool(num_pools) as pool:  # Simulate each cell
            output = pool.starmap(core, pool_in)
            for ix, out in enumerate(output):
                isis[:, :, iy, ix] = out[0]
                l_e[:, :, iy, ix] = out[1]
                l_p[:, :, iy, ix] = out[2]
                #lwe[:, :, :, iy, ix] = out[3]
                #lwp[:, :, :, iy, ix] = out[4]

    print('Done. ' + str(time.time() - start_time) + '                           ')

    if saving:
        save_file = '/' + str(datetime.now().month) + '_' + str(datetime.now().day) + '_' + str(datetime.now().hour)
        if os.path.exists(os.getcwd() + save_file) is False:
            os.mkdir(os.getcwd() + save_file)

        np.save(os.getcwd() + save_file + '/l_e', l_e)
        np.save(os.getcwd() + save_file + '/l_p', l_p)
        np.save(os.getcwd() + save_file + '/seeds', seed)
        np.save(os.getcwd() + save_file + '/ISIs', isis)
        #np.save(os.getcwd() + save_file + '/lwp', lwp)
        #np.save(os.getcwd() + save_file + '/lwe', lwe)


