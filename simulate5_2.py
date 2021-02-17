import os
import numpy as np
import main_functions as fns
import cell_models as cmz
from cell_params import params
from datetime import datetime


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

    saving = True

    # SIM PARAMS
    N_trials = 5
    LF = 8192 // 2 + 1
    n_cells = 200
    tau_ref = 2
    ISI = True
    burn = params['burn']
    n_eta = params['n_eta']
    SUMISI = False

    # DECODING PARAMS AND VARIABLES
    mean_rate = np.linspace(0.003, 0.018, 15)
    seed = np.random.uniform(0, 999999, (N_trials, len(mean_rate))).astype(int)
    lag = 9
    a = 40
    thresh = np.arange(3, 30)
    T = 8192 * 123 + lag #int(6e6)

    # PRE-ASSIGN ARRAYS
    pl1 = len(mean_rate)
    pl2 = thresh.shape[0] + 1

    l_e = np.zeros((pl2, N_trials, pl1))
    l_p = np.zeros((pl2, N_trials, pl1))
    lw_e = np.zeros((pl2, LF, N_trials, pl1))
    lw_p = np.zeros((pl2, LF, N_trials, pl1))

    isis = np.zeros((1000, 2, N_trials, pl1))

    for iz in range(N_trials):

        for ix in range(pl1):

            # SIMULATE SRM02 WITH MORE THAN ONE INTRA-BURST SPIKE -- 3 x 1D grid search
            print('running sim... ' + str(iz * pl1 + ix + 1), end='\r')
            sim, stim = cmz.srm02(params, n_cells, mean_rate[ix], tau_ref, T, seed=seed[iz, ix], ISI=ISI, SUMISI=False,
                                    simulator=cmz.method4, return_input=True)
            S, isi = sim
            isis[:, :, iz, ix] = isi

            # LOWER BOUND ANALYSIS
            print('running analysis... ' + str(iz * pl1 + ix + 1), end='\r')

            l_e[:, iz, ix], l_p[:, iz, ix], lw_e[:, :, iz, ix], lw_p[:, :, iz, ix] = fns.lba(np.copy(S), np.copy(stim[0]), np.copy(stim[1]), burn, n_eta, n_cells, thresh, lag=lag, decoder=1)

    print('Done.                                            ')

    if saving:
        save_file = '/' + str(datetime.now().month) + '_' + str(datetime.now().day) + '_' + str(datetime.now().hour)
        if os.path.exists(os.getcwd() + save_file) is False:
            os.mkdir(os.getcwd() + save_file)

        np.save(os.getcwd() + save_file + '/isis', isis)
        np.save(os.getcwd() + save_file + '/lw_e', lw_e)
        np.save(os.getcwd() + save_file + '/lw_p', lw_p)
        np.save(os.getcwd() + save_file + '/l_e', l_e)
        np.save(os.getcwd() + save_file + '/l_p', l_p)
        np.save(os.getcwd() + save_file + '/seed', seed)


