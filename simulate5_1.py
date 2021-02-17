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
    LF = 4096 // 2 + 1
    n_cells = 200
    mean_rate = 0.01
    tau_ref = 2
    ib_spikes = np.arange(1, 5)
    seed = np.random.uniform(0, 999999, (N_trials, len(ib_spikes))).astype(int)
    ISI = True
    burn = params['burn']
    n_eta = params['n_eta']
    SUMISI = False

    # DECODING PARAMS AND VARIABLES
    l_arange = np.arange(1, 20)
    te_arange = np.arange(5, 45)  # was previously np.arange(3, 20) # 05_20  THESE MUST BE EQUAL
    tb_arange = np.arange(5, 45)  # previously np.arange(5, 20) # 05_20
    lags = np.asarray([l_arange + 10 * x for x in range(len(ib_spikes))]) # updated to better find max 05_21
    MAX_LAG = np.max(lags)
    a = 40
    thresh_e = set_thresholds(te_arange, -a, a / 2) # nte x 3
    thresh_b = np.asarray([set_thresholds(tb_arange, a / val, - a / 2) for _, val in enumerate(ib_spikes)])
    s_t = 5
    T = 4096 * 123 + MAX_LAG #int(6e6)

    # PRE-ASSIGN ARRAYS
    pl1 = len(ib_spikes)
    pl2 = lags.shape[-1]
    pl3 = thresh_e.shape[0] + 1
    pl4 = thresh_b.shape[1] + 1

    l_p0 = np.zeros((N_trials, pl1, pl2))
    lw_p0 = np.zeros((LF, N_trials, pl1, pl2))

    l_e1 = np.zeros((pl3, N_trials, pl1))
    lw_e1 = np.zeros((pl3, LF, N_trials, pl1))

    l_p2 = np.zeros((pl4, N_trials, pl1))
    lw_p2 = np.zeros((pl4, LF, N_trials, pl1))

    for iz in range(N_trials):

        for ix in range(pl1):

            # SIMULATE SRM02 WITH MORE THAN ONE INTRA-BURST SPIKE -- 2 x 1D grid search
            print('running sim... ' + str(iz * pl1 + ix + 1), end='\r')
            sim, stim = cmz.srm02(params, n_cells, mean_rate, tau_ref, T, n_burst=ib_spikes[ix], seed=seed[iz, ix], ISI=ISI, SUMISI=False,
                                    simulator=cmz.method6, return_input=True)
            S, isi = sim

            # LOWER BOUND ANALYSIS
            print('running analysis... ' + str(iz * pl1 + ix + 1), end='\r')

            # FIND BEST LAG
            for iy in range(pl2):
                l_adjust = MAX_LAG - lags[ix, iy]

                if l_adjust == 0:
                    _, l_p0[iz, ix, iy], _, lw_p0[:, iz, ix, iy] = fns.lba_ao(np.copy(S), np.copy(stim[0]), np.copy(stim[1]), burn, n_eta, n_cells, lag=lags[ix, iy])

                else:
                    _, l_p0[iz, ix, iy], _, lw_p0[:, iz, ix, iy] = fns.lba_ao(np.copy(S[:, :, :-l_adjust]), np.copy(stim[0][:-l_adjust]), np.copy(stim[1][:-l_adjust]), burn, n_eta, n_cells, lag=lags[ix, iy])

            # FIND BEST EVENT THRESHOLD
            lag = lags[ix, np.argmax(l_p0[ix, :])]
            l_adjust = MAX_LAG - lag

            if l_adjust == 0:
                l_e1[:, iz, ix], l_p2[:, iz, ix], lw_e1[:, :, iz, ix], lw_p2[:, :, iz, ix] = fns.lba(np.copy(S), np.copy(stim[0]), np.copy(stim[1]), burn, n_eta, n_cells, thresh_e, thresh_b[ix, :, :], s_t, lag=lag, decoder=2)

            else:
                l_e1[:, iz, ix], l_p2[:, iz, ix], lw_e1[:, :, iz, ix], lw_p2[:, :, iz, ix] = fns.lba(np.copy(S[:, :, :-l_adjust]), np.copy(stim[0][:-l_adjust]), np.copy(stim[1][:-l_adjust]), burn, n_eta, n_cells, thresh_e, thresh_b[ix, :, :], s_t, lag=lag, decoder=2)

    print('Done.                                            ')

    if saving:
        save_file = '/' + str(datetime.now().month) + '_' + str(datetime.now().day)
        if os.path.exists(os.getcwd() + save_file) is False:
            os.mkdir(os.getcwd() + save_file)
        np.save(os.getcwd() + save_file + '/lwp_lag', lw_p0)
        np.save(os.getcwd() + save_file + '/l_p_lag', l_p0)

        np.save(os.getcwd() + save_file + '/lwe_et', lw_e1)
        np.save(os.getcwd() + save_file + '/l_e_et', l_e1)

        np.save(os.getcwd() + save_file + '/lwp', lw_p2)
        np.save(os.getcwd() + save_file + '/l_p', l_p2)
        np.save(os.getcwd() + save_file + '/seeds', seed)


