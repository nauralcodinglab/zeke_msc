import os
import numpy as np
import cell_models as cmz
from cell_params import params
from datetime import datetime
from multiprocessing import Pool

import time


def core(params, mean_rate, tau_ref, T, seed, ISI, SUMISI, burn, max_spike_length, n_cells, P, C):
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
    :param thresh:
    :param lag:
    :return:
    """

    # SIMULATE SRM02 WITH MORE THAN ONE INTRA-BURST SPIKE -- 3 x 1D grid search
    S, isis = cmz.srm(params, n_cells, mean_rate, tau_ref, T, seed=seed, ISI=ISI, SUMISI=SUMISI, return_input=False, P=P, C=C)

    SPIKES = S[0, burn:max_spike_length + burn]

    return SPIKES, isis


if __name__ == '__main__':

    start_time = time.time()

    saving = True
    num_pools = 4 #os.cpu_count()

    # SIM PARAMS
    N_trial = 10
    n_cells = 1
    seed = np.random.uniform(0, 999999, N_trial).astype(int)
    ISI = True
    burn = params['burn']
    n_eta = params['n_eta']
    SUMISI = False
    max_spike_length = 480000 * 10

    # DECODING PARAMS AND VARIABLES
    tau_ref = 0
    mean_rate = np.linspace(0.0010, 0.014, N_trial)
    T = max_spike_length #int(4e5) #int(6e6)

    P = np.ones(N_trial) * 10  #np.concatenate((np.ones(10), np.ones(10) * 10))
    C = np.ones(N_trial) * 0.01  #np.concatenate((np.ones(10) * 0.1, np.ones(10) * 0.01))  #NOTE: FIRST Cb WAS 0.5 ORIGINALLY

    # PRE-ASSIGN ARRAYS
    SPIKES = np.zeros((max_spike_length, N_trial)).astype(bool)
    isis = np.zeros((1000, N_trial))

    print('running simulation...    ')
    pool_in = [(params, mean_rate[ix], tau_ref, T, seed[ix], ISI, SUMISI, burn, max_spike_length, n_cells, P[ix], C[ix]) for ix in range(N_trial)]
    with Pool(num_pools) as pool:  # Simulate each cell
        output = pool.starmap(core, pool_in)
        for ix, out in enumerate(output):
            SPIKES[:, ix] = out[0]
            isis[:, ix] = out[1]

    print('Done.' + str(time.time() - start_time) + '                       ')

    if saving:
        save_file = '/fR4srm_' + str(datetime.now().month) + '_' + str(datetime.now().day) + '_' + str(datetime.now().hour) + 'i'
        if os.path.exists(os.getcwd() + save_file) is False:
            os.mkdir(os.getcwd() + save_file)

        np.save(os.getcwd() + save_file + '/tau_ref', tau_ref)
        np.save(os.getcwd() + save_file + '/mean_rate', mean_rate)
        np.save(os.getcwd() + save_file + '/P', P)
        np.save(os.getcwd() + save_file + '/C', C)
        np.save(os.getcwd() + save_file + '/seeds', seed)
        np.save(os.getcwd() + save_file + '/SPIKES', SPIKES)
        np.save(os.getcwd() + save_file + '/ISIs', isis)


