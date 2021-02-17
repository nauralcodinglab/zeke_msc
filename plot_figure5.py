import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == '__main__':

    # LOAD FILES FOR VARIABLE BURST
    load_file = '/figure5_1'
    le = np.mean(np.load(os.getcwd() + '/results' + load_file + '/l_e_et.npy'), axis=1)
    lp = np.mean(np.load(os.getcwd() + '/results' + load_file + '/l_p.npy'), axis=1)
    load_file = '/figure5_2'
    le1 = np.mean(np.load(os.getcwd() + '/results' + load_file + '/l_e.npy'), axis=1)
    lp1 = np.mean(np.load(os.getcwd() + '/results' + load_file + '/l_p.npy'), axis=1)
    isis = np.mean(np.load(os.getcwd() + '/results' + load_file + '/isis.npy'), axis=2)

    s_ind = np.argmax(le[:-1, :] + lp[:-1, :], axis=0)
    lp = np.asarray([lp[s_ind[ix], ix] for ix in range(len(s_ind))])
    le = np.asarray([le[s_ind[ix], ix] for ix in range(len(s_ind))])

    lp10 = lp1[-1, :]
    s_ind1 = np.argmax(le1[:-1, :] + lp1[:-1, :], axis=0)
    lp1 = np.asarray([lp1[s_ind1[ix], ix] for ix in range(len(s_ind1))])
    le1 = np.asarray([le1[s_ind1[ix], ix] for ix in range(len(s_ind1))])

    # LOAD FILES FOR VARIABLE RATE
    # load_file2 = '/may12'
    # lwe2 = np.load(os.getcwd() + '/results' + load_file2 + '/lwe_may9.npy')
    # lwp2 = np.load(os.getcwd() + '/results' + load_file2 + '/lwp_may9.npy')
    # le2 = np.load(os.getcwd() + '/results' + load_file2 + '/l_e_may9.npy')
    # lp2 = np.load(os.getcwd() + '/results' + load_file2 + '/l_p_may9.npy')


    # assume at this point that files are formatted so that l_ is 2 x ib and l_2 is 2 x rate


    # SET UP STUFF TO PLOT
    b_c = (1, 0.5, 0.5)  # Define burst colour. '#D3084C'
    b_light_c = (1, 0.8, 0.8)  # Define second burst colour. '#CB7390'
    colour_samples = np.linspace(0.6, 0.95, 2)
    Oranges = [cm.Reds(ix) for ix in colour_samples]
    Purples = [cm.Purples(ix) for ix in colour_samples]
    fs = 20
    fw = 'bold'

    y_axis1i = lp
    y_axis1ii = le + lp
    x_axis1 = np.arange(2, 6, 1)
    width = 0.35

    y_axis2i = lp1 / lp10
    y_axis2ii = le1 + lp1
    x_axis2 = isis.sum(0)[0] / (8192 * 123) * 1000 #np.linspace(0.003, 0.018, 15)  #np.linspace(0.004, 0.014, 10)

    # PLOT STUFF
    figbackground_c = 'white'
    fig = plt.figure(1, [4.5, 7], facecolor=figbackground_c,)

    # First plot

    ax1 = fig.add_subplot(211)
    ax1i = ax1.twinx()
    ax1.bar(x_axis1 - width / 2, y_axis1i, width, color=Oranges[0])
    ax1i.bar(x_axis1 + width / 2, y_axis1ii, width, color=Oranges[1])
    ax1.set_ylabel(r'$\mathbb{I}_{lb}^b$ (bits / s)', color=Oranges[0], size=fs)
    ax1i.set_ylabel(r'$\mathbb{I}_{lb}^b + \mathbb{I}_{lb}^e$ (bits / s)', color=Oranges[1], size=fs)

    ax1.set_xticks(x_axis1)
    ax1.set_xticklabels(x_axis1)
    ax1.set_xlabel('Intra-Burst Spikes', size=fs)
    ax1.set_ylim((np.max(y_axis1i) / 2, np.max(y_axis1i) * 1.02))
    ax1i.set_ylim((np.max(y_axis1ii) / 2, np.max(y_axis1ii) * 1.02))

    # Second plot

    ax2 = fig.add_subplot(212)
    ax2i = ax2.twinx()
    ax2.plot(x_axis2, y_axis2i, color=Purples[0], lw=3)
    ax2.scatter(x_axis2, y_axis2i, color=Purples[0])
    ax2i.plot(x_axis2, y_axis2ii, color=Purples[1], lw=3)
    ax2i.scatter(x_axis2, y_axis2ii, color=Purples[1])
    ax2.set_ylabel(r'$\mathbb{I}_{lb}^b / \mathbb{I}_{lb}^{*b}$', color=Purples[0], size=fs)
    ax2i.set_ylabel(r'$\mathbb{I}_{lb}^b + \mathbb{I}_{lb}^e$ (bits / s)', color=Purples[1], size=fs)

    ax2.set_xlabel('Event Rate (Hz)', size=fs)

    ax2.set_xticks(np.round(np.linspace(2, 17, 4), 3))
    ax2i.set_xticks(np.round(np.linspace(2, 17, 4), 3))



    fig.show()
    fig.tight_layout()








