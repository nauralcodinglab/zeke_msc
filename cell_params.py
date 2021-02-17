
# PARAMS FOR EXPERIMENT fR1 AND SET_LAGS
params = {

    'dt': 1,  # (ms)
    'P_e': 1,  # Total power (integral of psd) of event and burst input signals (mV^2). Usually 1
    'c_e': 0.1,  # Type of stochastic process and associated parameter for event/burst inputs USUALLY AT 0.1
    'P_b': 6,  # Usually 6
    'c_b': 0.05, # USUALLY AT 0.05
    'a_rho': 0.5,  # Scale for event link function (1/mV)
    'ta_F': 3,  # Time constant for burst link function (mV)
    'th_F': 4.5,  # Threshold for burst link function (mV)
    'a_eta': 7,  # Scale for refractory kernel (unitless)
    'ta_eta': 6,  # Time constant for relative refractory kernel (time-steps)
    'Del_e': 2,  # Abs. ref. period for before event spikes (time-steps)
    'b_rate': 0.15,  # (kHz)
    'Del_b': 2,  # Abs. ref. period before intra-burst spike (time-steps)
    'n_eta': 650,  # Length of rel. ref. period kernel (time-steps)
    'burn': 500,  # (timesteps)
    'n_intra_burst': 3
}
