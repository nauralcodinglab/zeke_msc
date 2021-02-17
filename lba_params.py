
# PARAMS FOR EXPERIMENT fR1 AND SET LAGS
nps = 8192  # usually 8192

params = {

    'nps': nps,  # (ms)
    'nov': nps // 2,  # Total power (integral of psd) of event and burst input signals (mV^2)
    'fs': 1000,  # Type of stochastic process and associated parameter for event/burst inputs USUALLY AT 0.1
    'g_bar': 6,
    'ta_rise': 0.05, # USUALLY AT 0.05
    'ta_decay': 0.5,  # Scale for event link function (1/mV)
    't_sf': 3,  # Time constant for burst link function (mV)
    't_syn': 60,  # Length of synaptic kernel (ms)
    'FREQ_MAX_e': int(nps / 2 * 0.4),  # Threshold for burst link function (mV) USUALLY SET TO: 0.4 (200HZ) => 0.048 for slower
    'FREQ_MAX_b': int(nps / 2 * 0.15),  # Scale for refractory kernel (unitless) USUALLY SET TO: 0.15 (75HZ) => 0.020 for slower
    'lag': 9,  # ms
    'scale_bs': 5,  # Scaling for ratio of numerator and denominator in burst fraction
    'E_ps': 1,  # (mV)
}