"""
File containing functions required for running experiments with IBM quantum devices
"""
# Imports
import numpy as np
import functools

from . import quantum_device_models


def normalized_L2_error(J_hat, J_num, xi_J):
    return np.linalg.norm((J_num - J_hat)/xi_J, 2)


def model_probabilities_ibmq_np_nd(J_num_nd, xi_J, uvec, w0, w1, w2, tvec_expt, FLAG_control_noise=False):
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """
    J_num = J_num_nd*xi_J

    Jix = J_num[0]
    Jiy = J_num[1]
    Jiz = J_num[2]
    Jzx = J_num[3]
    Jzy = J_num[4]
    Jzz = J_num[5]

    # Parameters that depend on J
    a = Jiz + ((-1) ** uvec) * Jzz
    abs_beta = np.sqrt((Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jzx ** 2) +
                       (Jiy ** 2 + ((-1) ** uvec) * 2 * Jiy * Jzy + Jzy ** 2))

    omega = np.sqrt(a ** 2 + abs_beta ** 2)

    delta = np.arcsin(a / omega)

    den_beta = Jix + ((-1) ** uvec) * Jzx
    num_beta = Jiy + ((-1) ** uvec) * Jzy
    phi = np.arctan2(num_beta, den_beta)  # arg(beta)

    # Time information
    if FLAG_control_noise:
        coeff = 1.50856579e-09
        teff = 6.27739558 / (omega + coeff * omega ** 2)
        tvec = tvec_expt + teff
    else:
        tvec = tvec_expt.copy()

    cos_omega_t = np.cos(omega * tvec)
    sin_omega_t = np.sin(omega * tvec)

    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2

    return fpvec


def log_likelihood_loss(J_nd, X_queries, Y_samples, xi_J=1e6*np.ones(6), xi_t=1e-7,
                        FLAG_readout_noise=False, readout_noise=(0.0, 0.0), FLAG_control_noise=False):
    """
    Can I get this from the ActionSpace itself?

    Inputs:
        J_nd: non-dimensionalized parameters
        xi_J: non-dimensionalization parameters
        X_queries: Queries made to the system of the form {query: number of shots}
        Y_samples: Samples from the system of the form {query: [number of 0 measurements, number of 1 measurements]

    Return log-likelihood loss depending on the different noise models being used
    """
    assert X_queries.keys() == Y_samples.keys()

    # Extract queries and number of shots from Y
    list_queries = list(Y_samples.keys())
    n_queries = len(list_queries)   # Note that this is the number of queries without taking shots of each into account

    # Number of 0s in samples
    Y0_samples = np.array([Y_samples[list_queries[ind]][0] for ind in range(n_queries)]).astype(int)
    Y1_samples = np.array([Y_samples[list_queries[ind]][1] for ind in range(n_queries)]).astype(int)

    # Total number of shots
    n_shots = sum(X_queries.values())

    # Parse the data we have
    mvec = np.array([list_queries[ind][0] for ind in range(n_queries)]).astype(int)
    uvec = np.array([list_queries[ind][1] for ind in range(n_queries)]).astype(int)
    tvec = xi_t*np.array([list_queries[ind][2] for ind in range(n_queries)])

    w0_vec = np.zeros(mvec.size).astype(np.int16)
    w1_vec = np.zeros(mvec.size).astype(np.int16)
    w2_vec = np.zeros(mvec.size).astype(np.int16)

    w0_vec[mvec == 0] = 1
    w1_vec[mvec == 1] = 1
    w2_vec[mvec == 2] = 1

    # Not including decoherence here for the time-being
    prob0_queries = model_probabilities_ibmq_np_nd(J_nd, xi_J, uvec, w0_vec, w1_vec, w2_vec, tvec,
                                                   FLAG_control_noise=FLAG_control_noise)

    # Readout noise
    r0 = r1 = 0.0
    if FLAG_readout_noise:
        r0, r1 = readout_noise

    # log-likelihood of target qubit being 0 or 1
    ll_0 = np.log((1 - r0) * prob0_queries + r1 * (1 - prob0_queries) + 1e-15)
    ll_1 = np.log((1 - r1) * (1. - prob0_queries) + r0 * prob0_queries + 1e-15)

    # log-likelihood of samples
    log_ll = Y0_samples * ll_0 + Y1_samples * ll_1

    loss = -np.sum(log_ll) / n_shots  # negative mean log likelihood

    return loss


def log_likelihood_loss_system_model(J_nd, X_queries, Y_samples, xi_J=1e6*np.ones(6), xi_t=1e-7,
                                     FLAG_readout_noise=False, readout_noise=(0.0, 0.0), FLAG_control_noise=False):
    """
    Can I get this from the ActionSpace itself?

    Inputs:
        J_nd: non-dimensionalized parameters
        xi_J: non-dimensionalization parameters
        X_queries: Queries made to the system of the form {query: number of shots}
        Y_samples: Samples from the system of the form {query: [number of 0 measurements, number of 1 measurements]

    Return log-likelihood loss depending on the different noise models being used
    """
    assert X_queries.keys() == Y_samples.keys()

    J_num = J_nd*xi_J

    # Create Quantum System Model based on the current parameters and noise model
    # Set up noise models for system model
    # Readout noise
    r0 = r1 = 0.0
    if FLAG_readout_noise:
        r0, r1 = readout_noise

    # Control Noise
    qs_noise = {'readout': [r0, r1],
                'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                   FLAG_ibmq_boel=True),
                'grad_control_noise': functools.partial(quantum_device_models.grad_data_driven_teff_noise_model,
                                                        FLAG_ibmq_boel=True)}

    qs_model = quantum_device_models.SystemModel(J_num, xi_J, noise=qs_noise,
                                                 FLAG_readout_noise=FLAG_readout_noise,
                                                 FLAG_control_noise=FLAG_control_noise)

    # Compute log-likelihood
    # loop over queries
    n_shots = 0
    loss = 0
    for query in Y_samples:
        # Extract from query
        m, u, t_nd = query
        t = xi_t*t_nd
        n_config = 2*m + u

        # compute log-likelihood using system model -- below actually returns negative log-likelihood
        ll_0 = qs_model.log_likelihood_loss(0, n_config, t, FLAG_noise=True)
        ll_1 = qs_model.log_likelihood_loss(1, n_config, t, FLAG_noise=True)

        Y0, Y1 = Y_samples[query]

        assert (Y0 + Y1) == X_queries[query]

        # Update loss
        loss += Y0*ll_0 + Y1*ll_1
        n_shots += Y0 + Y1

    loss = loss/n_shots  # mean negative log likelihood

    return loss




