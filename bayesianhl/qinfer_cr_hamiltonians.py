# imports
import os
import numpy as np
from scipy.linalg import expm

import qinfer as qi
import matplotlib.pyplot as plt


# Going back and forth between the different parameterizations
def transform_parameters(J_num):
    """
    Assuming parameters are "dimensional" here i.e., not non-dimensionalized
    :param J_num:
    :return:
    """
    Jix = J_num[0]
    Jiy = J_num[1]
    Jiz = J_num[2]
    Jzx = J_num[3]
    Jzy = J_num[4]
    Jzz = J_num[5]

    a0 = Jiz + Jzz
    a1 = Jiz - Jzz
    beta0 = (Jix + Jzx) + 1j * (Jiy + Jzy)
    beta1 = (Jix - Jzx) + 1j * (Jiy - Jzy)

    omega0 = np.sqrt(a0 ** 2 + np.abs(beta0) ** 2)
    omega1 = np.sqrt(a1 ** 2 + np.abs(beta1) ** 2)
    delta0 = np.arcsin(a0 / omega0)
    delta1 = np.arcsin(a1 / omega1)
    phi0 = np.angle(beta0)
    phi1 = np.angle(beta1)

    param_array = np.array([omega0, delta0, phi0, omega1, delta1, phi1])

    return param_array


# Ideal Cross-Resonance Hamiltonian with measurement on target qubit
class CRHamiltonian(qi.FiniteOutcomeModel):
    def __init__(self, xi_J=1e6*np.ones(6), xi_t=1e-7):
        # Ref: qinfer-examples/multimodal_testing.ipynb
        self.xi_J = xi_J
        self.xi_t = xi_t
        super(CRHamiltonian, self).__init__()

    @property
    def n_modelparams(self):
        return 6

    @property
    def is_n_outcomes_constant(self):
        return True

    def n_outcomes(self, expparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > -10, modelparams <= -10), axis=1)

    @property
    def expparams_dtype(self):
        return [('meas', 'int', 1), ('prep', 'int', 1), ('time', 'float', 1)]

    def likelihood(self, outcomes, modelparams, expparams):
        """
        This code may currently work with multiple model parameters for only one set of experimental parameters
            :param outcomes:
            :param modelparams:
            :param expparams:
        :return:
        """
        # We first call the superclass method
        super(CRHamiltonian, self).likelihood(outcomes, modelparams, expparams)

        # Define Pr(0 | modelparams; expparams), probability of 0 for each model and each experiment.
        pr0 = np.empty((modelparams.shape[0], expparams.shape[0]))

        Jix_nd, Jiy_nd, Jiz_nd, Jzx_nd, Jzy_nd, Jzz_nd = modelparams.T

        # Writing it in long form to avoid broadcasting issues
        Jix = self.xi_J[0] * Jix_nd
        Jiy = self.xi_J[1] * Jiy_nd
        Jiz = self.xi_J[2] * Jiz_nd
        Jzx = self.xi_J[3] * Jzx_nd
        Jzy = self.xi_J[4] * Jzy_nd
        Jzz = self.xi_J[5] * Jzz_nd

        # (meas, prep, t) -- (M, U, t)
        mvec = expparams['meas'].T
        uvec = expparams['prep'].T
        tvec_nd = expparams['time'].T

        # Get the dimensional time
        tvec = self.xi_t*tvec_nd

        # Start setting the likelihoods
        for idx_experiment in range(expparams.shape[0]):
            # Parameters that depend on J
            a = Jiz + ((-1) ** uvec) * Jzz
            abs_beta = np.sqrt((Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jzx ** 2) +
                               (Jiy ** 2 + ((-1) ** uvec) * 2 * Jiy * Jzy + Jzy ** 2))

            omega = np.sqrt(a ** 2 + abs_beta ** 2)
            cos_omega_t = np.cos(omega * tvec)
            sin_omega_t = np.sin(omega * tvec)

            delta = np.arcsin(a / omega)
            sin_delta = np.sin(delta)
            cos_delta = np.cos(delta)

            den_beta = Jix + ((-1) ** uvec) * Jzx
            num_beta = Jiy + ((-1) ** uvec) * Jzy
            phi = np.arctan2(num_beta, den_beta)  # arg(beta)

            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                        (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

            p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                        (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

            p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

            w0 = np.zeros(mvec.size)
            w1 = np.zeros(mvec.size)
            w2 = np.zeros(mvec.size)

            w0[mvec == 0] = 1
            w1[mvec == 1] = 1
            w2[mvec == 2] = 1

            fpvec = w0 * p0 + w1 * p1 + w2 * p2

            pr0[:, idx_experiment] = fpvec

        return CRHamiltonian.pr0_to_likelihood_array(outcomes, pr0)


# Noisy Cross-Resonance Hamiltonian with measurement on target qubit
class NoisyCRHamiltonian(qi.FiniteOutcomeModel):
    def __init__(self, xi_J=1e6*np.ones(6), xi_t=1e-7, readout_noise=(0.0, 0.0),
                 FLAG_readout_noise=False, FLAG_control_noise=True):
        # Ref: qinfer-examples/multimodal_testing.ipynb

        # Set non-dim parameters
        self.xi_J = xi_J
        self.xi_t = xi_t

        # Set readout noise
        self.readout_noise = readout_noise
        self.FLAG_readout_noise = FLAG_readout_noise

        # Set control noise
        delta_t_imperfect_pulse = lambda omega, coeff0, coeff1: coeff0 / (omega + coeff1 * omega ** 2)

        # parameters for ibmq_boel
        coeff0 = 6.27739558
        coeff1 = 1.50856579e-09

        self.FLAG_control_noise = FLAG_control_noise
        self.control_noise_coeff = [coeff0, coeff1]
        self.control_noise_model = delta_t_imperfect_pulse

        super(NoisyCRHamiltonian, self).__init__()

    @property
    def n_modelparams(self):
        return 6

    @property
    def is_n_outcomes_constant(self):
        return True

    def n_outcomes(self, expparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > -10, modelparams <= -10), axis=1)

    @property
    def expparams_dtype(self):
        return [('meas', 'int', 1), ('prep', 'int', 1), ('time', 'float', 1)]

    def likelihood(self, outcomes, modelparams, expparams):
        """
        This code may currently work with multiple model parameters for only one set of experimental parameters
            :param outcomes:
            :param modelparams:
            :param expparams:
        :return:
        """
        # We first call the superclass method
        super(NoisyCRHamiltonian, self).likelihood(outcomes, modelparams, expparams)

        # Get readout noise term
        r0, r1 = self.readout_noise

        # Define Pr(0 | modelparams; expparams), probability of 0 for each model and each experiment.
        pr0 = np.empty((modelparams.shape[0], expparams.shape[0]))

        Jix_nd, Jiy_nd, Jiz_nd, Jzx_nd, Jzy_nd, Jzz_nd = modelparams.T

        # Writing it in long form to avoid broadcasting issues
        Jix = self.xi_J[0] * Jix_nd
        Jiy = self.xi_J[1] * Jiy_nd
        Jiz = self.xi_J[2] * Jiz_nd
        Jzx = self.xi_J[3] * Jzx_nd
        Jzy = self.xi_J[4] * Jzy_nd
        Jzz = self.xi_J[5] * Jzz_nd

        # (meas, prep, t) -- (M, U, t)
        mvec = expparams['meas'].T
        uvec = expparams['prep'].T
        tvec_nd = expparams['time'].T

        # Get the dimensional time
        tvec = self.xi_t*tvec_nd

        # Start setting the likelihoods
        for idx_experiment in range(expparams.shape[0]):
            # Parameters that depend on J
            a = Jiz + ((-1) ** uvec) * Jzz
            abs_beta = np.sqrt((Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jzx ** 2) +
                               (Jiy ** 2 + ((-1) ** uvec) * 2 * Jiy * Jzy + Jzy ** 2))

            omega = np.sqrt(a ** 2 + abs_beta ** 2)

            # Time information
            if self.FLAG_control_noise:
                teff = self.control_noise_model(omega, self.control_noise_coeff[0], self.control_noise_coeff[1])
                tvec = tvec + teff

            cos_omega_t = np.cos(omega * tvec)
            sin_omega_t = np.sin(omega * tvec)

            delta = np.arcsin(a / omega)
            sin_delta = np.sin(delta)
            cos_delta = np.cos(delta)

            den_beta = Jix + ((-1) ** uvec) * Jzx
            num_beta = Jiy + ((-1) ** uvec) * Jzy
            phi = np.arctan2(num_beta, den_beta)  # arg(beta)

            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                        (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

            p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                        (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

            p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

            w0 = np.zeros(mvec.size)
            w1 = np.zeros(mvec.size)
            w2 = np.zeros(mvec.size)

            w0[mvec == 0] = 1
            w1[mvec == 1] = 1
            w2[mvec == 2] = 1

            fpvec = w0 * p0 + w1 * p1 + w2 * p2

            if self.FLAG_readout_noise:
                fpvec = (1 - r0) * fpvec + r1 * (1 - fpvec)

            pr0[:, idx_experiment] = fpvec

        return NoisyCRHamiltonian.pr0_to_likelihood_array(outcomes, pr0)
