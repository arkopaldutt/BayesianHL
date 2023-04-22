# imports
import os
import numpy as np
from scipy.linalg import expm

import qinfer as qi
import matplotlib.pyplot as plt


# Single Interaction Model
class SingleInteractionModel(qi.FiniteOutcomeModel):

    @property
    def n_modelparams(self):
        return 1

    @property
    def is_n_outcomes_constant(self):
        return True

    def n_outcomes(self, expparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > 0, modelparams <= 1), axis=1)

    @property
    def expparams_dtype(self):
        return [('time', 'float', 1)]

    def likelihood(self, outcomes, modelparams, expparams):
        # We first call the superclass method, which basically
        # just makes sure that call count diagnostics are properly
        # logged.
        super(SingleInteractionModel, self).likelihood(outcomes, modelparams, expparams)

        # Next, since we have a two-outcome model, everything is defined by
        # Pr(0 | modelparams; expparams), so we find the probability of 0
        # for each model and each experiment.
        #
        # We do so by taking a product along the modelparam index (len 1,
        # indicating omega_1), then squaring the result.
        pr0 = 0.5*np.cos(modelparams * expparams['time']) + 0.5

        # Now we use pr0_to_likelihood_array to turn this two index array
        # above into the form expected by SMCUpdater and other consumers
        # of likelihood().
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)


# Simplified CR Hamiltonian
class SimplifiedCRHamiltonian(qi.FiniteOutcomeModel):

    @property
    def n_modelparams(self):
        return 3

    @property
    def is_n_outcomes_constant(self):
        return True

    def n_outcomes(self, expparams):
        return 2

    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > 0, modelparams <= 1), axis=1)

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
        super(SimplifiedCRHamiltonian, self).likelihood(outcomes, modelparams, expparams)

        # Define Pr(0 | modelparams; expparams), probability of 0 for each model and each experiment.
        pr0 = np.empty((modelparams.shape[0], expparams.shape[0]))

        Jix, Jiy, Jzx = modelparams.T

        # (meas, prep, t) -- (M, U, t)
        mvec = expparams['meas'].T
        uvec = expparams['prep'].T
        tvec = expparams['time'].T

        for idx_experiment in range(expparams.shape[0]):
            # beta = Jix + (0+1j) * Jiy + (-1)**uvec * Jzx
            abs_beta = np.sqrt(Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jiy ** 2 + Jzx ** 2)
            cos_abs_beta_t = np.cos(abs_beta * tvec)
            sin_abs_beta_t = np.sin(abs_beta * tvec)
            arg_beta = np.arctan2(Jiy, Jix + ((-1) ** uvec) * Jzx)
            cos_arg_beta = np.cos(arg_beta)
            sin_arg_beta = np.sin(arg_beta)

            p0 = 0.5 * ((cos_abs_beta_t + sin_arg_beta * sin_abs_beta_t) ** 2 + (
                        cos_arg_beta * sin_abs_beta_t) ** 2)
            p1 = 0.5 * ((cos_abs_beta_t - cos_arg_beta * sin_abs_beta_t) ** 2 + (
                        sin_arg_beta * sin_abs_beta_t) ** 2)
            p2 = cos_abs_beta_t ** 2

            w0 = np.zeros(mvec.size)
            w1 = np.zeros(mvec.size)
            w2 = np.zeros(mvec.size)

            w0[mvec == 0] = 1
            w1[mvec == 1] = 1
            w2[mvec == 2] = 1

            fpvec = w0 * p0 + w1 * p1 + w2 * p2

            pr0[:, idx_experiment] = fpvec

        return SimplifiedCRHamiltonian.pr0_to_likelihood_array(outcomes, pr0)
