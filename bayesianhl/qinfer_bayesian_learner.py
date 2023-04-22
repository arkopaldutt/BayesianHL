"""
File containing functions to run learning experiments using Qinfer = SMC + Bayes Risk
"""
# File containing all the functions for running experiments with the passive or active learner

# Add paths
import os
import numpy as np
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt
import functools

import pickle

# Hamiltonian model
from .qinfer_cr_hamiltonians import NoisyCRHamiltonian
from .action_space import ActionSpace
from .utilities_quantum_device import log_likelihood_loss

# estimators
import qinfer as qi

# Risk Heuristic
from .bayes_risk_heuristic import BayesRiskHeuristic


class QinferExperimentRunner(object):
    """
    Set up of learning experiment
    """
    def __init__(self, J_truth_nd, query_space, xi_J, xi_t,
                 FLAG_readout_noise=False, readout_noise=(0.0, 0.0),
                 FLAG_control_noise=True, N_0=972, N_batch=486, max_iter=10, n_particles=5000):

        self.J_truth_nd = J_truth_nd

        # Normalization factors for error computation -- Don't need 1e6 here as we are comparing the non-dim directly
        self.xi_J_error = np.ones(shape=J_truth_nd.shape)

        # Extract parameters relevant to the query space
        self.init_query_space = query_space
        self.moset = query_space['moset']
        self.prepset = query_space['prepset']
        self.time_stamps = query_space['time_stamps']  # This may be changed over the course of the simulation

        # For the simulator -- will change for experimental data
        self._max_shots_ActionSpace = 1e8

        # Noise models
        self.FLAG_readout_noise = FLAG_readout_noise
        self.FLAG_control_noise = FLAG_control_noise

        # Parameters relevant to parameterization and estimation
        self.xi_t = xi_t
        self.xi_J = xi_J

        self.time_stamps_nd = self.time_stamps / self.xi_t

        # Relevant for estimator
        self.n_particles = n_particles

        # Parameters relevant for sampling queries
        self.N_0 = N_0
        self.N_batch = N_batch
        self.max_iter = max_iter

        # Set up quantum system model
        self.readout_noise = readout_noise

        if self.FLAG_control_noise:
            print("Running QinferExperimentRunner with control noise model on!")

        self.sys_model = qi.BinomialModel(NoisyCRHamiltonian(xi_J=self.xi_J, xi_t=self.xi_t,
                                                             FLAG_control_noise=self.FLAG_control_noise))

        # Set up estimator
        Jmin = -10
        Jmax = 10

        prior_J = qi.UniformDistribution([[Jmin, Jmax]] * 6)
        self.prior_bayesian_estimator = prior_J
        bayesian_estimator = qi.SMCUpdater(self.sys_model, n_particles, prior_J)
        self.bayesian_estimator = bayesian_estimator

        # Define the Bayes Risk heuristic so we can start doing this
        risk_sys_model = NoisyCRHamiltonian(xi_J=self.xi_J, xi_t=self.xi_t,
                                            FLAG_control_noise=self.FLAG_control_noise)
        Q = np.ones(6)

        # Get the action space for defining the experimental space of the risk taker
        A_cr = ActionSpace(self.moset, self.prepset, self.time_stamps_nd, self.xi_t,
                           xi_J=self.xi_J, n_shots=self._max_shots_ActionSpace)

        risk_heuristic = BayesRiskHeuristic(bayesian_estimator, risk_sys_model, Q, A_cr.action_space, prior_J)
        self.risk_heuristic = risk_heuristic

    @staticmethod
    def logger(log_file, k_iter, mse_J, loss_J, J_num):
        """
        Inputs:
            log_file:
            k_iter: Iteration number
            mse_J: MSE in J
            loss_J: NOT REALLY INCLUDED HERE! (set to 0)
            J_num:

        Returns nothing
        """
        f_log = open(log_file, "a+")
        f_log.write("%d %f %f %f %f %f %f %f %f\n" % (k_iter, np.sqrt(mse_J), loss_J,
                                                      J_num[0], J_num[1], J_num[2],
                                                      J_num[3], J_num[4], J_num[5]))
        f_log.close()

    def merge_sample_data(self, Y_p, Y_q):
        """
        Inputs
            Y_p: Data from the pool of the form {query: [number of 0s, number of 1s]
            Y_q: Data from most recent query also of the form {query: [number of 0s, number of 1s]}

        Returns:
            Merged data Y_p
        """
        for query_temp in list(Y_q.keys()):
            if query_temp in Y_p.keys():
                Y_p[query_temp] += Y_q[query_temp]
            else:
                Y_p.update({query_temp: Y_q[query_temp]})

        return Y_p

    def update_hamiltonian_estimate(self, X_data):
        """
        Inputs
            X_data: Queries being made of form {query: number of shots}

        Returns:
            Y_data: samples collected of form {query: [number of 0s, number of 1s]
        """
        Y_data = {}
        for query_temp in list(X_data.keys()):
            m, u, t = query_temp
            n_shots = X_data[query_temp]

            experiment_temp = np.array([(m, u, t, n_shots)], dtype=self.sys_model.expparams_dtype)
            outcomes_temp = self.sys_model.simulate_experiment(self.J_truth_nd, experiment_temp)

            # Y samples
            if query_temp in Y_data.keys():
                Y_data[query_temp] += np.array([n_shots - outcomes_temp, outcomes_temp])
            else:
                Y_data[query_temp] = np.array([n_shots - outcomes_temp, outcomes_temp])

            # Update estimate
            self.bayesian_estimator.update(outcomes_temp, experiment_temp)

        return Y_data

    def AL_runner(self, verbose=False, do_plot=False, log_file=None):
        """
        Calculates the Bayesian estimate according to incoming samples
        which are queried through an active learning strategy

        As this is simulated, we just take in values of J, time-stamps and their scalings
        for generating/querying data

        This would be modified when it is a pool of data, etc.

        Inputs:
        env_cr = environment to be used for querying and creating the datasets
        query_opt = algorithm for query optimization (default is random)

        Outputs:
        Returns output
        """
        # Initialize action space
        A_cr = ActionSpace(self.moset, self.prepset, self.time_stamps_nd, self.xi_t,
                           xi_J=self.xi_J, n_shots=self._max_shots_ActionSpace)

        # Variable definitions
        N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
        # loss_num = []
        mse_vec = [] # Current error from the truth if given
        loss_num = []
        J_num = []
        J_nd_num = []
        N_p_vec = []
        q_vec = []
        n_shots_vec = [] # Holds the number of shots available in each query

        # Uniform probability distribution over the pool
        p_U = (1/N_config)*np.ones(N_config)

        n_samples_query_U = round(self.N_0 / N_config)
        set_P = n_samples_query_U * np.ones(N_config)   # Number of shots made so far for each query

        # Update number of queries collected so far
        N_p = self.N_0

        # Create initial dataset using set_P (and not p_U -- generalize later)
        # Initial set of queries
        X_p = A_cr.sample_action_space(set_P, self.N_0, FLAG_query=False)

        # Get data from simulator and update estimator on the fly
        Y_p = self.update_hamiltonian_estimate(X_p)

        # Update action space with actions sampled
        A_cr.update_dict_action_space(X_p, Y_p)

        # Update parameters being tracked
        J_hat_nd = self.bayesian_estimator.est_mean()
        J_num.append(self.xi_J*J_hat_nd)
        J_nd_num.append(J_hat_nd)
        N_p_vec.append(N_p)
        q_vec.append(p_U)
        n_shots_vec.append(N_p)

        # update log-likelihood
        _loss = log_likelihood_loss(J_hat_nd, X_p, Y_p, xi_J=self.xi_J, xi_t=self.xi_t,
                                    FLAG_readout_noise=self.FLAG_readout_noise,
                                    readout_noise=self.readout_noise,
                                    FLAG_control_noise=self.FLAG_control_noise)
        loss_num.append(_loss)

        # Write to log file
        mse_temp = np.linalg.norm(self.J_truth_nd - J_hat_nd, 2) ** 2
        mse_vec.append(mse_temp)

        if log_file is not None:
            self.logger(log_file, 0, mse_temp, loss_num[-1], J_num[-1])

        print('Passive Learning -- Uniform query distribution')

        # Passive Learning
        for k in range(self.max_iter):
            # Update the query constraints
            # N_tot_old = N_p, N_tot = N_p + N_batch, q_old = q_vec[-1]
            # self.set_query_constraints(A_cr, N_p + self.N_batch, self.N_batch, N_p, q_vec[-1])

            # Get best experiment design as suggested by Qinfer+BayesRisk
            q_expt, q, _ = self.risk_heuristic.get_best_query(self.bayesian_estimator)

            # q_vec is 0 everywhere except at q_expt id where it is 1
            set_Q = self.N_batch * q  # The set corresponding to the above experiment
            X_q = A_cr.sample_action_space(set_Q, self.N_batch, FLAG_query=False)

            # Merge query sets and number of queries taken so far
            X_p = A_cr.merge_queries(X_p, X_q)
            N_p += self.N_batch

            # Get data from simulator and update estimator on the fly
            Y_q = self.update_hamiltonian_estimate(X_q)

            # Merge sample data
            Y_p = self.merge_sample_data(Y_p, Y_q)

            # Update action space with actions sampled
            A_cr.update_dict_action_space(X_q, Y_q)

            # Update stuff being tracked
            J_hat_nd = self.bayesian_estimator.est_mean()
            J_num.append(self.xi_J * J_hat_nd)
            J_nd_num.append(J_hat_nd)
            N_p_vec.append(N_p)
            q_vec.append(q)

            # update log-likelihood
            _loss = log_likelihood_loss(J_hat_nd, X_p, Y_p, xi_J=self.xi_J, xi_t=self.xi_t,
                                        FLAG_readout_noise=self.FLAG_readout_noise,
                                        readout_noise=self.readout_noise,
                                        FLAG_control_noise=self.FLAG_control_noise)
            loss_num.append(_loss)

            # Currently meaningless
            n_shots_vec.append(N_p)

            # Write to log file
            mse_temp = np.linalg.norm(self.J_truth_nd - J_hat_nd, 2) ** 2
            mse_vec.append(mse_temp)

            if log_file is not None:
                self.logger(log_file, k+1, mse_temp, loss_num[-1], J_num[-1])

            if verbose:
                print('Done with %d' % k)

        # Results so far
        results = {'loss': loss_num, 'mse': mse_vec, 'J_hat': J_num, 'J_truth_nd': self.J_truth_nd, 'xi_J': self.xi_J,
                   'N_p': N_p_vec, 'q': q_vec, 'n_shots': n_shots_vec, 'data': X_p, 'samples': Y_p, 'A_cr': A_cr}

        return results
