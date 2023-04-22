"""
File containing functions to run learning experiments using a Bayesian optimizer (Sequential Monte Carlo)
"""
# File containing all the functions for running experiments with the passive or active learner

# Add paths
import os
import copy
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

# Active Learning
from . import quantum_device_models

# estimators
import qinfer as qi


class ActiveExperimentRunner(object):
    """
    Set up of learning experiment
    TODO: Add query constraints
    """
    def __init__(self, J_truth_nd, query_space, xi_J, xi_t, active_learner,
                 quantum_device_oracle=None, FLAG_simulator=False,
                 FLAG_query_constraints=False, query_constraints_info=False,
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

        # Setup simulator or dataset oracle
        self.FLAG_simulator = FLAG_simulator
        if FLAG_simulator:
            self.dataset_oracle = None
        else:
            # Number of shots have hopefully been set up correctly by query_constraints_ref below!
            self.dataset_oracle = quantum_device_oracle

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
        if self.FLAG_simulator:
            self.readout_noise = readout_noise
        else:
            self.readout_noise = quantum_device_oracle.readout_noise

        if self.FLAG_control_noise:
            print("Running with control noise model on!")

        self.sys_model = qi.BinomialModel(NoisyCRHamiltonian(xi_J=self.xi_J, xi_t=self.xi_t,
                                                             readout_noise=self.readout_noise,
                                                             FLAG_readout_noise=self.FLAG_readout_noise,
                                                             FLAG_control_noise=self.FLAG_control_noise))

        # Set up estimator
        Jmin = -10
        Jmax = 10

        prior_J = qi.UniformDistribution([[Jmin, Jmax]] * 6)
        self.prior_bayesian_estimator = prior_J
        bayesian_estimator = qi.SMCUpdater(self.sys_model, n_particles, prior_J)
        self.bayesian_estimator = bayesian_estimator

        # Set the active learner
        self.active_learner = active_learner

        # Parameters relevant to query optimization or query constraints in AL procedure -- really messy!
        self.FLAG_query_constraints = FLAG_query_constraints
        if FLAG_query_constraints:
            # TODO: Make sure the query constraints being fed in match the oracle's query constraints
            # N_shots can't be np.inf or np.nan as the array is later changed to an array of integers.
            # Making it a large value for now!
            default_query_constraints_info = {'query_constraints_ref': {'N_shots': 1e8},
                                              'query_optimization_type': 'batch'}

            if query_constraints_info is None:
                self.query_constraints_ref = default_query_constraints_info['query_constraints_ref']
                self.query_optimization_type = default_query_constraints_info['query_optimization_type']
            else:
                if 'query_constraints_ref' in query_constraints_info.keys():
                    self.query_constraints_ref = query_constraints_info['query_constraints_ref']
                else:
                    self.query_constraints_ref = default_query_constraints_info['query_constraints_ref']

                if 'query_optimization_type' in query_constraints_info.keys():
                    self.query_optimization_type = query_constraints_info['query_optimization_type']
                else:
                    self.query_optimization_type = default_query_constraints_info['query_optimization_type']

            if self.query_constraints_ref is not None:
                if 'N_shots' in self.query_constraints_ref.keys():
                    self._max_shots_ActionSpace = self.query_constraints_ref['N_shots']
            else:
                self._max_shots_ActionSpace = 1e8
        else:
            # Number of shots in the ActionSpace for each query if no query constraints present
            self._max_shots_ActionSpace = 1e8

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

    def set_query_constraints(self, A_cr, N_tot, N_batch, N_tot_old, q_old):
        if self.FLAG_query_constraints:
            if self.query_optimization_type == 'batch':
                upper_bound_q = [np.amin([A_cr.action_n_shots[ind] / N_batch, 1.0]) for ind in range(A_cr.N_actions)]
                query_constraints = {'N_tot_old': N_tot_old, 'N_tot': N_tot,
                                     'q_old': q_old, 'upper_bound_q': upper_bound_q, 'FLAG_lower_limits': False}
            else:
                query_constraints = {'N_tot_old': N_tot_old, 'N_tot': N_tot,
                                     'q_old': q_old, 'N_shots': self.query_constraints_ref['N_shots'],
                                     'FLAG_lower_limits': True}

            self.active_learner.update(FLAG_constraints=self.FLAG_query_constraints,
                                       query_constraints=query_constraints)

    def query_optimization(self, A_cr, qs_model, N_p, q_vec, p_U):
        """
        Optimizes query distribution according to the active learner
        :param A_cr:
        :param qs_model:
        :param N_p:
        :param q_vec:
        :param p_U:

        TODO: Add query constraints
        """
        # N_tot_old = N_p, N_tot = N_p + N_batch, q_old = q_vec[-1]
        self.set_query_constraints(A_cr, N_p + self.N_batch, self.N_batch, N_p, q_vec[-1])

        # Query optimization
        q_dist = self.active_learner.optimal_query_distribution(A_cr, qs_model,
                                                                p_ref=p_U, FLAG_verbose_solver=False)

        # Mix with uniform distribution over the valid action set
        N_actions_filtered = len(A_cr.filtered_actions())
        p_U_filtered = (1 / N_actions_filtered) * np.ones(N_actions_filtered)
        lambda_m = 1.0 - 1. / ((N_p) ** (1 / 6))
        q_dist = lambda_m * q_dist + (1 - lambda_m) * p_U_filtered

        # Sample from query distribution q without noting that some queries have already been made
        X_q = A_cr.sample_action_space(q_dist, self.N_batch, actions_list=A_cr.nonempty_actions(),
                                       FLAG_query=True)

        return q_dist, X_q

    def update_hamiltonian_estimate(self, X_data, dict_action_to_index=None):
        """
        Inputs
            X_data: Queries being made of form {query: number of shots}
            dict_action_to_index: current hack for being able to

        Returns:
            Y_data: samples collected of form {query: [number of 0s, number of 1s]
        """
        Y_data = {}
        for query_temp in list(X_data.keys()):
            m, u, t = query_temp
            n_shots = X_data[query_temp]
            experiment_temp = np.array([(m, u, t, n_shots)], dtype=self.sys_model.expparams_dtype)

            if self.FLAG_simulator:
                outcomes_temp = self.sys_model.simulate_experiment(self.J_truth_nd, experiment_temp)
            else:
                ind_action = dict_action_to_index[(m, u, t)]
                samples_temp = self.dataset_oracle.sample_expt_data(ind_action, nsamples=n_shots)
                outcomes_temp = int(np.sum(samples_temp))    # As we require only the number of 1s

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

        # Get data from simulator/dataset and update estimator on the fly
        if self.FLAG_simulator:
            Y_p = self.update_hamiltonian_estimate(X_p)
        else:
            Y_p = self.update_hamiltonian_estimate(X_p, copy.deepcopy(A_cr.dict_action_to_index))

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

        # Create Quantum System Model based on the estimate so far
        # Noise Model -- running without any decoherence here
        qs_noise = {'readout': self.readout_noise,
                    'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                       FLAG_ibmq_boel=True),
                    'grad_control_noise': functools.partial(
                        quantum_device_models.grad_data_driven_teff_noise_model,
                        FLAG_ibmq_boel=True)}

        qs_model = quantum_device_models.SystemModel(J_num[-1], self.xi_J, noise=qs_noise,
                                                     FLAG_readout_noise=self.FLAG_readout_noise,
                                                     FLAG_control_noise=self.FLAG_control_noise)

        print('Active Learning -- HAL-FI/HAL-FIR query distribution')
        # Active Learning (fixed query space for now)
        for k in range(self.max_iter):
            # Update the query constraints
            # N_tot_old = N_p, N_tot = N_p + N_batch, q_old = q_vec[-1]
            # self.set_query_constraints(A_cr, N_p + self.N_batch, self.N_batch, N_p, q_vec[-1])

            # Query optimization
            q, X_q = self.query_optimization(A_cr, qs_model, N_p, q_vec, p_U)

            # Merge query sets and number of queries taken so far
            X_p = A_cr.merge_queries(X_p, X_q)
            N_p += self.N_batch

            # Get data from simulator and update estimator on the fly
            if self.FLAG_simulator:
                Y_q = self.update_hamiltonian_estimate(X_q)
            else:
                Y_q = self.update_hamiltonian_estimate(X_q, copy.deepcopy(A_cr.dict_action_to_index))

            # Merge sample data
            Y_p = self.merge_sample_data(Y_p, Y_q)

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

            # Update SystemModel
            qs_model.update(J_num[-1])

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
