"""
File containing functions to run learning experiments using Qinfer = SMC + Bayes Risk
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
from . import quantum_device_oracle

# estimators
import qinfer as qi
from .bayesian_learner_dataset import quick_bayesian_estimate

# Risk Heuristic
from .bayes_risk_heuristic import BayesRiskHeuristic


class QinferExperimentRunner(object):
    """
    Set up of learning experiment
    """
    def __init__(self, J_truth_nd, query_space, xi_J, xi_t,
                 dataset_oracle=None, FLAG_simulator=True, FLAG_testing=False,
                 FLAG_readout_noise=False, readout_noise=(0.0, 0.0),
                 FLAG_control_noise=True, N_0=972, N_batch=486, max_iter=10, max_shots_query=1e8,
                 n_particles=5000):
        """
        Inputs:
        :param J_truth_nd: non-dim parameters
        :param query_space:
        :param xi_J:
        :param xi_t:
        :param quantum_device_oracle:
        :param FLAG_simulator:
        :param FLAG_testing: FLAG to switch on creating a testing dataset and then get errors against that
        :param FLAG_readout_noise:
        :param readout_noise:
        :param FLAG_control_noise:
        :param N_0:
        :param N_batch:
        :param max_iter:
        :param n_particles:
        """
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
            # For the simulator -- will change for experimental data
            self._max_shots_ActionSpace = max_shots_query
            self.dataset_oracle = None
        else:
            # hard coded at the moment
            self._max_shots_ActionSpace = max_shots_query
            self.dataset_oracle = dataset_oracle

            assert self._max_shots_ActionSpace == np.amin(dataset_oracle.expt_data['n_samples_actions'])

        print("shots per query set to %d" % self._max_shots_ActionSpace)

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
            self.readout_noise = dataset_oracle.readout_noise

        if self.FLAG_control_noise:
            print("Running QinferExperimentRunner with control noise model on!")

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

        # Define the Bayes Risk heuristic so we can start doing this
        risk_sys_model = NoisyCRHamiltonian(xi_J=self.xi_J, xi_t=self.xi_t,
                                            readout_noise=self.readout_noise,
                                            FLAG_readout_noise=self.FLAG_readout_noise,
                                            FLAG_control_noise=self.FLAG_control_noise)
        Q = np.ones(6)

        # Get the action space for defining the experimental space of the risk taker
        A_cr = ActionSpace(self.moset, self.prepset, self.time_stamps_nd, self.xi_t,
                           xi_J=self.xi_J, n_shots=self._max_shots_ActionSpace)

        risk_heuristic = BayesRiskHeuristic(bayesian_estimator, risk_sys_model, Q, A_cr.action_space, prior_J)
        self.risk_heuristic = risk_heuristic

        # Assessing performance against a testing dataset
        self.FLAG_testing = FLAG_testing

    @staticmethod
    def logger(log_file, k_iter, mse_J, loss_J, J_num, num_mini_batches):
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
        f_log.write("%d %f %f %f %f %f %f %f %f %d\n" % (k_iter, np.sqrt(mse_J), loss_J,
                                                      J_num[0], J_num[1], J_num[2],
                                                      J_num[3], J_num[4], J_num[5], num_mini_batches))
        f_log.close()

    def logger_experimental_data(self, log_file, mse_train_vec, loss_vec, J_vec, mse_test_vec, num_mini_batches_vec):
        """
        Logger special to experimental dataset or wherever there is a notion of testing dataset and not ground truth
        Ref: https://stackoverflow.com/questions/2769061/how-to-erase-the-file-contents-of-text-file-in-python

        Inputs:
            log_file:
            k_iter: Iteration number
            mse_J: MSE in J
            loss_J:
            J_num:

        Returns nothing
        """
        f_log = open(log_file, "a")
        f_log.seek(0)
        f_log.truncate()

        for k_iter in range(self.max_iter + 1):
            J_num = np.copy(J_vec[k_iter])
            f_log.write("%d %f %f %f %f %f %f %f %f %f %d\n" % (k_iter,
                                                                np.sqrt(mse_train_vec[k_iter]),
                                                                loss_vec[k_iter],
                                                                J_num[0], J_num[1], J_num[2],
                                                                J_num[3], J_num[4], J_num[5],
                                                                np.sqrt(mse_test_vec[k_iter]),
                                                                num_mini_batches_vec[k_iter]))
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

    def outcomes_queries(self, X_data, dict_action_to_index=None):
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
                outcomes_temp = int(np.sum(samples_temp))  # As we require only the number of 1s

            # Y samples
            if query_temp in Y_data.keys():
                Y_data[query_temp] += np.array([n_shots - outcomes_temp, outcomes_temp])
            else:
                Y_data[query_temp] = np.array([n_shots - outcomes_temp, outcomes_temp])

        return Y_data

    def update_hamiltonian_estimate(self, X_data, Y_data):
        """
        Inputs
            X_data: Queries being made of form {query: number of shots}
            Y_data: Outcomes of queries made from X_data; of form {query: [number of 0s, number of 1s]}
            dict_action_to_index: current hack for being able to

        Returns:
            Y_data: samples collected of form {query: [number of 0s, number of 1s]}
        """
        assert X_data.keys() == Y_data.keys()

        for query_temp in list(X_data.keys()):
            m, u, t = query_temp
            n_shots = X_data[query_temp]
            experiment_temp = np.array([(m, u, t, n_shots)], dtype=self.sys_model.expparams_dtype)

            assert n_shots == np.sum(Y_data[query_temp])

            # Y samples
            outcomes_temp = Y_data[query_temp][1]

            # Update estimate
            self.bayesian_estimator.update(outcomes_temp, experiment_temp)

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
        num_mini_batches_vec = [] # Holds the number of mini-batches done within one batch

        # Uniform probability distribution over the pool
        p_U = (1/N_config)*np.ones(N_config)

        n_samples_query_U = round(self.N_0 / N_config)
        set_P = n_samples_query_U * np.ones(N_config)   # Number of shots made so far for each query

        # Update number of queries collected so far
        N_p = self.N_0

        # Create initial dataset using set_P (and not p_U -- generalize later)
        # Initial set of queries
        X_p = A_cr.sample_action_space(set_P, self.N_0, FLAG_query=False)

        # Get data from simulator/dataset
        if self.FLAG_simulator:
            Y_p = self.outcomes_queries(X_p)
        else:
            Y_p = self.outcomes_queries(X_p, copy.deepcopy(A_cr.dict_action_to_index))

        # Update estimator with queried examples
        self.update_hamiltonian_estimate(X_p, Y_p)

        # Update parameters being tracked
        J_hat_nd = self.bayesian_estimator.est_mean()
        J_num.append(self.xi_J*J_hat_nd)
        J_nd_num.append(J_hat_nd)
        N_p_vec.append(N_p)
        q_vec.append(p_U)
        n_shots_vec.append(N_p)
        num_mini_batches_vec.append(1)

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
            self.logger(log_file, 0, mse_temp, loss_num[-1], J_num[-1], num_mini_batches_vec[-1])

        print('Active Learning with Qinfer -- Bayes risk query distribution')

        # Passive Learning
        for k in range(self.max_iter):
            # Divide batch into mini-batches according to how many required to get one batch worth of shots
            num_mini_batches = 0
            nshots_mini_batches = 0
            q_mini_batch = np.zeros(A_cr.N_actions)

            while nshots_mini_batches < self.N_batch:
                # Get best batch of queries as suggested by Qinfer+BayesRisk
                valid_queries_indices = copy.deepcopy(A_cr.nonempty_actions())
                n_actions_left = copy.deepcopy(A_cr.action_n_shots[valid_queries_indices])

                # q is 0 everywhere except at the current best query index in X_q where it is nshots
                X_q, q, nshots_temp, _ = self.risk_heuristic.get_minibatch_queries(self.N_batch - nshots_mini_batches,
                                                                               self.bayesian_estimator,
                                                                               valid_queries_indices, n_actions_left)

                # Update shots left as we didn't use the sample_action_space method of A_cr
                A_cr.update_shots(X_q, nshots_temp)

                # Merge query sets and number of queries taken so far
                X_p = A_cr.merge_queries(X_p, X_q)

                # Get data from simulator/dataset and update estimator on the fly
                if self.FLAG_simulator:
                    Y_q = self.outcomes_queries(X_q)
                else:
                    Y_q = self.outcomes_queries(X_q, copy.deepcopy(A_cr.dict_action_to_index))

                # Update estimate with queried examples
                self.update_hamiltonian_estimate(X_q, Y_q)

                # Merge sample data
                Y_p = self.merge_sample_data(Y_p, Y_q)

                # Update action space with actions sampled
                A_cr.update_dict_action_space(X_q, Y_q)

                # Update quantities being tracked within a batch
                q_mini_batch += q*nshots_temp/self.N_batch  # Normalized with respect to the number of shots against it
                nshots_mini_batches += nshots_temp
                num_mini_batches += 1

            # Update number of queries made so far
            N_p += self.N_batch

            # Update stuff being tracked
            J_hat_nd = self.bayesian_estimator.est_mean()
            J_num.append(self.xi_J * J_hat_nd)
            J_nd_num.append(J_hat_nd)
            N_p_vec.append(N_p)
            q_vec.append(q_mini_batch)  # Note the normalization above
            num_mini_batches_vec.append(num_mini_batches)

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
                self.logger(log_file, k+1, mse_temp, loss_num[-1], J_num[-1], num_mini_batches_vec[-1])

            if verbose:
                print('Done with %d' % k)

        # Results so far
        results = {'loss': loss_num, 'mse': mse_vec, 'J_hat': J_num, 'J_truth_nd': self.J_truth_nd, 'xi_J': self.xi_J,
                   'N_p': N_p_vec, 'q': q_vec, 'n_shots': n_shots_vec, 'data': X_p, 'samples': Y_p, 'A_cr': A_cr,
                   'num_mini_batches': num_mini_batches_vec}

        if self.FLAG_testing and self.FLAG_simulator is False:
            # Get RMSE with respect to a testing dataset as well
            # Create testing dataset
            X_test, Y_test = quantum_device_oracle.get_testing_dataset(self.dataset_oracle, A_cr)

            # We don't update the A_cr (which corresponds to the training data) with X_test or Y_test

            # Get "estimate" of J from the testing dataset
            J_test_nd = quick_bayesian_estimate(X_test, Y_test, self.sys_model)

            mse_test = np.zeros(self.max_iter + 1)

            # Compute the testing RMSE
            for ind in range(self.max_iter + 1):
                J_hat_nd = J_num[ind]/self.xi_J
                mse_test[ind] = np.linalg.norm(J_test_nd - J_hat_nd, 2) ** 2

            if log_file is not None:
                self.logger_experimental_data(log_file, mse_vec, loss_num, J_num, mse_test, num_mini_batches_vec)

            # Update results -- not including testing dataset for the time-being
            results.update({'mse_test': mse_test})
            results.update({'J_test_nd': J_test_nd})

        return results
