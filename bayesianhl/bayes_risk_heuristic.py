"""
Defines the Bayes risk heuristc for our experiments with Noisy CRHamiltonians
"""
import numpy as np
import qinfer as qi


class BayesRiskHeuristic(object):
    def __init__(self, updater, sys_model, Q, experiment_space, SOME_PRIOR, name=None, subsample_particles=None):
        # Had issues with qi.Heuristic class with carrying out multiple measurements so decided to just do this by hand
        # Ref: RiskHeuristic class from https://github.com/ihincks/nv-adaptive/blob/master/src/adaptive.py
        # System model -- Note that this is usually the parent in our case
        self._ham_model = sys_model
        self._ham_model._Q = Q

        self.n_particles = updater.n_particles if subsample_particles is None else subsample_particles
        self._risk_taker = qi.SMCUpdater(self._ham_model, self.n_particles, SOME_PRIOR)
        self._update_risk_particles(updater)
        self._eps = experiment_space

        self.name = "Bayes Risk, Q={}".format(Q) if name is None else name
        self.risk_history = []

    def _update_risk_particles(self, updater):
        n_mps = self._risk_taker.model.base_model.n_modelparams
        if self.n_particles == updater.n_particles:
            locs = updater.particle_locations[:, :n_mps]
            weights = updater.particle_weights
        else:
            locs = updater.sample(n=self.n_particles)[:, :n_mps]
            weights = np.ones(self.n_particles) / self.n_particles

        self._risk_taker.particle_locations = locs
        self._risk_taker.particle_weights = weights

    def get_best_query(self, updater, valid_experiments=None):
        if valid_experiments is None:
            all_eps = self._eps
        else:
            all_eps = valid_experiments

        self._update_risk_particles(updater)

        risk = []
        for ind_ep in range(len(all_eps)):
            m, u, t_nd = all_eps[ind_ep]
            ep_temp = np.array([(m, u, t_nd)], dtype=self._ham_model.expparams_dtype)
            risk.append(self._risk_taker.bayes_risk(ep_temp))

        self.risk_history += [risk]
        best_idx = np.argmin(risk, axis=0)
        eps = np.array([all_eps[best_idx]])

        q_vec = np.zeros(len(all_eps))
        q_vec[best_idx] = 1.0

        return eps, q_vec, risk

    def get_batched_queries(self, N_batch, updater, valid_queries_indices, n_actions_left):
        """
        Assuming that the query space is not changing

        Inputs:
            updater: BayesianEstimator being used by the learner
            valid_queries_indices: indices in experiment_space that still have actions left as given by n_actions_left
            n_actions_left: Number of actions/shots left in dataset/budget for valid_queries

        Returns: set of queries that can be made, q_vec (normalized), risk of this batch of queries
        """
        n_valid_queries = len(valid_queries_indices)

        # Update the risk particles
        self._update_risk_particles(updater)

        # Compute Bayes risk for each query
        risk_queries = np.ones(n_valid_queries)
        for ind_query in range(n_valid_queries):
            valid_query_idx = valid_queries_indices[ind_query]

            m, u, t_nd = self._eps[valid_query_idx]
            ep_temp = np.array([(m, u, t_nd)], dtype=self._ham_model.expparams_dtype)

            risk_queries[ind_query] = self._risk_taker.bayes_risk(ep_temp)

        # At the end of above, we have in order [valid_query_index, nshots_left, bayes risk]
        # Sort the risk in ascending order and accordingly sort the query indices + n_actions_left
        ind_sort_ascending = np.argsort(risk_queries)
        sorted_queries_indices = valid_queries_indices[ind_sort_ascending]
        sorted_nshots_left = n_actions_left[ind_sort_ascending]

        # Exhaust queries in order of lowest risk to highest risk until we get a set of N_batch
        q_vec = np.zeros(len(self._eps))
        X_queries = {}
        nshots = 0
        ind_q = 0
        while nshots < N_batch and ind_q < n_valid_queries:
            # Get the index of query in the array of valid queries (sorted by ascending order of risk)
            ind_query_temp = sorted_queries_indices[ind_q]
            nshots_query_temp = sorted_nshots_left[ind_q]
            m, u, t_nd = self._eps[ind_query_temp]

            nshots_query = np.amin([nshots_query_temp, N_batch-nshots])
            X_queries.update({(m, u, t_nd): nshots_query})

            # Update distribution
            q_vec[ind_query_temp] = nshots_query

            # Update counters
            ind_q += 1
            nshots += nshots_query

        if nshots < N_batch:
            raise RuntimeError("No queries with shots left! Was only able to get %d/%d queries" %(nshots, N_batch))

        assert np.isclose(sum(X_queries.values()), N_batch)

        # Normalize distribution
        assert np.isclose(np.sum(q_vec), N_batch)
        q_vec = q_vec/N_batch

        return X_queries, q_vec, risk_queries

    def get_minibatch_queries(self, N_minibatch, updater, valid_queries_indices, n_actions_left):
        """
        Assuming that the query space is not changing

        Inputs:
            updater: BayesianEstimator being used by the learner
            valid_queries_indices: indices in experiment_space that still have actions left as given by n_actions_left
            n_actions_left: Number of actions/shots left in dataset/budget for valid_queries

        Returns: set of queries that can be made, q_vec (normalized), risk of this batch of queries
        """
        n_valid_queries = len(valid_queries_indices)

        # Update the risk particles
        self._update_risk_particles(updater)

        # Compute Bayes risk for each query
        risk_queries = np.ones(n_valid_queries)
        for ind_query in range(n_valid_queries):
            valid_query_idx = valid_queries_indices[ind_query]

            m, u, t_nd = self._eps[valid_query_idx]
            ep_temp = np.array([(m, u, t_nd)], dtype=self._ham_model.expparams_dtype)

            risk_queries[ind_query] = self._risk_taker.bayes_risk(ep_temp)

        # Get the best query
        best_idx = np.argmin(risk_queries, axis=0)
        ind_best_query = valid_queries_indices[best_idx]

        m, u, t_nd = self._eps[ind_best_query]

        nshots_best_query = np.amin([N_minibatch, n_actions_left[best_idx]])
        X_q = {(m, u, t_nd): nshots_best_query}

        q_vec = np.zeros(len(self._eps))
        q_vec[best_idx] = 1.0

        return X_q, q_vec, nshots_best_query, risk_queries

