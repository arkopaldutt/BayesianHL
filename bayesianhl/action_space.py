"""
Defines the action space for CR Hamiltonians and Bayesian estimators -- the set of experiments that can be taken
"""
import numpy as np
import scipy.linalg
import scipy.stats
import scipy.optimize
import collections
import random


class ActionSpace(object):
    """
    Define the space of all possible actions/configurations (queries)

    For the two cross-resonance coupled qubits.
    """

    def __init__(self, moset, prepset, tset, xi_t, xi_J, n_shots=1e8, xi_param=None, freq_convert=None):
        """
        Function to initialize the pool (set of all possible actions/queries)

        moset = Set of measurement operators (dictionary)
        prepset = Set of preparation operators (dictionary)
        tset = Set of time-stamps that can be queried (non-dimensional)
        xi_t = scalings for time
        xi_param = scalings for parameter set (Should include xi_J later)

        TO DO: Handling both xi_param and xi_J

        action_space is array of [m,u,t] where
        m and u correspond to keys in moset and prepset
        t are the non-dimensional times from tset
        """
        self.moset = moset
        self.prepset = prepset
        self.tset = tset
        self.xi_t = xi_t
        self.xi_J = xi_J
        self.xi_param = xi_param

        # Time-stamps and frequency information of ActionSpace
        time_stamps = tset * xi_t
        _dt = (time_stamps[-1] - time_stamps[0]) / (len(time_stamps) - 1)
        _sample_freq = 1.0 / _dt

        if freq_convert is None:
            freq_convert = _sample_freq * 2.0 * np.pi / len(time_stamps)

        action_space = []
        action_n_shots = []     # Number of shots available for each action

        ## HACK! Replace below with pandas objects later
        ## HACK! Note the rounding required!!! hardcoded at the moment ... Lines 265 and 310,312
        dict_action_space = {}
        dict_action_to_index = {}
        dict_time_to_index = {}
        dict_index_to_time = {}

        # Should probably vectorize this -- And replace the dict_action_space thingy

        ## TO DO: The mappings between indices and time-stamps only works for the fixed action query space
        #t_nd0 = tset[0]
        #dt_nd = np.abs(tset[1]-tset[0])
        # Note that dt in tset is not constant and hence ind_t = np.floor((t-t0)/dt).astype(int) wouldn't be reliable!

        count_time = 0
        ind_action = 0
        for t_nd in tset:
            ind_t = count_time
            dict_index_to_time.update({ind_t: t_nd})

            # str is important. Number as it is was confusing line 315
            dict_time_to_index.update({str(np.round(t_nd, 3)): ind_t})

            # Note that moset and prepset are just arrays unlike Action_Space object of HAL-FI
            for m in moset:
                for u in prepset:
                    action_temp = (m, u, t_nd)
                    action_space.append(action_temp)
                    action_n_shots.append(n_shots)

                    # Dictionary: key is tuple of (m, u, ind_t) and value is [nsamples, n0]
                    dict_action_space.update({(m, u, ind_t): [0, 0]})

                    # So we can move from actions to indices and vice-versa
                    dict_action_to_index.update({action_temp: ind_action})
                    ind_action += 1

            count_time += 1

        self.action_space = action_space

        # [nsamples, number of 0s]
        self.dict_action_space = dict_action_space

        self.action_n_shots = np.asarray(action_n_shots, dtype=int)
        self.max_n_shots = n_shots

        self.dict_action_to_index = dict_action_to_index
        self.dict_time_to_index = dict_time_to_index
        self.dict_index_to_time = dict_index_to_time

        # Number of actions
        self.N_actions_M = len(moset)    # Just M
        self.N_actions_U = len(prepset)  # Just U
        self.N_actions_t = len(tset)    # Just time
        self.N_actions = len(action_space)  # Total = _M*_U*_t

        # Property of sampling from action space - freq_convert (lines 553-556 of process_data.py)
        self.freq_convert = freq_convert
        self.sample_freq = _sample_freq

    def actions_pruning(self, samples_actions):
        """
        Prunes the action set that is to be queried and makes sure that can indeed satisfy query constraints
        (e.g.. nshots available)
        """
        # get collection (julia functionality) of samples_actions -- inefficient (may need to be replaced by pandas)
        # ref: https://stackoverflow.com/questions/23240969/python-count-repeated-elements-in-the-list/23240989
        dict_actions_q = dict(collections.Counter(samples_actions))

        # convert above dictionary to shots made for each query
        # ref: https://stackoverflow.com/questions/23668509/dictionary-keys-and-values-to-separate-numpy-arrays
        ind_actions_q = np.fromiter(dict_actions_q.keys(), dtype=int)
        nshots_actions_q = np.fromiter(dict_actions_q.values(), dtype=int)

        # get queries for which number of shots requested exceeded that allowed and set
        nshots_left_over = self.action_n_shots[ind_actions_q] - nshots_actions_q
        ind_shots_exceeded = np.where(nshots_left_over < 0)[0]

        # set actions_q according to what can be done
        nshots_actions_q[ind_shots_exceeded] += nshots_left_over[ind_shots_exceeded]

        # nshots left over that need to have associated actions
        nshots_not_set = -1*np.sum(nshots_left_over[ind_shots_exceeded])

        # update nshots left over now
        nshots_left_over = self.action_n_shots[ind_actions_q] - nshots_actions_q

        # start setting up an array containing the actions to be made
        samples_actions_pruned = []
        for ind in range(len(nshots_actions_q)):
            ind_query = ind_actions_q[ind]
            n_query_i = nshots_actions_q[ind]
            samples_actions_pruned.extend([ind_query] * n_query_i)

        # sample shots not set uniformly from left over sequentially
        nshots_actions_temp = np.copy(self.action_n_shots)
        nshots_actions_temp[ind_actions_q] = nshots_left_over

        for ind_shot in range(nshots_not_set):
            # get non-empty actions (indices of them)
            valid_action_set_temp = self.nonempty_actions(nshots_actions_temp=nshots_actions_temp)

            # choose one uniformly from above
            ind_query = random.choice(valid_action_set_temp)
            samples_actions_pruned.extend([ind_query])
            nshots_actions_temp[ind_query] -= 1

        return samples_actions_pruned

    def sample_action_space(self, p_query, N_batch, actions_list=None, FLAG_query=True):
        """
        p_query = pdf distribution of the query
        N_batch = number of batch of queries to issue

        FLAG_query to indicate if probability distribution being used or number of samples.
        Var of interest here is p_query

        Usage:
            FLAG_query is true and p_query[i] = 0.5 then ith query is sampled with prob 0.5
            FLAG_query is false and p_query[i] = 5 then ith query is sampled 5 times

        returns actions = {action description: number of shots against action}
        DOES NOT RETURN corresponding samples
        """
        # Sample from the discrete space of actions
        if actions_list is None:
            actions_list = np.arange(self.N_actions)
            N_valid_actions = self.N_actions
        else:
            N_valid_actions = len(actions_list)

        if FLAG_query:
            p_actions = scipy.stats.rv_discrete(name="p_actions", values=(actions_list, p_query))

            samples_actions_query = p_actions.rvs(size=N_batch)
            samples_actions = self.actions_pruning(samples_actions_query)
        else:
            if N_batch % N_valid_actions != 0:
                print('Working in non-query mode. N_batch should be a multiple of N_actions')
                return None

            samples_actions = []
            p_query = np.asarray(p_query, dtype=int)
            for ind_query in range(N_valid_actions):
                # number of samples associated with query_i
                n_query_i = p_query[ind_query]

                # Ref: https://stackoverflow.com/questions/4654414/python-append-item-to-list-n-times
                samples_actions.extend([actions_list[ind_query]] * n_query_i)

        # Create dictionary of queries {action/query: n_shots}
        X_q = {}

        for i in range(N_batch):
            ind_action_i = samples_actions[i]
            m_i, u_i, t_i = self.action_space[ind_action_i]

            action_i = (m_i, u_i, t_i)
            if action_i in X_q.keys():
                X_q[action_i] += 1
            else:
                X_q.update({action_i: 1})

            # Update shots left for that query
            self.action_n_shots[ind_action_i] -= 1

        return X_q

    def update_shots(self, X_q, N_batch):
        """
        This is to update the number of shots left over for valid queries if "sample_action_space" method is not used
        e.g., if we use bayes_risk_heuristic.get_batched_queries

        Inputs:
        :param X_q: dataset containing all queries
        :param N_batch:
        :return:
        """
        assert sum(X_q.values()) == N_batch

        list_queries = list(X_q.keys())
        for i in range(len(list_queries)):
            query_i = list_queries[i]

            # Update shots left for that query
            ind_action_i = self.dict_action_to_index[query_i]
            self.action_n_shots[ind_action_i] -= X_q[query_i]

            if self.action_n_shots[ind_action_i] < 0:
                raise RuntimeError("Negative shots left for current query index %d" % ind_action_i)

    def sample_dataset(self, quantum_device_oracle, query, nshots):
        """
        Sampling from quantum_device_oracle
        Only doing one query at a time at the moment because that is all is required
        """
        # Find location of query in action_space
        ind_action = self.dict_action_to_index[query]

        # Setup only for DeviceOracles with FLAG_classification=True at the moment!
        samples_query = quantum_device_oracle.sample_expt_data(ind_action, nsamples=nshots)

        number_ones = int(np.sum(samples_query))

        return number_ones

    @staticmethod
    def merge_queries(X_p, X_q):
        """
        :param X_p: queries from the pool so far
        :param X_q: most recent queries
        :return: merged datasets
        """
        for query_temp in list(X_q.keys()):
            if query_temp in X_p.keys():
                X_p[query_temp] += X_q[query_temp]
            else:
                X_p.update({query_temp: X_q[query_temp]})

        return X_p

    def update_dict_action_space(self, X_queries, Y_samples):
        """
        Inputs
        dict X_queries: {query: n_shots}
        dict Y_queries: {query: [number of 0s, number of 1s]
        """
        assert X_queries.keys() == Y_samples.keys()

        for query in Y_samples:
            assert np.sum(Y_samples[query]) == X_queries[query]

            # Update dict_action_space
            # keys are of form (m,u,ind_t) so need to determine ind_t
            m, u, t = query
            ind_t = self.dict_time_to_index[str(np.round(t, 3))]

            # [nsamples, n0]
            self.dict_action_space[(m, u, ind_t)] += np.array([X_queries[query], Y_samples[query][0]])

    def nonempty_actions(self, nshots_actions_temp=None, nshots_threshold=1):
        """
        Function that returns indices of queries/actions that are non-empty and/or (> nshots_threshold)
        and corresponding action-space
        """
        if nshots_actions_temp is None:
            nshots_actions_temp = self.action_n_shots

        valid_action_set = np.where(nshots_actions_temp > nshots_threshold)[0]

        return valid_action_set

    def filtered_actions(self, FLAG_uncertainty_filtering=False):
        """
        At the moment, filter actions to make sure constraints that are satisfied
        ahead of time
        """
        valid_action_set_temp = self.nonempty_actions()

        filtered_actions_list = []

        for ind in range(len(valid_action_set_temp)):
            filtered_actions_list.append(self.action_space[valid_action_set_temp[ind]])

        return filtered_actions_list
