"""
Defines the oracles based on simulator or experimental data set
"""
import numpy as np
import scipy.linalg
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import collections
import random

from . import process_data, action_space

# pauli matrices
si = np.array([ [1, 0], [0, 1] ])
sx = np.array([ [0, 1], [1, 0] ])
sy = np.array([ [0, -1j], [1j, 0] ])
sz = np.array([ [1, 0], [0, -1] ])

# hadamard
hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)


class DeviceOracle(object):
    """
    Represent Nature's knowledge of the (hidden) model Hamiltonian, providing
    a method to obtain sample measurement observable results.

    For the two cross-resonance coupled qubits.
    """

    def __init__(self, J, noise=None, expt_data=None, quantum_device='ibmq_boeblingen',
                 FLAG_simulator=False, FLAG_queryable_expt_data=False,
                 FLAG_readout_noise=False, FLAG_control_noise=False, FLAG_decoherence=False):
        """
        Inputs:
            Jix, Jiy, Jiz, Jzx, Jzy, Jzz = qubit coupling strengths (known to Nature, not to Physicist)

            noise is a dictionary with the following keys:
            readout: (r0,r1)
            imperfect_pulse_shaping: (teff0, teff1)
            decoherence: two-qubit decoherence model (assumed always!)

            Allowing nature to be a dataset itself
            FLAG_simulator = True for simulator and False for experimental dataset
            FLAG_queryable_data = True if expt data already in desired format for it to be queryable

            If False, exp_data better not be empty and must be in the ibm_data format

        TODO:
        1. Generalize depolarization models allowed
        2. Generalize handling of datasets
        """
        # Define Hamiltonian of quantum device oracle
        if len(J) != 6:
            raise ValueError("Expected 6 CR Hamiltonian parameters, got %d" % len(J))

        Jix, Jiy, Jiz, Jzx, Jzy, Jzz = J

        self.J = J

        self.Jix = Jix
        self.Jiy = Jiy
        self.Jiz = Jiz
        self.Jzx = Jzx
        self.Jzy = Jzy
        self.Jzz = Jzz

        self.IX = self.kron(si, sx)
        self.IY = self.kron(si, sy)
        self.IZ = self.kron(si, sz)
        self.ZX = self.kron(sz, sx)
        self.ZY = self.kron(sz, sy)
        self.ZZ = self.kron(sz, sz)

        self.hmat = (Jix * self.IX + Jiy * self.IY + Jiz * self.IZ +
                     Jzx * self.ZX + Jzy * self.ZY + Jzz * self.ZZ)

        self.basis = np.eye(4)  # basis vectors |0>, |1>, |2>, |3>
        self.psi0 = self.basis[:, 0]  # |0>

        # Noise is a part of the environment description too ;)

        # AND Giving the choice to not consider noise during estimation even if present
        self.FLAG_readout_noise = FLAG_readout_noise
        self.FLAG_control_noise = FLAG_control_noise
        self.FLAG_decoherence = FLAG_decoherence

        if noise is not None:
            if self.FLAG_readout_noise is True:
                self.readout_noise = noise['readout']
            else:
                self.readout_noise = np.array([0.0, 0.0])

            if self.FLAG_control_noise is True:
                self.imperfect_pulse_shaping = noise['imperfect_pulse_shaping']
            else:
                self.imperfect_pulse_shaping = None

            if self.FLAG_decoherence is True:
                self.decoherence_model = noise['decoherence_model']
            else:
                self.decoherence_model = None

        self.FLAG_simulator = FLAG_simulator

        # This snippet of code below is different between bayesianhl and the package hamiltonianlearner
        if FLAG_simulator is False:
            print('Experimental data simulator setup')
        else:
            print('Simulated data simulator setup')

        if expt_data is None:
            raise ValueError('You requested an oracle with access to data but you didnt give anything to work with!')
        else:
            self.FLAG_classification = expt_data['FLAG_classification']
            if FLAG_queryable_expt_data:
                self.expt_data = expt_data
                self.device = expt_data['device']
            else:
                if 'device' in expt_data.keys():
                    self.device = expt_data['device']
                    if expt_data['device'] == 'ibmq_boeblingen':
                        self.expt_data = process_data.create_queryable_dataset_ibmq(expt_data,
                                                                                    FLAG_classification=self.FLAG_classification)
                    else:
                        print('Warning: unknown device key. Using default way of creating oracle from dataset')
                        self.expt_data = process_data.create_queryable_dataset(expt_data)
                else:
                    # Do the default
                    self.device = quantum_device
                    self.expt_data = process_data.create_queryable_dataset(expt_data)

    def print_info(self):
        if self.FLAG_simulator:
            print('Oracle: Simulated dataset')
        else:
            print('Oracle: Experimental dataset')

        print('Noise Sources:')
        print('Readout Noise: FLAG=%r, Value=%s' %(self.FLAG_readout_noise, self.readout_noise))
        print('Control Noise: FLAG=%r' % self.FLAG_control_noise)
        print('Decoherence: FLAG=%r' % self.FLAG_decoherence)

    @staticmethod
    def kron(a, b):
        return np.array(scipy.linalg.kron(a, b))

    def sample_expt_data(self, ind_action, nsamples=1):
        """
        1. Pop random measurement outcome corresponding to ind_action
        2. Update the size of that array in the oracle

        :param ind_action:
        :param nsamples:
        :return:
        """
        samples = []

        if self.FLAG_classification is False:
            samples_p0 = []
            samples_p1 = []

        for ind_sample in range(nsamples):
            # Get number of samples remaining for particular action
            nsamples_action_i = self.expt_data['n_samples_actions'][ind_action]

            if nsamples_action_i > 0:
                # Get a random index to sample
                rand_sample = np.random.randint(0, high=nsamples_action_i)

                samples.append(self.expt_data[ind_action].pop(rand_sample))
                self.expt_data['n_samples_actions'][ind_action] -= 1

                if self.FLAG_classification is False:
                    samples_p0.append(self.expt_data['samples_p0'][ind_action].pop(rand_sample))
                    samples_p1.append(self.expt_data['samples_p1'][ind_action].pop(rand_sample))

            else:
                print('Have used up all the samples from this action!')

        if self.FLAG_classification is False:
            return samples, samples_p0, samples_p1
        else:
            return samples


# Quick helper functions to get testing dataset
def get_testing_dataset(env_quantum_sys, A_cr_train, max_shots_query=None):
    """
    Get the testing dataset after creation of training dataset

    This is not a method of ActionSpace to ensure that we don't accidentally change the object of interest

    Inputs:
        A_cr_train: ActionSpace associated with the training dataset
        env_quantum_sys: Oracle
        max_shots_query: Maximum shots we allow for each query in the testing dataset

    Returns:
        X_test: Testing dataset
        A_cr_test: ActionSpace associated with the testing dataset
    TODO: Needs to be generalized for adaptively growing time where experimental datasets are stitched over rounds
    """
    if env_quantum_sys.FLAG_simulator is not False:
        raise RuntimeError('Type of system oracle doesnt allow for this! Set FLAG_simulator!')

    X_test = {}
    Y_test = {}

    # Go through list of remaining actions and create testing dataset
    for ind_query in range(A_cr_train.N_actions):
        n_remaining_actions = A_cr_train.action_n_shots[ind_query]
        if max_shots_query is not None:
            n_remaining_actions = np.amin([n_remaining_actions, max_shots_query])

        if n_remaining_actions > 0:
            query = A_cr_train.action_space[ind_query]
            num_ones = np.sum(env_quantum_sys.sample_expt_data(ind_query, nsamples=n_remaining_actions))

            X_test.update({query: n_remaining_actions})

            # query: [number of 0s, number of 1s]
            Y_test.update({query: [n_remaining_actions - num_ones, num_ones]})

    return X_test, Y_test
