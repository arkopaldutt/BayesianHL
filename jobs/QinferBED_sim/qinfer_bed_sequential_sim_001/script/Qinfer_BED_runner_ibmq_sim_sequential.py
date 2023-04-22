"""
Function for PL experiments on dataset with SMC solver
"""
# imports
import os, sys, shutil
import pathlib
import argparse
import numpy as np
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")

# For local package imports
PROJECT_PATH = str(pathlib.Path().resolve().parent)
sys.path.append(PROJECT_PATH)

from bayesianhl.action_space import ActionSpace
from bayesianhl.qinfer_bayesian_learner_sequential import QinferExperimentRunner
from bayesianhl.utils import job_helper
from bayesianhl import quantum_device_models
from bayesianhl.quantum_device_oracle import DeviceOracle


# 1. Setup a dataset oracle based on the experimental data from a quantum device
DATA_dir = '../data/'

# Device (ibmq_boeblingen) being considered
# Load the data from which we will consider the parameters and time-stamps
ibm_data = job_helper.setup_oracle(DATA_DIR=DATA_dir)

# Set true values
J_truth = np.array([[-4568258.88132824, -1465045.02848701,  -290468.97835928,
                    6499545.9801579,  1390900.08768167,   413722.74870734]])

# Setup dataset oracle
# Oracle properties
FLAG_simulator = True

# Noise Models
FLAG_readout_noise = True
FLAG_control_noise = True

# Noise
param_truth = quantum_device_models.transform_parameters(J_truth[0])

# Control Noise
teff = quantum_device_models.data_driven_teff_noise_model(param_truth, FLAG_ibmq_boel=True)
expt_noise ={'readout': ibm_data['misclassif_error'], 'imperfect_pulse_shaping': teff}

# Create oracle
oracle_qs_expt = DeviceOracle(J_truth[0], noise=expt_noise, expt_data=ibm_data,
                              FLAG_simulator=FLAG_simulator,
                              FLAG_readout_noise=FLAG_readout_noise,
                              FLAG_control_noise=FLAG_control_noise)

# 2. Setup query space and ActionSpace
n_J = 6
xi_J = 1e6*np.ones((n_J,))
xi_t = 1e-7

# This is what we will actually run with -- non-dimensional parameters
J_truth_nd = J_truth/xi_J

print('Using J_num which we already discovered before!')

# Define query space
moset = [0,1,2]
prepset = [0,1]
time_stamps = ibm_data['time_stamps'][0:81]
tset = time_stamps/xi_t

query_space = {'moset': moset, 'prepset': prepset, 'time_stamps': time_stamps}

A_cr = ActionSpace(moset, prepset, tset, xi_t, xi_J)

# 3. Define conditions for experiment runner
max_shots_query = 512
max_iter = 250
N_0 = 5*A_cr.N_actions
N_batch = A_cr.N_actions
n_particles = 10000
FLAG_testing = True

# 4. For saving and logging info
# Creation of save directory
expt_number = 1
FLAG_job_restart = False
policy_type = 'qinfer_bed_sequential'

if FLAG_simulator:
    SAVE_DIR = policy_type + '_sim_%03d' % expt_number
else:
    SAVE_DIR = policy_type + '_expt_%03d' % expt_number

# Create log-file (summary of all runs) and denote entries
log_filename = SAVE_DIR + '/log_job_%d.txt' % expt_number

if not os.access(SAVE_DIR, os.F_OK):
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not FLAG_job_restart:
        f_log = open(log_filename, "a+")
        f_log.write("Run Iter RMSE Test_Error\n")
        f_log.close()

# Save script to folder to know what we ran
SAVE_DIR_script = SAVE_DIR + '/script'

if not os.access(SAVE_DIR_script, os.F_OK):
    os.makedirs(SAVE_DIR_script, exist_ok=True)

current_script_file = os.path.basename(__file__)
shutil.copy(current_script_file, SAVE_DIR_script)

# log file for a particular run
run_log_filename = SAVE_DIR + '/run_log_job_%d.txt' % expt_number

# For the runs
parser = argparse.ArgumentParser(description='PL Run')
parser.add_argument('--run_seed', type=int, default=0, metavar='N')
args = parser.parse_args()

# Define the experiment
qinfer_expt_run = QinferExperimentRunner(J_truth_nd, query_space, xi_J, xi_t,
                                         dataset_oracle=oracle_qs_expt, FLAG_simulator=FLAG_simulator,
                                         FLAG_testing=FLAG_testing,
                                         FLAG_readout_noise=FLAG_readout_noise,
                                         FLAG_control_noise=FLAG_control_noise,
                                         N_0=N_0, N_batch=N_batch, max_iter=max_iter,
                                         max_shots_query=max_shots_query, n_particles=n_particles)

print("Going to carry out run %d with %d iterations" %(args.run_seed, max_iter))

start_time = time.perf_counter()

if __name__ == '__main__':
    i_run = args.run_seed
    np.random.seed(10 * (i_run + 2))
    print('Run %d with Passive Learning \n Policy Type: %s, Oracle: Simulator (%s)'
          % (i_run, policy_type, FLAG_simulator))

    log_file_run = SAVE_DIR + '/log_Run_%d.txt' % i_run
    results_qinfer = qinfer_expt_run.AL_runner(log_file=log_file_run)

    # Update RMSE and log results
    f_log = open(log_filename, "a+")

    loss_hat = results_qinfer['loss']
    mse = results_qinfer['mse']

    for i_iter in range(max_iter + 1):
        f_log.write("%d %d %f %f \n" % (i_run, i_iter, np.sqrt(mse[i_iter]), loss_hat[i_iter]))

    f_log.close()

    # pickle_result_file = SAVE_DIR + '/Run_%d.pickle' % i_run
    #
    # with open(pickle_result_file, 'wb') as handle:
    #     pickle.dump(results_FI, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # return results_FI

finish_time = time.perf_counter()

print(f'Finished in {round(finish_time - start_time, 2)} second(s)')
