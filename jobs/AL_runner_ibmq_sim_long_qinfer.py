"""
Function for PL experiments on simulator with baseline solver considering Nyquist criterion is satisfied
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
from bayesianhl.active_bayesian_learner import ActiveExperimentRunner
from bayesianhl import design_experiment


# 1. Setup system model and truth
n_J = 6
xi_J = 1e6*np.ones((n_J,))
xi_t = 1e-7

# Set true values
J_truth = np.array([[-4568258.88132824, -1465045.02848701,  -290468.97835928,
                    6499545.9801579,  1390900.08768167,   413722.74870734]])

# This is what we will actually run with -- non-dimensional parameters
J_truth_nd = J_truth/xi_J

print('Using J_num which we already discovered before!')

# 2. Define query space
moset = [0,1,2]
prepset = [0,1]
time_stamps = np.linspace(1.0e-7, 18e-7, 3*81)
tset = time_stamps/xi_t

query_space = {'moset': moset, 'prepset': prepset, 'time_stamps': time_stamps}

A_cr = ActionSpace(moset, prepset, tset, xi_t, xi_J)

# 2b. Define Active Learner
policy_type = 'FI'
FLAG_normalization = True
FLAG_noise = True
growth_time_stamps = None
FLAG_constraints = True

query_constraints_ref = {'N_shots': 512}
FLAG_lower_limits = False

HAL_FI_J = design_experiment.ActiveLearner(policy_type=policy_type,
                                           FLAG_normalization=FLAG_normalization,
                                           FLAG_noise=FLAG_noise,
                                           FLAG_constraints=FLAG_constraints,
                                           query_constraints=query_constraints_ref,
                                           type_param='J')

FLAG_query_constraints = True
query_constraints_info = {'query_constraints_ref': None, 'query_optimization_type': 'batch'}

# 3. Define conditions for experiment runner
max_iter = 70
N_0 = 5*A_cr.N_actions
N_batch = A_cr.N_actions
n_particles = 20000

# 4. For saving and logging info
# Creation of save directory
expt_number = 2
FLAG_simulator = True
FLAG_control_noise = True

FLAG_job_restart = False

if FLAG_simulator:
    SAVE_DIR = policy_type + '_sim_long_qinfer_%03d' % expt_number
else:
    SAVE_DIR = policy_type + '_expt_long_qinfer_%03d' % expt_number

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
AL_expt_run = ActiveExperimentRunner(J_truth_nd, query_space, xi_J, xi_t, HAL_FI_J,
                                     FLAG_query_constraints=FLAG_query_constraints,
                                     query_constraints_info=query_constraints_info,
                                     FLAG_control_noise=FLAG_control_noise,
                                     N_0=N_0, N_batch=N_batch, max_iter=max_iter, n_particles=n_particles)

print("Going to carry out run %d with %d iterations" %(args.run_seed, max_iter))

start_time = time.perf_counter()

if __name__ == '__main__':
    i_run = args.run_seed
    np.random.seed(10 * (i_run + 2))
    print('Run %d with Passive Learning \n Policy Type: %s, Oracle: Simulator (%s)'
          % (i_run, policy_type, FLAG_simulator))

    log_file_run = SAVE_DIR + '/log_Run_%d.txt' % i_run
    results_PL = AL_expt_run.AL_runner(log_file=log_file_run)

    # Update RMSE and log results
    f_log = open(log_filename, "a+")

    loss_hat = results_PL['loss']
    mse = results_PL['mse']

    for i_iter in range(max_iter + 1):
        f_log.write("%d %d %f %f \n" % (i_run, i_iter, np.sqrt(mse[i_iter]), loss_hat[i_iter]))

    f_log.close()

finish_time = time.perf_counter()

print(f'Finished in {round(finish_time - start_time, 2)} second(s)')