"""
Includes functions to help running job scripts
"""
import numpy as np
import pickle
import scipy.linalg

# package imports
from .. import process_data


def setup_oracle(ind_amp=1, DATA_DIR='../data/', FLAG_classification=True):
    """
    Quick helper function to get an experimental dataset for setting up an oracle
    """
    # Parameters of the different jobs
    meas_level_expt = 1
    n_shots = 512
    n_job = 1
    cr_amp_array = [0.24, 0.30, 0.36, 0.42, 0.48]

    # Load data
    pickle_result_filename = 'ibmq_boel_fixed_qs_data_aligned_A_0_%d_meas_%d_shots_%d_job_%d.pickle' % (
        100 * cr_amp_array[ind_amp], meas_level_expt, n_shots, n_job)

    pickle_result_file = DATA_DIR + pickle_result_filename

    # Readout calibration and formatting again
    ibm_data = process_data.make_dataset_ibmq_device(pickle_result_file,
                                                     FLAG_classification=FLAG_classification, do_plot=False)

    return ibm_data
