# BayesianHL
(Code on Bayesian Hamiltonian Learning accompanying "Active Learning of Quantum System Hamiltonians yields Query Advantage")

We consider the problem of learning the cross-resonance (CR) Hamiltonian. We compare the batch-mode active learning algorithm for Hamiltonian learning based on Fisher information
([HAL-FI](https://github.com/arkopaldutt/HAL)) against the sequential active learning algorithm defined in [Qinfer](https://arxiv.org/abs/1610.00336). We use the Bayesian estimator of sequential
Monte Carlo (SMC) method as described part of the [Qinfer](https://github.com/QInfer/python-qinfer) package. Please check out the [HAL](https://github.com/arkopaldutt/HAL) repo for using HAL-FI with different estimators such as
maximum likelihood estimation and regression.

## Code design

The structure of the code in `bayesianhl` is as follows
* `quantum_device_models.py`
* `quantum_device_oracle.py`
* `qinfer_system_models.py` defines simpler Hamiltonian learning models on which different methods from this repo can be tested on
* `qinfer_cr_hamiltonians.py` defines the CR Hamiltonian in the presence and absence of different noise sources 
* `action_space.py` 
* `design_experiment.py`
* 
* `active_bayesian_learner.py` 
* `qinfer_bayesian_learner.py` 

Additionally, the `cases` directory includes the following Jupyter notebooks that demonstrate usage of the code:
* `demo_hamiltonian_learning.ipynb` describes how to define different CR-like Hamiltonians and run passive learning experiments on them
* `demo_adaptivity_qinfer.ipynb` works through how queries are obtained as part of the sequential active learner in Qinfer and how to run learning experiments with this method on CR Hamiltonians
* `demo_experiments_cr_hamiltonian.ipynb` describes how to run learning experiments on the CR Hamiltonian on a simulator with HAL-FI combined with the SMC method (a Bayesian estimator)

Finally, if you want to run larger jobs, there are scripts on learning experiments in `jobs`. Post-processing of the data generated through the numerical experiments is illustrated in [HAL](https://github.com/arkopaldutt/HAL). Data obtained in our numerical experiments on the simulator and used in generating results for the paper can also be found here.
Please contact the authors for access to the experimental data sets collected from an IBM Quantum device.

### Requirements

To run this package, we recommend installing the requirements specified in [env_requirements.yml](https://github.com/arkopaldutt/BayesianHL/blob/main/env_requirements.yml) if you
want to run experiments in learning the CR Hamiltonian with the sequential active learner in Qinfer. Use [active_env_requirements.yml](https://github.com/arkopaldutt/BayesianHL/blob/main/active_env_requirements.yml)
if you want to run experiments in learning the CR Hamiltonian with HAL-FI combined with a Bayesian estimator. The second requirements file 
includes additional dependencies of mosek, cvxopt, and cvxpy, that are required by HAL-FI.

### Note
The code accompanying our [paper](https://arxiv.org/abs/2112.14553) is divided into two repos of [BayesianHL]() and [HAL](https://github.com/arkopaldutt/HAL) as
the former has slightly different dependencies to use the methods from [Qinfer](https://github.com/QInfer/python-qinfer).

## Citing this repository

To cite this repository please include a reference to our [paper](https://arxiv.org/abs/2112.14553).

