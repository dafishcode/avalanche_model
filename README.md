# spiking_network_criticality
Repo for building recurrent neural networks for modelling criticality. 

## What is this repo for?
* the modelling of cascading avalanche dynamics using recurrent neural networks
* the simulation of network response properties and dynamics in network models
* the simulation of small LIF networks in python


## What does this repo contain?
* Modules contain functions which for network fitting and modeling
* Accompanying ipynotebooks demonstrate how to use the modules


### Modules
'admin_functions.py' - useful administrative functions useful 

'avalanches.py' - module for calculating avalanche dynamics and critical statistics

'IS.py' - module for Bayesian importance sampling approach for loglikelihood ratio testing of power law vs lognormal distribution

'network_mod.py' - module for the generation of network model architectures and simulation of dynamics, and data fitting

'network_stim.py' - module for the running of network perturbations and calculations of netwokr dynamics and response properties

'network.py' - module for the construction of brain networks from empirical data

### Notebooks

'net_fit.ipynb' - notebook for the generation of network model architectures and simulation of dynamics, and data fitting

'network_stim.ipynb' - notebook for the running of network perturbations and calculations of network dynamics and response properties

'LIF_py.ipynb' - notebook for the construction of small networks of LIF neurons in python

'network.ipynb' - notebook for the construction of brain networks from empirical data
