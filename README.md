# Informative Path Planning: Exploring Performance of Robots for Science Missions

Adaptive sampling is a crtical autonomous behavior for robotic systems in a number of contexts -- from selecting actions to perform to refine a gripping/manipulation controller, to selecting the next best place to observe a sample in an uknown environment. The specific focus of this respository is to make a simple adaptive sampling playground for simualted point-robots in smooth, Gaussian environments -- environments that can be commonly found in nature (rolling hills, distribution of flora, fluid dispersion, etc). We frame the problem as a POMDP, and solve it using a Monte Carlo Tree Search (MCTS) instance. A number of parameters can be tuned by a user of this repository, including:
* planning horizon (supports myopic planning)
* type of MCTS (vanilla, or double progressive-widening are allowed)
* selection of obstacles
* computation and other budgets for mission execution
* path sets (we support point-to-point navigation, or specified pathsets, all parameerized as dubins curves)
* reward function for quantifying the value of an observation (UCB, max-value entropy, and expected improvement are supported)

Further, we've built out a number of functions which allow for close analysis of a mission, including:
* trajectory plotting
* record of the belief state
* tracking of accumulated information gain, optimal selections, mean squared error, and more

A quickstart guide is provided in the **demo.ipynb** notbook; and detailed comments are provided within the scripts themselves.


## Relevant Literature

Please note that the basis of most of the fundamental implementation is derived from the following papers:
* [No-Regret Replanning under Uncertainty - Sun et al. 2016](https://arxiv.org/pdf/1609.05162.pdf)
* [Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design - rinivas et al. 2010](https://arxiv.org/pdf/1609.05162.pdf)
* [Multi-Modal Active Perception for Information Gathering in Science Missions - Arora et al. 2017](https://arxiv.org/pdf/1712.09716.pdf)
* [Max-value Entropy Search for Efficient Bayesian Optimization - Wang and Jegelka 2017](https://arxiv.org/pdf/1703.01968.pdf)


## Necessary Imports and other Logistics

To run the demo notebook and to use some of the libraries, the following packages are necessary:

* numpy
* scipy
* GPy (a Gaussian Process library)
* dubins (a Dubins curve generation library)
* matplotlib
* Ipython

The code is written in Python 2.7+ syntax, and minor adjustments may need to be made to be Python 3 compatible. 