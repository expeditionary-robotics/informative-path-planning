# Informative Path Planning: Exploring Performance of Robots for Science Missions

In this repository there are a number of Jupyter notebooks which walk through the development of a simulated environment that a simulated robot can explore-exploit using classical informative path planning (IPP) frameworks.

In **ipp.ipynb** you can find the majority of our fundamental implementation laid out in a single notebook. For the latest working simulations, direct your attention to **nonmyopic_experiments.py** and **demo.ipynb**. 

The **real_data_sim.ipynb** and **demo_processing.ipynb** implementations require datasets which are not provided in this repository. To inquire about access to a dataset, please contact the repository contributors. In the **scripts** folder you can find miscellaneous helper functions and prototypes of fully-implemented features.


## Relevant Literature

Please note that the basis of most of the fundamental implementation is derived from the following papers:
* [No-Regret Replanning under Uncertainty - Sun et al. 2016](https://arxiv.org/pdf/1609.05162.pdf)
* [Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design - rinivas et al. 2010](https://arxiv.org/pdf/1609.05162.pdf)
* [Multi-Modal Active Perception for Information Gathering in Science Missions - Arora et al. 2017](https://arxiv.org/pdf/1712.09716.pdf)


## Necessary Imports and other Logistics

To run these notebooks, the following packages are necessary:

* numpy
* scipy
* GPy (a Gaussian Process library)
* dubins (a Dubins curve generation library)
* matplotlib
* Ipython

The code is written in Python 2.7+ syntax, and minor adjustments may need to be made to be Python 3 compatible. 