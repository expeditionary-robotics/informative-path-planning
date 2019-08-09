# ~/usr/bin/python

'''
This library is used to define custom kernels for PLUMES
License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

from .kern import Kern
from .periodic import PeriodicExponential
from .standard_periodic import StdPeriodic
from .ODE_st import ODE_st
from .rbf import RBF
from ...core.parameterization import Param
import numpy as np
import copy

class Swell(Kern):
    def __init__(self, input_dim, lengthscale_x=1., lengthscale_t=1., variance_x=1., variance_t=1., active_dims=None, name='swell'):
        super(Swell, self).__init__(input_dim, active_dims, name)
        assert input_dim == 3, "Swell Requires 3 Dimensions"
        self.variance_x = variance_x
        self.variance_t = variance_t
        self.lengthscale_x = lengthscale_x
        self.lengthscale_t = lengthscale_t
        self.spatial_kern = RBF(input_dim=2,
                                variance=self.variance_x,
                                lengthscale=self.lengthscale_x,
                                ARD=True)
        self.temporal_kern = PeriodicExponential(input_dim=1,
                                 variance=self.variance_t,
                                 lengthscale=self.lengthscale_t)

    def _get_params(self):
        return np.hstack((self.spatial_kern._get_params(), self.temporal_kern._get_params()))

    def _set_params(self, x):
        limit = self.spatial_kern.num_params
        self.spatial_kern._set_params(x[:limit])
        self.temporal_kern._set_params(x[limit:])

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        spatial_x = X[:, :-1]
        temporal_x = X[:, -1]
        spatial_x2 = X2[:, :-1]
        temporal_x2 = X2[:, -1]

        temporal_x = np.reshape(temporal_x, (1, temporal_x.shape[0]))
        temporal_x2 = np.reshape(temporal_x2, (1, temporal_x2.shape[0]))

        K_spatial = self.spatial_kern.K(spatial_x, spatial_x2)
        K_temporal = self.temporal_kern.K(temporal_x, temporal_x2)

        return K_spatial + K_temporal
        

    def Kdiag(self, X):
        return self.spatial_kern.Kdiag(X[:, :-1]) + self.temporal_kern.Kdiag(np.reshape(X[:,-1], (1, X[:,-1].shape[0])))

    def update_gradients_full(self, dl_dK, X, X2=None):
        if X2 is None:
            X2 = X
        
        temporal_x = np.reshape(X[:,-1], (1, X[:,-1].shape[0]))
        temporal_x2 = np.reshape(X2[:,-1], (1, X2[:,-1].shape[0]))
        self.spatial_kern.update_gradients_full(dl_dK, X[:, :-1], X2[:, :-1])
        self.temporal_kern.update_gradients_full(dl_dK, temporal_x, temporal_x2)

class Transport(Kern):
    def __init__(self, input_dim, lengthscale=1., variance=1., active_dims=None, name='transport'):
        super(Transport, self).__init__(input_dim, active_dims, name)
        self.variance = variance

    def _get_params(self):
        pass

    def _set_params(self, x):
        pass

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        temp = dynamic_func(X[:,0], X[:,1], X[:,-1])
        temp2 = dynamic_func(X2[:,0], X2[:,1], X2[:,-1])
        return self.variance * np.dot(np.square(temp), np.square(temp2.T))

    def Kdiag(self, X):
        return np.diag(self.K(X))

    def update_gradients_full(self, dl_dK, X, X2=None):
        pass

def dynamic_func(x1, x2, t):
    ang1 = 3.50*np.sin(t/12.)
    ang2 = 3.50*np.cos(t/12.)
    temp1 = np.exp(-(np.power(((x1-5.-ang1)/0.7),2)))
    temp2 = np.exp(-(np.power(((x2-5.-ang2)/0.7),2)))
    f = np.multiply(temp1,temp2)
    f = np.array(list(f))[:,None]
    return f
