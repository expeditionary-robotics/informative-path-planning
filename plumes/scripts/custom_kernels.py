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
from .linear import Linear
from ...core.parameterization import Param
import numpy as np
import copy

class Swell(Kern):
    def __init__(self, input_dim, lengthscale_x=1., lengthscale_t=1., variance_x=1., variance_t=1., period=100., active_dims=None, name='swell'):
        super(Swell, self).__init__(input_dim, active_dims, name)
        assert input_dim == 3, "Swell Requires 3 Dimensions"
        self.variance_x = variance_x
        self.variance_t = variance_t
        self.lengthscale_x = lengthscale_x
        self.lengthscale_t = lengthscale_t
        self.period = period
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
        return np.diag(self.K(X))

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
    f = 1.5*np.multiply(temp1,temp2)
    f = np.array(list(f))[:,None]
    return f

class Seperable(Kern):
    ''' Note: This is a playground kernel '''
    def __init__(self, input_dim=3, lengthscale_x=1.5, lengthscale_t=1., variance_x=100., variance_t=100., period=360., center=(5., 5.), active_dims=None, name='seperable'):
        super(Seperable, self).__init__(input_dim, active_dims, name)
        assert input_dim == 3, "Seperable Requires 3 Dimensions"
        self.variance_x = variance_x
        self.variance_t = variance_t
        self.lengthscale_x = lengthscale_x
        self.lengthscale_t = lengthscale_t
        self.period = period
        self.center = center
        self.spatial_kern = RBF(input_dim=1,
                                variance=self.variance_x,
                                lengthscale=self.lengthscale_x,
                                ARD=True)
        self.angle_kern = RBF(input_dim=1,
                                variance=self.variance_x,
                                lengthscale=self.lengthscale_x,
                                ARD=True)
        self.temporal_kern = StdPeriodic(input_dim=1,
                                         variance=self.variance_t,
                                         lengthscale=self.lengthscale_t,
                                         period=period)
        # self.temporal_kern = StdPeriodic(input_dim=3,
        #                                  variance=self.variance_t,
        #                                  lengthscale=self.lengthscale_t,
        #                                  period=period)
        # self.spatial_kern = StdPeriodic(input_dim=1,
        #                                 variance=self.variance_t,
        #                                 lengthscale=self.lengthscale_t,
        #                                 period=period)
        # self.temporal_kern = Linear(input_dim=1)
        # self.spatial_kern_cir = RBF(input_dim=2,
        #                         variance=100.,
        #                         lengthscale=1.5,
        #                         ARD=True)
        # self.spatial_kern_x = StdPeriodic(input_dim=1,
        #                                   variance=self.variance_x,
        #                                   lengthscale=self.lengthscale_x,
        #                                   period=period)
        # self.spatial_kern_y = StdPeriodic(input_dim=1,
        #                                   variance=self.variance_x,
        #                                   lengthscale=self.lengthscale_x,
        #                                   period=period)

    def _get_params(self):
        # return np.hstack((self.spatial_kern_x._get_params(), self.spatial_kern_y._get_params()))
        # return np.hstack((self.spatial_kern._get_params(), self.temporal_kern._get_params()))
        return np.hstack((self.spatial_kern._get_params(), self.angle_kern._get_params()))

    def _set_params(self, x):
        # limit = self.spatial_kern_x.num_params
        # self.spatial_kern_x._set_params(x[:limit])
        # self.spatial_kern_y._set_params(x[limit:])
        limit = self.spatial_kern.num_params
        self.spatial_kern._set_params(x[:limit])
        # self.temporal_kern._set_params(x[limit:])
        self.angle_kern._set_params(x[limit:])


    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        spatial_x = np.reshape(X[:,0], (1, X[:,0].shape[0]))
        spatial_x2 = np.reshape(X2[:,0], (1, X2[:,0].shape[0]))
        spatial_y = np.reshape(X[:,1], (1, X[:,1].shape[0]))
        spatial_y2 = np.reshape(X2[:,1], (1, X2[:,1].shape[0]))
        temporal = np.reshape(X[:,2], (1, X[:,2].shape[0]))
        temporal2 = np.reshape(X2[:,2], (1, X2[:,2].shape[0]))
        # return self.spatial_kern_x.K(spatial_x, spatial_x2) + self.spatial_kern_y.K(spatial_y, spatial_y2)

        radius = np.sqrt(np.square(spatial_x - self.center[0]) + np.square(spatial_y - self.center[1]))
        radius2 = np.sqrt(np.square(spatial_x2 - self.center[0]) + np.square(spatial_y2 - self.center[1]))
        angle = np.arctan2(spatial_y, spatial_x)
        angle2 = np.arctan2(spatial_y2, spatial_x2)

        arclength = np.multiply(radius.T,(temporal.T*np.pi/180. - angle.T))
        arclength2 = np.multiply(radius2.T,(temporal2.T*np.pi/180. - angle2.T))

        chord = radius.T*2*np.sin(np.fabs((temporal.T*np.pi/180. - angle.T)/2.))
        chord2 = radius2.T*2*np.sin(np.fabs((temporal2.T*np.pi/180. - angle2.T)/2.))


        # k_ang = self.temporal_kern.K(angle.T-temporal.T, angle2.T-temporal2.T)
        # k_spat = self.spatial_kern.K(radius.T*np.cos(angle.T-temporal.T), radius2.T*np.cos(angle2.T-temporal2.T))
        # k_spaty = self.spatial_kern.K(radius.T*np.sin(angle.T-temporal.T), radius2.T*np.sin(angle2.T-temporal2.T))
        k_chord = self.angle_kern.K(chord, chord2)

        #Let the radius be modeled as an RBF
        # k_spat = self.spatial_kern.K(radius.T, radius2.T)
        # k_spat_x = self.spatial_kern.K(spatial_x.T, spatial_x2.T)
        # k_spat_y = self.spatial_kern.K(spatial_y.T, spatial_y2.T)
        # k_spat_x = self.spatial_kern.K(radius.T*np.cos(temporal.T/12.), radius2.T*np.cos(temporal2.T/12.))
        # k_spat_y = self.spatial_kern.K(radius.T*np.sin(temporal.T/12.), radius2.T*np.sin(temporal2.T/12.))
        #Let the mode be modeled as an RBF
        # k_cir = self.spatial_kern_cir.K(X[:,:-1], X2[:,:-1])

        #Let the angle be modeled as a function of time
        k_temp = self.temporal_kern.K(temporal.T, temporal2.T)
        # k_temp = self.spatial_kern.K(temporal.T, temporal2.T)
        # k_temp = self.spatial_kern.K(np.arctan2(temporal.T), np.arctan2(temporal2.T))
        # xvar = self.spatial_kern.K(spatial_x.T, spatial_x2.T) + k_temp
        # yvar = self.spatial_kern.K(spatial_y.T, spatial_y2.T) + k_temp
        return k_chord

    def Kdiag(self, X):
        return np.diag(self.K(X))

    def update_gradients_full(self, dl_dK, X, X2=None):
        if X2 is None:
            X2 = X

        spatial_x = np.reshape(X[:,0], (1, X[:,0].shape[0]))
        spatial_x2 = np.reshape(X2[:,0], (1, X2[:,0].shape[0]))
        spatial_y = np.reshape(X[:,1], (1, X[:,1].shape[0]))
        spatial_y2 = np.reshape(X2[:,1], (1, X2[:,1].shape[0]))
        temporal = np.reshape(X[:,2], (1, X[:,2].shape[0]))
        temporal2 = np.reshape(X2[:,2], (1, X2[:,2].shape[0]))

        radius = np.sqrt(np.square(spatial_x - self.center[0]) + np.square(spatial_y - self.center[1]))
        radius2 = np.sqrt(np.square(spatial_x2 - self.center[0]) + np.square(spatial_y2 - self.center[1]))
        angle = np.arctan2(spatial_y, spatial_x)
        angle2 = np.arctan2(spatial_y2, spatial_x2)

        arclength = np.multiply(radius.T,(temporal.T - np.arctan2(spatial_y.T, spatial_x.T)))
        arclength2 = np.multiply(radius2.T,(temporal2.T - np.arctan2(spatial_y2.T, spatial_x2.T)))

        chord = 2*np.sin((temporal.T*np.pi/180. - angle.T)/2.)
        chord2 = 2*np.sin((temporal2.T*np.pi/180. - angle2.T)/2.)

        self.spatial_kern.update_gradients_full(dl_dK, radius.T, radius2.T)
        self.angle_kern.update_gradients_full(dl_dK, chord, chord2)

class PolarPeriodic(Kern):
    def __init__(self, input_dim=3, lengthscale_x=1.5, lengthscale_t=1., variance_x=100., variance_t=100., period=360., center=(5., 5.), active_dims=None, name='seperable'):
        super(PolarPeriodic, self).__init__(input_dim, active_dims, name)
        assert input_dim == 3, "Seperable Requires 3 Dimensions"
        self.variance_x = variance_x
        self.variance_t = variance_t
        self.lengthscale_x = lengthscale_x
        self.lengthscale_t = lengthscale_t
        self.period = period
        self.center = center

        self.kern = RBF(input_dim=2,
                        variance=self.variance_x,
                        lengthscale=self.lengthscale_x,
                        ARD=True)
    
    def _get_params(self):
        return self.kern._get_params()

    def _set_params(self, x):
        return self.kern._set_params(x)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        spatial_x = np.reshape(X[:,0], (1, X[:,0].shape[0]))
        spatial_x2 = np.reshape(X2[:,0], (1, X2[:,0].shape[0]))
        spatial_y = np.reshape(X[:,1], (1, X[:,1].shape[0]))
        spatial_y2 = np.reshape(X2[:,1], (1, X2[:,1].shape[0]))
        temporal = np.reshape(X[:,2], (1, X[:,2].shape[0]))
        temporal2 = np.reshape(X2[:,2], (1, X2[:,2].shape[0]))

        radius = np.sqrt(np.square(spatial_x - self.center[0]) + np.square(spatial_y - self.center[1]))
        radius2 = np.sqrt(np.square(spatial_x2 - self.center[0]) + np.square(spatial_y2 - self.center[1]))
        angle = 2*np.pi*temporal/self.period + np.arctan2(spatial_y-self.center[1], spatial_x-self.center[0])
        angle2 = 2*np.pi*temporal2/self.period + np.arctan2(spatial_y2-self.center[1], spatial_x2-self.center[0])

        xprime = np.multiply(radius, np.cos(angle))
        xprime2 = np.multiply(radius2, np.cos(angle2))
        yprime = np.multiply(radius, np.sin(angle))
        yprime2 = np.multiply(radius2, np.sin(angle2))

        return self.kern.K(np.vstack([xprime, yprime]).T, np.vstack([xprime2, yprime2]).T)


    def Kdiag(self, X):
        return self.kern.Kdiag(X)


    def update_gradients_full(self, dl_dK, X, X2=None):
        if X2 is None:
            X2 = X
        self.kern.update_gradients_full(self, dl_dK, X, X2)
