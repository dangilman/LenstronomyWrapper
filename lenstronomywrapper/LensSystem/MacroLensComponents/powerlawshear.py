from lenstronomywrapper.LensSystem.MacroLensComponents.macromodel_base import ComponentBase
import numpy as np
from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian

class PowerLawShear(ComponentBase):

    def __init__(self, redshift, kwargs_init=None, theta_E=1., gamma=2.,
                 shear_mag=0.03, shear_angle=0., center_x=0., center_y=0., e1=0.1, e2=0., convention_index=False,
                 reoptimize=False):

        self._reoptimize = reoptimize
        gamma1, gamma2 = shear_polar2cartesian(np.pi * shear_angle/180, shear_mag)
        if kwargs_init is None:
            kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y, 'e1': e1, 'e2': e2, 'gamma': gamma},
                            {'gamma1': gamma1, 'gamma2': gamma2}]

        super(PowerLawShear, self).__init__(self.lens_model_list, [redshift]*self.n_models, kwargs_init, convention_index)

    @classmethod
    def from_cartesian_shear(cls, redshifts, kwargs_init=None, theta_E=1, gamma=2, gamma1=0.05,
                             gamma2=0.0, center_x=0, center_y=0, e1=0.1, e2=0, convention_index=False, reoptimize=False):

        phi, shear_mag = shear_cartesian2polar(gamma1, gamma2)
        shear_angle = 180*phi/np.pi
        powerlawshear = cls(redshifts, kwargs_init, theta_E, gamma,
                   shear_mag, shear_angle, center_x, center_y, e1, e2, convention_index, reoptimize)

        return powerlawshear

    @property
    def n_models(self):
        return 2

    @property
    def fixed_models(self):
        return [{}, {'ra_0': 0, 'dec_0': 0}]

    @property
    def param_init(self):

        if self._reoptimize:
            return self._kwargs
        else:
            # basically random
            return [{'theta_E': 1., 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.1, 'e2': -0.1, 'gamma': 2.},
                    {'gamma1': 0.04, 'gamma2': 0.01}]

    @property
    def param_sigma(self):

        if self._reoptimize:
            return [{'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.05},
                    {'gamma1': 0.02, 'gamma2': 0.02}]
        else:
            return [{'theta_E': 0.3, 'center_x': 0.2, 'center_y': 0.2, 'e1': 0.3, 'e2': 0.3, 'gamma': 0.2},
                    {'gamma1': 0.1, 'gamma2': 0.1}]

    @property
    def param_lower(self):

        lower = [{'theta_E': 0.05, 'center_x': -2., 'center_y': -2., 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5},
                 {'gamma1': -0.4, 'gamma2': -0.4}]
        return lower

    @property
    def param_upper(self):

        upper = [{'theta_E': 3., 'center_x': 2., 'center_y': 2., 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5},
                 {'gamma1': 0.4, 'gamma2': 0.4}]
        return upper

    @property
    def lens_model_list(self):
        return ['SPEMD', 'SHEAR']

    @property
    def redshift_list(self):
        return [self.zlens] * len(self.lens_model_list)
