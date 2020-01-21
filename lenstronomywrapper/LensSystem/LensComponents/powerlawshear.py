from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase
import numpy as np
from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian, ellipticity2phi_q, phi_q2_ellipticity

class PowerLawShear(ComponentBase):

    def __init__(self, redshift, kwargs_init=None, theta_E=1., gamma=2.,
                 shear=0.03, shear_angle=0., center_x=0., center_y=0., ellip=0.1, ellip_angle=0., convention_index=False,
                 reoptimize=False, prior=[]):

        self._reoptimize = reoptimize
        self._prior = prior

        if kwargs_init is None:
            gamma1, gamma2 = shear_polar2cartesian(np.pi * shear_angle / 180, shear)
            e1, e2 = phi_q2_ellipticity(ellip_angle * np.pi/180, 1-ellip)
            kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y, 'e1': e1, 'e2': e2, 'gamma': gamma},
                            {'gamma1': gamma1, 'gamma2': gamma2}]

        self.x_center, self.y_center = center_x, center_y

        super(PowerLawShear, self).__init__(self.lens_model_list, [redshift]*self.n_models,
                                            kwargs_init, convention_index, fixed=False)

    @classmethod
    def from_cartesian(cls, redshifts, kwargs_init=None, theta_E=1, gamma=2, gamma1=0.05,
                             gamma2=0.0, center_x=0, center_y=0, e1=0.1, e2=0, convention_index=False,
                       reoptimize=False, prior=[]):

        phi, shear_mag = shear_cartesian2polar(gamma1, gamma2)
        phi_ellip, q = ellipticity2phi_q(e1, e2)
        shear_angle = 180*phi/np.pi
        ellip = 1-q
        ellip_angle = phi_ellip * 180/np.pi

        powerlawshear = cls(redshifts, kwargs_init, theta_E, gamma,
                   shear_mag, shear_angle, center_x, center_y, ellip, ellip_angle, convention_index, reoptimize, prior)

        return powerlawshear

    @property
    def priors(self):

        indexes = []
        priors = []
        for prior in self._prior:
            pname = prior[0]
            if pname == 'gamma1' or pname == 'gamma2':
                idx = 1
            else:
                idx = 0
            indexes.append(idx)
            priors.append(prior)

        return indexes, priors

    @property
    def n_models(self):
        return 2

    @property
    def fixed_models(self):
        if self.fixed:
            return self.kwargs
        else:
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
