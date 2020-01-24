from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase
import numpy as np
from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian, ellipticity2phi_q, phi_q2_ellipticity

class PowerLawShearConvergence(ComponentBase):

    def __init__(self, redshift, kwargs_init=None, theta_E=1., gamma=2.,
                 shear=0.03, shear_angle=0., center_x=0., center_y=0., ellip=0.1, ellip_angle=0., kappa_ext=0.,
                 convention_index=False, reoptimize=False, prior=[]):

        self._reoptimize = reoptimize
        self._prior = prior

        if kwargs_init is None:
            gamma1, gamma2 = shear_polar2cartesian(np.pi * shear_angle / 180, shear)
            e1, e2 = phi_q2_ellipticity(ellip_angle * np.pi/180, 1-ellip)
            kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y, 'e1': e1, 'e2': e2, 'gamma': gamma},
                            {'gamma1': gamma1, 'gamma2': gamma2}, {'kappa_ext': kappa_ext}]

        super(PowerLawShearConvergence, self).__init__(self.lens_model_list,
                                                       [redshift]*self.n_models, kwargs_init,
                                                       convention_index, False, reoptimize)

    @classmethod
    def from_cartesian(cls, redshifts, kwargs_init=None, theta_E=1, gamma=2, gamma1=0.05,
                       gamma2=0.0, center_x=0, center_y=0, e1=0.1, e2=0, kappa_ext=0.,
                       convention_index=False, reoptimize=False, prior=[]):

        phi, shear_mag = shear_cartesian2polar(gamma1, gamma2)
        phi_ellip, q = ellipticity2phi_q(e1, e2)
        shear_angle = 180 * phi / np.pi
        ellip = 1 - q
        ellip_angle = phi_ellip * 180 / np.pi

        powerlawshearconv = cls(redshifts, kwargs_init, theta_E, gamma,
                            shear_mag, shear_angle, center_x, center_y, ellip, ellip_angle, kappa_ext, convention_index,
                            reoptimize, prior)

        return powerlawshearconv

    @property
    def n_models(self):
        return 3

    @property
    def priors(self):

        indexes = []
        priors = []
        for prior in self._prior:
            pname = prior[0]
            if pname == 'kappa_ext':
                idx = 2
            elif pname == 'gamma1' or pname == 'gamma2':
                idx = 1
            else:
                idx = 0
            indexes.append(idx)
            priors.append(prior)

        return indexes, priors

    @property
    def fixed_models(self):
        if self.fixed:
            return self.kwargs
        else:
            return [{}] * self.n_models

    @property
    def fixed_models(self):
        if self.fixed:
            return self.kwargs
        else:
            return [{}, {'ra_0': 0, 'dec_0': 0}, {'ra_0': 0, 'dec_0': 0}]

    @property
    def param_init(self):

        if self._reoptimize:
            return self.reoptimize_sigma
        else:
            # basically random
            return [{'theta_E': 1., 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.2, 'e2': -0.2, 'gamma': 2.},
                    {'gamma1': 0.04, 'gamma2': 0.04}, {'kappa_ext': 0.}]

    @property
    def param_sigma(self):

        return [{'theta_E': 0.5, 'center_x': 0.2, 'center_y': 0.2, 'e1': 0.4, 'e2': 0.4, 'gamma': 0.25},
                {'gamma1': 0.1, 'gamma2': 0.1}]

    @property
    def param_lower(self):

        lower = [{'theta_E': 0.05, 'center_x': -2., 'center_y': -2., 'e1': -0.9, 'e2': -0.9, 'gamma': 1.5},
                 {'gamma1': -0.4, 'gamma2': -0.4}, {'kappa_ext': -0.2}]
        return lower

    @property
    def param_upper(self):

        upper = [{'theta_E': 3., 'center_x': 2., 'center_y': 2., 'e1': 0.6, 'e2': 0.6, 'gamma': 2.5},
                 {'gamma1': 0.4, 'gamma2': 0.4}, {'kappa_ext': 0.2}]
        return upper

    @property
    def lens_model_list(self):
        return ['SPEMD', 'SHEAR', 'CONVERGENCE']

    @property
    def redshift_list(self):
        return [self.zlens] * len(self.lens_model_list)

    @property
    def kwargs(self):
        return self._kwargs
