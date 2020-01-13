from lenstronomywrapper.MacroLensComponents.macromodel_base import ComponentBase
import numpy as np
from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian

class PowerLawShear(ComponentBase):

    def __init__(self, lens_model_names, redshifts, theta_E=1, gamma=2,
                 shear_mag=0.03, shear_angle=0, center_x=0, center_y=0, e1=0.1, e2=0, convention_index=False):

        gamma1, gamma2 = shear_polar2cartesian(np.pi * shear_angle/180, shear_mag)
        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y, 'e1': e1, 'e2': e2, 'gamma': gamma},
                            {'gamma1': gamma1, 'gamma2': gamma2}]

        super(PowerLawShear, self).__init__(lens_model_names, redshifts, kwargs_init, convention_index)

    @classmethod
    def from_cartesian_shear(cls, lens_model_names, redshifts, theta_E=1, gamma=2, gamma1=0.05,
                             gamma2=0.0, center_x=0, center_y=0, e1=0.1, e2=0, convention_index=False):

        phi, shear_mag = shear_cartesian2polar(gamma1, gamma2)
        shear_angle = 180*phi/np.pi
        powerlawshear = cls(lens_model_names, redshifts,  theta_E, gamma,
                   shear_mag, shear_angle, center_x, center_y, e1, e2, convention_index)

        return powerlawshear

    @property
    def n_models(self):
        return 2
