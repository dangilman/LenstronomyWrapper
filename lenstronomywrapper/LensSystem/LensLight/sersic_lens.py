from lenstronomywrapper.LensSystem.LensLight.light_base import LightBase
import numpy as np

class SersicLens(LightBase):

    def __init__(self, kwargs_sersic, reoptimize=True, prior=[], concentric_with_model=None):

        self._reoptimize = reoptimize
        self._kwargs = kwargs_sersic

        super(SersicLens, self).__init__(concentric_with_model, prior)

    @property
    def fixed_models(self):
        return [{}]

    @property
    def light_model_list(self):
        return ['SERSIC']

    @property
    def kwargs_light(self):

        return self._kwargs

    @property
    def param_init(self):

        return self.kwargs_light

    @property
    def param_sigma(self):
        return [{'amp': 1500, 'R_sersic': 0.5, 'n_sersic': 1.5, 'center_x': 0.5, 'center_y': 0.5}]

    @property
    def param_lower(self):

        lower_x, lower_y = -10, -10
        lower = [{'amp': 0.0000001, 'R_sersic': 0.0001, 'n_sersic': 0.1, 'center_x': lower_x, 'center_y': lower_y}]
        return lower

    @property
    def param_upper(self):

        upper_x, upper_y = 10, 10
        upper = [{'amp': 500000, 'R_sersic': 5, 'n_sersic': 9, 'center_x': upper_x, 'center_y': upper_y}]
        return upper
