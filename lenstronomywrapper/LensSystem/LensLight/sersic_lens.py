from lenstronomywrapper.LensSystem.light_reconstruct_base import LightReconstructBase
import numpy as np

class SersicLens(LightReconstructBase):

    def __init__(self, kwargs_sersic, reoptimize=True, prior=[], concentric_with_model=None):

        self._reoptimize = reoptimize
        self._kwargs = kwargs_sersic
        self._prior = prior

        super(SersicLens, self).__init__(concentric_with_model=concentric_with_model)

    def surface_brightness(self, xgrid, ygrid, lensmodel, lensmodel_kwargs):

        source_light_instance = self.sourceLight

        try:
            beta_x, beta_y = lensmodel.ray_shooting(xgrid, ygrid, lensmodel_kwargs)
            surf_bright = source_light_instance.surface_brightness(beta_x, beta_y, self.kwargs_light)

        except:
            shape0 = xgrid.shape
            beta_x, beta_y = lensmodel.ray_shooting(xgrid.ravel(), ygrid.ravel(), lensmodel_kwargs)
            surf_bright = source_light_instance.surface_brightness(beta_x, beta_y, self.kwargs_light)
            surf_bright = surf_bright.reshape(shape0, shape0)

        return surf_bright

    @property
    def priors(self):

        indexes = []
        priors = []
        for prior in self._prior:
            idx = 0
            indexes.append(idx)
            priors.append(prior)

        return indexes, priors

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
        if self._reoptimize:
            return self.reoptimize_sigma
        else:
            return [{'amp': 500, 'R_sersic': 0.2, 'n_sersic': 0.5, 'center_x': 0.05, 'center_y': 0.05}]

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
