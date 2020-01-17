from lenstronomy.LightModel.light_model import LightModel
from lenstronomywrapper.LensSystem.light_reconstruct_base import LightReconstructBase

class SersicLens(LightReconstructBase):

    def __init__(self, kwargs_sersic, reoptimize=False):

        self._reoptimize = reoptimize
        self._kwargs = kwargs_sersic
        self._light_x, self._light_y = kwargs_sersic['center_x'], kwargs_sersic['center_y']

        super(SersicLens).__init__()

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
    def fixed_models(self):
        return [{}]

    @property
    def light_model_list(self):
        return ['SERSIC']

    @property
    def kwargs_light(self):

        return [self._kwargs]

    @property
    def lensLight(self):
        return LightModel(self.light_model_list)

    @property
    def param_init(self):

        if self._reoptimize:
            return self.kwargs_light
        else:
            # basically random
            return [{'amp': 1000, 'R_sersic': 0.3, 'n_sersic': 4.0, 'center_x': 0., 'center_y': 0.}]

    @property
    def param_sigma(self):

        if self._reoptimize:
            return [
                {'amp': 500, 'R_sersic': 0.2, 'n_sersic': 0.5, 'center_x': 0.05, 'center_y': 0.05}]
        else:
            return [{'amp': 1000, 'R_sersic': 0.8, 'n_sersic': 1.5, 'center_x': 0.2, 'center_y': 0.2}]

    @property
    def param_lower(self):

        lower = [{'amp': 1, 'R_sersic': 0.01, 'n_sersic': 0.1, 'center_x': -2., 'center_y': -2.}]
        return lower

    @property
    def param_upper(self):

        upper = [{'amp': 50000, 'R_sersic': 5, 'n_sersic': 9, 'center_x': 2., 'center_y': 2.}]
        return upper
