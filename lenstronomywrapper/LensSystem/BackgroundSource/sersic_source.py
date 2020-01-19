from lenstronomywrapper.LensSystem.light_reconstruct_base import LightReconstructBase

class SersicSource(LightReconstructBase):

    def __init__(self, kwargs_sersic, reoptimize=False, prior=[], concentric_with_source=None):

        self._reoptimize = reoptimize
        self._kwargs = kwargs_sersic
        self._prior = prior
        self._source_x, self._source_y = kwargs_sersic[0]['center_x'], kwargs_sersic[0]['center_y']

        super(SersicSource, self).__init__(concentric_with_source=concentric_with_source)

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
    def source_centroid(self):
        return self._source_x, self._source_y

    @property
    def light_model_list(self):
        return ['SERSIC_ELLIPSE']

    @property
    def kwargs_light(self):

        return self._kwargs

    @property
    def param_init(self):

        if self._reoptimize:
            return self.kwargs_light
        else:
            # basically random
            return [{'amp': 3000, 'R_sersic': 0.5, 'n_sersic': 4.0, 'center_x': 0., 'center_y': 0.,
                     'e1': 0.2, 'e2': -0.1}]

    @property
    def param_sigma(self):

        if self._reoptimize:
            return [{'amp': 500, 'R_sersic': 0.2, 'n_sersic': 0.5, 'center_x': 0.05, 'center_y': 0.05,
                     'e1': 0.1, 'e2': 0.1}]
        else:
            return [{'amp': 1000, 'R_sersic': 0.8, 'n_sersic': 1.5, 'center_x': 0.2, 'center_y': 0.2,
                     'e1': 0.25, 'e2': 0.25}]

    @property
    def param_lower(self):

        lower = [{'amp': 0.000000000001, 'R_sersic': 0.000001, 'n_sersic': 0.1, 'center_x': -2., 'center_y': -2.,
                  'e1': -0.9, 'e2': -0.9}]
        return lower

    @property
    def param_upper(self):

        upper = [{'amp': 50000, 'R_sersic': 5, 'n_sersic': 9, 'center_x': 2., 'center_y': 2.,
                  'e1': 0.9, 'e2': 0.9}]
        return upper
