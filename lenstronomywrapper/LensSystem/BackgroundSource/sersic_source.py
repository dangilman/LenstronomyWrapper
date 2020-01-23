from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase

class SersicSource(SourceBase):

    def __init__(self, kwargs_sersic, reoptimize=True, prior=[], concentric_with_source=None):

        self._reoptimize = reoptimize
        self._kwargs = kwargs_sersic
        source_x, source_y = kwargs_sersic[0]['center_x'], kwargs_sersic[0]['center_y']

        super(SersicSource, self).__init__(concentric_with_source, prior, source_x, source_y)

    @property
    def fixed_models(self):
        return [{}]

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
            return self.reoptimize_sigma
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
