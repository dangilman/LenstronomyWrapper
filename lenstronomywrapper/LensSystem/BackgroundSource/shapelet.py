from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase

class Shapelet(SourceBase):

    def __init__(self, kwargs_shapelet, n_max, reoptimize=True, prior=[], concentric_with_source=None):

        self._reoptimize = reoptimize

        kwargs_shapelet[0]['n_max'] = int(n_max)
        source_x, source_y = kwargs_shapelet[0]['center_x'], kwargs_shapelet[0]['center_y']

        self._kwargs = kwargs_shapelet
        self._nmax = int(n_max)

        super(Shapelet, self).__init__(concentric_with_source, prior, source_x, source_y)

    @property
    def fixed_models(self):
        return [{'n_max': self._nmax}]

    @property
    def light_model_list(self):
        return ['SHAPELETS']

    @property
    def kwargs_light(self):

        return self._kwargs

    @property
    def param_init(self):

        if self._reoptimize:
            return self.kwargs_light
        else:
            # basically random
            return [{'amp': 1000., 'beta': 0., 'n_max': int(self._nmax), 'center_x': 0., 'center_y': 0.}]

    @property
    def param_sigma(self):

        return [{'amp': 10000., 'beta': 0.1, 'n_max': int(self._nmax), 'center_x': 0.1, 'center_y': 0.1}]

    @property
    def param_lower(self):

        return [{'amp': 0, 'beta': 0, 'n_max': 0, 'center_x': -100, 'center_y': -100}]

    @property
    def param_upper(self):

        return [{'amp': 100, 'beta': 100, 'n_max': 150, 'center_x': 100, 'center_y': 100}]
