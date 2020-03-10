from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase

class Shapelet(SourceBase):

    def __init__(self, kwargs_shapelet, reoptimize=True, prior=[], concentric_with_source=None):

        self.reoptimize = reoptimize

        source_x, source_y = kwargs_shapelet[0]['center_x'], kwargs_shapelet[0]['center_y']

        self._kwargs = kwargs_shapelet
        self._nmax = int(kwargs_shapelet[0]['n_max'])

        super(Shapelet, self).__init__(concentric_with_source, prior, source_x, source_y)

    @property
    def fixed_models(self):
        return [{'n_max': int(self._nmax)}]

    @property
    def light_model_list(self):
        return ['SHAPELETS']

    @property
    def kwargs_light(self):

        return self._kwargs

    @property
    def param_init(self):

        return self.kwargs_light

    @property
    def param_sigma(self):

        if self.reoptimize:

            amp_scale, beta_scale, centroid_scale = 0.2, 0.1, 0.1

            old_kwargs = self._kwargs[0]
            new_kwargs = {}
            for key in self._kwargs[0].keys():
                if key == 'center_x' or key == 'center_y':
                    new_kwargs[key] = max(0.001, centroid_scale * old_kwargs[key])
                elif key == 'amp':
                    new_kwargs[key] = max(1., amp_scale * old_kwargs[key])
                elif key == 'n_max':
                    new_kwargs[key] = self._nmax
                elif key == 'beta':
                    new_kwargs[key] = max(0.00001, beta_scale * old_kwargs[key])
                else:
                    raise Exception('param name ' + str(key) + 'not recognized.')

            return [new_kwargs]
        else:
            return [{'amp': 5000., 'beta': 0.05, 'n_max': int(self._nmax), 'center_x': 0.1, 'center_y': 0.1}]

    @property
    def param_lower(self):

        return [{'amp': 0, 'beta': 0, 'n_max': 0, 'center_x': -100, 'center_y': -100}]

    @property
    def param_upper(self):

        return [{'amp': 100, 'beta': 100, 'n_max': 150, 'center_x': 100, 'center_y': 100}]
