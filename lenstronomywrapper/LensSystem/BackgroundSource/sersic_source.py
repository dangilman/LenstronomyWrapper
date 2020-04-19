from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase
import numpy as np

class SersicSource(SourceBase):

    def __init__(self, kwargs_sersic, reoptimize=False, prior=[],
                 concentric_with_source=None, redshift=None):

        self.reoptimize = reoptimize
        self._kwargs = kwargs_sersic
        source_x, source_y = kwargs_sersic[0]['center_x'], kwargs_sersic[0]['center_y']

        super(SersicSource, self).__init__(concentric_with_source, prior, source_x, source_y,
                                           redshift)

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

        return self.kwargs_light

    @property
    def param_sigma(self):

        if self.reoptimize:

            amp_scale, r_sersic_scale, n_sersic_scale, centroid_scale, e12_scale = 0.2, 0.05, 0.1, 0.1, 0.05

            old_kwargs = self._kwargs[0]
            new_kwargs = {}
            for key in self._kwargs[0].keys():
                if key == 'center_x' or key == 'center_y':
                    new_kwargs[key] = max(0.001, centroid_scale * old_kwargs[key])
                elif key == 'n_sersic':
                    new_kwargs[key] = max(0.025, n_sersic_scale * old_kwargs[key])
                elif key == 'amp':
                    new_kwargs[key] = max(1., amp_scale * old_kwargs[key])
                elif key == 'R_sersic':
                    new_kwargs[key] = max(0.01, r_sersic_scale * old_kwargs[key])
                elif key == 'e1' or key == 'e2':
                    new_kwargs[key] = max(0.0005, np.absolute(e12_scale * old_kwargs[key]))
                else:
                    raise Exception('param name '+str(key) + ' not recognized.')

            return [new_kwargs]
        else:
            return [{'amp': 1000, 'R_sersic': 0.3, 'n_sersic': 2.0, 'center_x': 0.2, 'center_y': 0.2,
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
