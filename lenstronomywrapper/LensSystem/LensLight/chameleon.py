from lenstronomywrapper.LensSystem.LensLight.light_base import LightBase
import numpy as np

class DoubleChameleon(LightBase):

    def __init__(self, kwargs_chameleon, reoptimize=False, prior=[],
                 concentric_with_model=None, concentric_with_lens_light=None,
                 custom_prior=None):

        self.reoptimize = reoptimize
        self._kwargs = kwargs_chameleon

        self._concentric_with_lens_light = concentric_with_lens_light

        super(DoubleChameleon, self).__init__(concentric_with_model, prior, custom_prior)

    @property
    def n_models(self):
        return 1

    @property
    def concentric_with_lens_light(self):

        return self._concentric_with_lens_light

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
        return ['DOUBLE_CHAMELEON']

    @property
    def kwargs_light(self):

        return self._kwargs

    @property
    def param_init(self):

        return self.kwargs_light

    @property
    def param_sigma(self):

        if self.reoptimize:

            amp_scale, w_scale, centroid_scale, e12_scale = 0.2, 0.2, 0.1, 0.1

            old_kwargs = self._kwargs[0]
            new_kwargs = {}
            for key in self._kwargs[0].keys():
                if key == 'center_x' or key == 'center_y':
                    new_kwargs[key] = abs(centroid_scale * old_kwargs[key])
                elif key == 'amp':
                    new_kwargs[key] = abs(amp_scale * old_kwargs[key])
                elif key in ['w_c1', 'w_t1', 'w_c2', 'w_t2']:
                    new_kwargs[key] = abs(w_scale * old_kwargs[key])
                elif key in ['e12', 'e22']:
                    new_kwargs[key] = abs(e12_scale * old_kwargs[key])
                elif key == 'ratio':
                    new_kwargs[key] = abs(0.25 * old_kwargs[key])
                else:
                    raise Exception('param name ' + str(key) + 'not recognized.')
            return [new_kwargs]

        else:
            return [{'amp1': 2000., 'ratio': 1.,'w_c1': 2., 'w_t1': 2., 'w_c2': 2., 'w_t2': 2.,
                     'e11': 0.3, 'e12': 0.3, 'e21': 0.3, 'e22': 0.3,
                     'center_x': 0.3, 'center_y': 0.3}]

    @property
    def param_lower(self):

        lower_x, lower_y = -10, -10
        lower = [{'amp1': 0., 'ratio': 1e-3, 'w_c1': 1e05, 'w_t1': 1e-5, 'w_c2': 1e-5, 'w_t2': 1e-5,
                     'e11': -0.5, 'e12': -0.5, 'e21': -0.5, 'e22': -0.5,
                     'center_x': lower_x, 'center_y': lower_y}]
        return lower

    @property
    def param_upper(self):

        upper_x, upper_y = 10, 10
        upper = [{'amp1': 0., 'ratio': 100, 'w_c1': 5, 'w_t1': 5, 'w_c2':5, 'w_t2': 5,
                  'e11': 0.5, 'e12': 0.5, 'e21': 0.5, 'e22': 0.5,
                  'center_x': upper_x, 'center_y': upper_y}]
        return upper
