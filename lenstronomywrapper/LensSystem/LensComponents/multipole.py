from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

pi = 3.14159265359

class Multipole(ComponentBase):

    def __init__(self, redshift, kwargs_init=None,
                 prior=[], convention_index=False, reoptimize=False,
                 concentric_with_lens_model=None,
                 concentric_with_lens_light=None,
                 kwargs_fixed=None,
                 custom_prior=None):

        self._redshift = redshift
        self._prior = prior
        self.reoptimize = reoptimize

        self._concentric_with_lens_model = concentric_with_lens_model
        self._concentric_with_lens_light = concentric_with_lens_light
        self.kwargs_fixed = kwargs_fixed

        super(Multipole, self).__init__(self.lens_model_list, [redshift], kwargs_init,
                                           convention_index, False, reoptimize, custom_prior)

    @property
    def concentric_with_lens_light(self):

        return self._concentric_with_lens_light

    @property
    def concentric_with_lens_model(self):

        return self._concentric_with_lens_model

    @property
    def fixed_models(self):

        if self.fixed:
            return self.kwargs
        else:
            if self.kwargs_fixed is None:
                return [{}]
            else:
                return [self.kwargs_fixed]

    @property
    def n_models(self):
        return 1

    def set_physical_location(self, x, y):
        self.physical_x = x
        self.physical_y = y

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
    def param_init(self):

        return self.kwargs

    @property
    def param_sigma(self):

        if self.reoptimize:
            return [{'m': 1, 'a_m': 0.005, 'phi_m': pi, 'center_x': 0.1, 'center_y': 0.1}]
        else:
            return [{'m': 1, 'a_m': 0.005, 'phi_m': pi/10, 'center_x': 1., 'center_y': 1.}]

    @property
    def param_lower(self):
        lower = [{'m': 2, 'a_m': 0.00, 'phi_m': 0., 'center_x': -10, 'center_y': -10}]
        return lower

    @property
    def param_upper(self):
        upper = [{'m': 100, 'a_m': 1., 'phi_m': 180., 'center_x': 10, 'center_y': 10}]
        return upper

    @property
    def lens_model_list(self):
        return ['MULTIPOLE']

    @property
    def redshift_list(self):
        return [self._redshift] * len(self.lens_model_list)
