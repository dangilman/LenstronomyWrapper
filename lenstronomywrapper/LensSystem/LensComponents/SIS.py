from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

class SISsatellite(ComponentBase):

    def __init__(self, redshift, kwargs_init=None,
                 prior=[], fixed=False, convention_index=False, reoptimize=False,
                 concentric_with_lens_model=None,
                 concentric_with_lens_light=None):

        self._redshift = redshift
        self._prior = prior
        self.reoptimize = reoptimize

        self._concentric_with_lens_model = concentric_with_lens_model
        self._concentric_with_lens_light = concentric_with_lens_light

        super(SISsatellite, self).__init__(self.lens_model_list, [redshift], kwargs_init,
                                           convention_index, fixed, reoptimize)

    @property
    def concentric_with_lens_light(self):

        return self._concentric_with_lens_light

    @property
    def concentric_with_lens_model(self):

        return self._concentric_with_lens_model

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
            return [{'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}]
        else:
            return [{'theta_E': 0.3, 'center_x': 0.3, 'center_y': 0.3}]

    @property
    def param_lower(self):
        lower = [{'theta_E': 0.001, 'center_x': -10, 'center_y': -10}]
        return lower

    @property
    def param_upper(self):
        upper = [{'theta_E': 3., 'center_x': 10, 'center_y': 10}]
        return upper

    @property
    def lens_model_list(self):
        return ['SIS']

    @property
    def redshift_list(self):
        return [self._redshift] * len(self.lens_model_list)
