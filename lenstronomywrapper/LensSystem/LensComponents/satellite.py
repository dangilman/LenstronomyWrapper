from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

class SISsatellite(ComponentBase):

    def __init__(self, redshift, kwargs_init=None, theta_E=None, center_x=None, center_y=None,
                 prior=[], fixed=False, convention_index=False, reoptimize=True):

        if kwargs_init is None:
            kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        self._redshift = redshift
        self._prior = prior

        super(SISsatellite, self).__init__(self.lens_model_list, [redshift], kwargs_init,
                                           convention_index, fixed, reoptimize)

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
