from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

class CircularSersic(ComponentBase):

    def __init__(self, redshift, kwargs_init, prior=[],
                 fixed=False, convention_index=False, reoptimize=False):

        self._redshift = redshift
        self._prior = prior
        self.reoptimize = reoptimize

        super(CircularSersic, self).__init__(self.lens_model_list, [redshift], kwargs_init,
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

        if self.reoptimize:
            return [{'k_eff': 0.05, 'R_sersic': 0.25, 'n_sersic': 0.5,'center_x': 0.1, 'center_y': 0.1}]
        else:
            return [{'k_eff': 0.5, 'R_sersic': 1., 'n_sersic': 1.5, 'center_x': 0.5, 'center_y': 0.5}]

    @property
    def param_lower(self):
        lower = [{'k_eff': 0., 'R_sersic': 0., 'n_sersic': 0.5,
                           'center_x': -100.,
                           'center_y': -100.}]
        return lower

    @property
    def param_upper(self):
        upper = [{'k_eff': 100., 'R_sersic': 100., 'n_sersic': 8.,
                           'center_x': 100.,
                           'center_y': 100.}]
        return upper

    @property
    def lens_model_list(self):
        return ['SERSIC']

    @property
    def redshift_list(self):
        return [self._redshift] * len(self.lens_model_list)
