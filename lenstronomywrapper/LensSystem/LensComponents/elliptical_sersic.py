from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

class EllpiticalSersic(ComponentBase):

    def __init__(self, redshift, kwargs_init, prior=[],
                 fixed=False, convention_index=False, reoptimize=False, concentric_with_model=None,
                 fixed_sersic_index=None):

        self._redshift = redshift
        self._prior = prior
        self.reoptimize = reoptimize
        self._fixed_sersic_index = fixed_sersic_index

        super(EllpiticalSersic, self).__init__(self.lens_model_list, [redshift], kwargs_init,
                                           convention_index, fixed, reoptimize, concentric_with_model)

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
    def fixed_models(self):

        if self.fixed:
            return self.kwargs
        else:
            if self._fixed_sersic_index is not None:
                return [{'n_sersic': self._fixed_sersic_index}]
            else:
                return [{''}]

    @property
    def param_init(self):

        return self.kwargs

    @property
    def param_sigma(self):

        if self._fixed_sersic_index is not None:
            n_sersic_sigma = 1e-9
        else:
            n_sersic_sigma = 1.5

        if self.reoptimize:
            return [{'k_eff': 0.05, 'R_sersic': 0.25, 'n_sersic': n_sersic_sigma/3., 'e1': 0.2, 'e2': 0.2, 'center_x': 0.1, 'center_y': 0.1}]
        else:
            return [{'k_eff': 0.5, 'R_sersic': 1., 'n_sersic': n_sersic_sigma, 'e1': 0.3, 'e2': 0.3, 'center_x': 0.5, 'center_y': 0.5}]

    @property
    def param_lower(self):

        lower = [{'k_eff': 0., 'R_sersic': 0.,
                  'e1': -0.5, 'e2': -0.5, 'center_x': -100.,
                  'center_y': -100.}]

        if self._fixed_sersic_index is None:
            lower[0]['n_sersic'] = 0.5
        else:
            lower[0]['n_sersic'] = self._fixed_sersic_index * 0.9999

        return lower

    @property
    def param_upper(self):
        upper = [{'k_eff': 100., 'R_sersic': 100.,
                           'e1': 0.5, 'e2': 0.5, 'center_x': 100.,
                           'center_y': 100.}]

        if self._fixed_sersic_index is None:
            upper[0]['n_sersic'] = 9
        else:
            upper[0]['n_sersic'] = self._fixed_sersic_index * 1.00001

        return upper

    @property
    def lens_model_list(self):
        return ['SERSIC_ELLIPSE_GAUSS_DEC']

    @property
    def redshift_list(self):
        return [self._redshift] * len(self.lens_model_list)
