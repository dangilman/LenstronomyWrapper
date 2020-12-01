from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

class NFW(ComponentBase):

    def __init__(self, redshift, kwargs_init, prior=[],
                 convention_index=False, reoptimize=False,
                 concentric_with_lens_model=None,
                 concentric_with_lens_light=None, kwargs_fixed=None,
                 custom_prior=None
                 ):

        self._redshift = redshift
        self._prior = prior
        self.reoptimize = reoptimize

        self._concentric_with_lens_model = concentric_with_lens_model
        self._concentric_with_lens_light = concentric_with_lens_light
        self.kwargs_fixed = kwargs_fixed

        super(NFW, self).__init__(self.lens_model_list, [redshift], kwargs_init,
                                           convention_index, False, reoptimize, custom_prior)

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
    def fixed_models(self):

        if self.fixed:
            return self.kwargs
        else:
            if self.kwargs_fixed is None:
                return [{}]
            else:
                return [self.kwargs_fixed]

    @property
    def param_init(self):

        return self.kwargs

    @property
    def param_sigma(self):

        if self.reoptimize:
            return [{'alpha_Rs': 0.05, 'Rs': 0.2, 'center_x': 0.05, 'center_y': 0.05, 'e1': 0.1, 'e2': 0.1}]
        else:
            return [{'alpha_Rs': 0.5, 'Rs': 2., 'center_x': 0.5, 'center_y': 0.5, 'e1': 0.5, 'e2': 0.5}]

    @property
    def param_lower(self):

        lower = [{'alpha_Rs': 0.0, 'Rs': 1e-3,
                  'e1': -0.5, 'e2': -0.5, 'center_x': -100.,
                  'center_y': -100.}]

        return lower

    @property
    def param_upper(self):
        upper = [{'alpha_Rs': 20.0, 'Rs': 10,
                           'e1': 0.5, 'e2': 0.5, 'center_x': 100.,
                           'center_y': 100.}]


        return upper

    @property
    def lens_model_list(self):
        return ['NFW_ELLIPSE']

    @property
    def redshift_list(self):
        return [self._redshift] * len(self.lens_model_list)
