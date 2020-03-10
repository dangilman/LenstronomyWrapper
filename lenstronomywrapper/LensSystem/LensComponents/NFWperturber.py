from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

class NFWperturber(ComponentBase):

    def __init__(self, redshift, kwargs_init=None, param_min=None, param_max=None, prior=[], convention_index=False):

        self._prior = prior

        self._kwargs_init = kwargs_init

        self._redshift = redshift

        self.reoptimize = False

        self.x_center, self.y_center = kwargs_init[0]['center_x'], kwargs_init[0]['center_y']

        if param_min is None:
            self._logM_min, self._cmin, self._xmin, self._ymin = 1, 2, -10, -10
            param_min = [{'logM': self._logM_min, 'concentration': self._cmin, 'center_x': self._xmin, 'center_y': self._ymin}]
        if param_max is None:
            self._logM_max, self._cmax, self._xmax, self._ymax = 12, 20, 10, 10
            param_max = [{'logM': self._logM_max, 'concentration': self._cmax, 'center_x': self._xmax, 'center_y': self._ymax}]

        self._param_min, self._param_max = param_min, param_max
        super(NFWperturber, self).__init__(self.lens_model_list, [redshift]*self.n_models, self._kwargs_init,
                                            convention_index, fixed=False)

    @classmethod
    def from_Mc(cls, redshift, logM, concentration, center_x, center_y, param_min=None, param_max=None, prior=[], convention_index=False):

        kwargs_init = [{'logM': logM, 'concentration': concentration, 'center_x': center_x, 'center_y': center_y}]

        powerlawshear = cls(redshift, kwargs_init, param_min, param_max, prior, convention_index)

        return powerlawshear

    @property
    def priors(self):

        indexes = []
        priors = []
        for prior in self._prior:
            pname = prior[0]
            if pname == 'gamma1' or pname == 'gamma2':
                idx = 1
            else:
                idx = 0
            indexes.append(idx)
            priors.append(prior)

        return indexes, priors

    @property
    def n_models(self):
        return 1

    @property
    def param_init(self):

        return self._kwargs_init

    @property
    def param_sigma(self):

        return [{'logM': 2, 'concentration': 3, 'center_x': 0.5, 'center_y': 0.5}]

    @property
    def param_lower(self):
        return self._param_min

    @property
    def param_upper(self):
        return self._param_max

    @property
    def lens_model_list(self):
        return ['NFW_MC']

    @property
    def redshift_list(self):
        return [self._redshift]
