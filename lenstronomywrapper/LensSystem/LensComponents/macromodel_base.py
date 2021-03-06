from lenstronomywrapper.LensSystem.lens_reconstruct_base import ReconstructBase

class ComponentBase(ReconstructBase):

    def __init__(self, lens_model_names, redshifts, kwargs, convention_index, fixed, reoptimize,
                 custom_prior=None):

        self.zlens = redshifts[0]
        self.redshifts = redshifts
        self.lens_model_names = lens_model_names
        self.update_kwargs(kwargs)
        self.convention_index = convention_index
        self.fixed = fixed
        self._reoptimize = reoptimize

        self.x_center, self.y_center = kwargs[0]['center_x'], kwargs[0]['center_y']

        if isinstance(custom_prior, list):
            self.custom_prior = custom_prior
        else:
            if custom_prior is None or custom_prior is False:
                self.custom_prior = []
            else:
                self.custom_prior = [custom_prior]

        super(ComponentBase).__init__()

    def update_prior(self, new_prior):

        self._prior = new_prior

    def update_kwargs(self, kwargs):

        self._kwargs = kwargs

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def fixed_models(self):
        if self.fixed:
            return self.kwargs
        else:
            return [{}] * self.n_models

    @property
    def reoptimize_sigma(self):
        kwargs = self.kwargs
        kw_sigma = []
        for kw in kwargs:
            new = {}
            for key in kw.keys():
                new[key] = kw[key] * 0.15
            kw_sigma.append(new)
        return kw_sigma
