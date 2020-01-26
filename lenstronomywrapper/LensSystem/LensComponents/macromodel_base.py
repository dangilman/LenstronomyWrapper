from lenstronomywrapper.LensSystem.lens_reconstruct_base import ReconstructBase

class ComponentBase(ReconstructBase):

    def __init__(self, lens_model_names, redshifts, kwargs, convention_index, fixed, reoptimize,
                 background_inflection_point):

        self.zlens = redshifts[0]
        self.redshifts = redshifts
        self.lens_model_names = lens_model_names
        self.update_kwargs(kwargs)
        self.convention_index = convention_index
        self.fixed = fixed
        self._reoptimize = reoptimize

        self.x_center, self.y_center = kwargs[0]['center_x'], kwargs[0]['center_y']

        super(ComponentBase).__init__()

    def update_prior(self, new_prior):

        self._prior = new_prior

    def lenstronomy_args(self):

        return self.lens_model_names, self.redshifts, self._kwargs, [self.convention_index]*len(self.lens_model_names)

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
