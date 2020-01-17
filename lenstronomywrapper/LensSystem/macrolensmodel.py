class MacroLensModel(object):

    def __init__(self, components):

        if not isinstance(components, list):
            components = [components]
        self.components = components
        self.n_lens_models = self._count_models(components)

    @property
    def zlens(self):
        return self.components[0].zlens

    def update_kwargs(self, new_kwargs):

        if len(new_kwargs) != self.n_lens_models:
            raise Exception('New and existing keyword arguments must be the same length.')

        count = 0
        for model in self.components:
            n = model.n_models
            new = new_kwargs[count:(count+n)]
            model.update_kwargs(new)
            count += n

    def get_lenstronomy_args(self):

        lens_model_list, redshift_list, kwargs, observed_convention_index_bool = [], [], [], []
        for component in self.components:
            model_names, model_redshifts, model_kwargs, model_convention_index = component.lenstronomy_args()
            observed_convention_index_bool += model_convention_index

            lens_model_list += model_names
            redshift_list += model_redshifts
            kwargs += model_kwargs

        observed_convention_index = None
        for i, value in enumerate(observed_convention_index_bool):
            if value:
                if observed_convention_index is None:
                    observed_convention_index = [i]
                else:
                    observed_convention_index.append(i)

        return lens_model_list, redshift_list, kwargs, observed_convention_index

    @staticmethod
    def _count_models(components):

        n = 0
        for component in components:
            n += component.n_models
        return n

    @property
    def fixed_models(self):
        fixed_models = []
        for component in self.components:
            fixed_models += component.fixed_models
        return fixed_models

    @property
    def param_init(self):
        param_init = []
        for component in self.components:
            param_init += component.param_init
        return param_init

    @property
    def param_sigma(self):
        param_sigma = []
        for component in self.components:
            param_sigma += component.param_sigma
        return param_sigma

    @property
    def param_lower(self):
        param_lower = []
        for component in self.components:
            param_lower += component.param_lower
        return param_lower

    @property
    def param_upper(self):
        param_upper = []
        for component in self.components:
            param_upper += component.param_upper
        return param_upper

    @property
    def lens_model_list(self):
        lens_model_list, _, _, _ = self.get_lenstronomy_args()
        return lens_model_list

    @property
    def redshift_list(self):
        _, redshift_list, _, _ = self.get_lenstronomy_args()
        return redshift_list

    @property
    def kwargs(self):
        _, _, kwargs, _ = self.get_lenstronomy_args()
        return kwargs
