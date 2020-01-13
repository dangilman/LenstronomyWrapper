class MacroLensModel(object):

    def __init__(self, components):

        self.components = components
        self.n_lens_models = self._count_models(components)

    def update_kwargs(self, new_kwargs):

        if len(new_kwargs) != self.n_lens_models:
            raise Exception('New and existing keyword arguments must be the same length.')

        count = 0
        for model in self.components:
            n = model.n_models
            if n==1:
                model.update_kwargs(new_kwargs[count])
            else:
                model.update_kwargs(new_kwargs[count:(count+n)])
            count += n

    def get_lenstronomy_args(self):

        count = 0
        lens_model_list, redshift_list, kwargs, observed_convention_index = [], [], [], None
        for component in self.components:
            model_names, model_redshifts, model_kwargs, model_convention_inds = component.lenstronomy_args()
            if model_convention_inds is not False:
                if observed_convention_index is None:
                    observed_convention_index = [count + ind for ind in model_convention_inds]
                else:
                    observed_convention_index += [count + ind for ind in model_convention_inds]

            count += component.n_models

            lens_model_list += model_names
            redshift_list += model_redshifts
            kwargs += model_kwargs

        return lens_model_list, redshift_list, kwargs

    @property
    def _count_models(self):

        n = 0
        for component in self.components:
            n += component.n_models
        return n
