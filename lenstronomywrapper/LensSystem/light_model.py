from lenstronomy.LightModel.light_model import LightModel as LenstronomyLightModel

class LightModel(object):

    def __init__(self, components):

        if not isinstance(components, list):
            components = [components]
        self.components = components
        self.n_light_models = self._count_models(components)

    def surface_brightness(self, xgrid, ygrid, lensmodel, lensmodel_kwargs):

        light = 0
        for component in self.components:
            light += component.surface_brightness(xgrid, ygrid, lensmodel, lensmodel_kwargs)
        return light

    @property
    def priors(self):
        priors = []
        component_index = 0
        for component in self.components:

            prior_index, prior = component.priors
            new = []
            for idx, prior_i in zip(prior_index, prior):
                new += [[idx + component_index] + prior_i]
            priors += new
            component_index += component.n_models

        return priors

    def update_kwargs(self, new_kwargs):

        if len(new_kwargs) != self.n_light_models:
            raise Exception('New and existing keyword arguments must be the same length.')

        count = 0
        for model in self.components:
            n = model.n_models
            new = new_kwargs[count:(count+n)]
            model.update_kwargs(new)
            count += n

    @property
    def lensLight(self):
        return LenstronomyLightModel(self.light_model_list)

    @property
    def sourceLight(self):
        return LenstronomyLightModel(self.light_model_list)

    @staticmethod
    def _count_models(components):

        n = 0
        for component in components:
            n += component.n_models
        return n

    @property
    def priors(self):
        priors = []
        component_index = 0
        for component in self.components:

            prior_index, prior = component.priors
            new = []
            for idx, prior_i in zip(prior_index, prior):
                new += [[idx + component_index] + prior_i]
            priors += new
            component_index += component.n_models

        return priors

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
    def light_model_list(self):
        light_model_list = []
        for component in self.components:
            light_model_list += component.light_model_list
        return light_model_list

    @property
    def kwargs_light(self):
        kwargs_light = []
        for component in self.components:
            kwargs_light += component.kwargs_light
        return kwargs_light
