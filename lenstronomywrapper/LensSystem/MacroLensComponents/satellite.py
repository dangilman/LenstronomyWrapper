from lenstronomywrapper.LensSystem.MacroLensComponents.macromodel_base import ComponentBase

class SISsatellite(ComponentBase):

    def __init__(self, redshift, theta_E, center_x, center_y):

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        self._redshift = redshift
        super(SISsatellite, self).__init__(self.lens_model_list, [redshift], kwargs_init, False)

    @property
    def n_models(self):
        return 1

    def set_physical_location(self, x, y):
        self.physical_x = x
        self.physical_y = y

    @property
    def fixed_models(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def param_init(self):
        return self.kwargs

    @property
    def param_sigma(self):
        return [{'theta_E': 0.1, 'center_x': 0.1, 'center_y': 0.1}]

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
