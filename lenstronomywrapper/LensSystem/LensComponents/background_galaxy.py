from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase

class BackgroundSIS(ComponentBase):

    def __init__(self, redshift, theta_E, center_x, center_y, convention_index=True):

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]
        self._redshift = redshift
        super(BackgroundSIS, self).__init__(self.lens_model_list, [redshift], kwargs_init, convention_index)

    @property
    def n_models(self):
        return 1

    def set_physical_location(self, x, y):

        self.physical_x = x
        self.physical_y = y
