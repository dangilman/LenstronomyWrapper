from lenstronomywrapper.MacroLensComponents.macromodel_base import ComponentBase

class BackgroundSIS(ComponentBase):

    def __init__(self, lens_model_names, redshifts, theta_E, center_x, center_y, convention_index=False):

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]

        super(BackgroundSIS, self).__init__(lens_model_names, redshifts, kwargs_init, convention_index)

    @property
    def n_models(self):
        return 1
