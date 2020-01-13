from lenstronomywrapper.MacroLensComponents.macromodel_base import ComponentBase

class SISsatellite(ComponentBase):

    def __init__(self, lens_model_names, redshifts, theta_E, center_x, center_y):

        kwargs_init = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y}]

        super(SISsatellite, self).__init__(lens_model_names, redshifts, kwargs_init, False)

    @property
    def n_models(self):
        return 1
