from lenstronomywrapper.LensSystem.light_reconstruct_base import LightReconstructBase, LightModel

class LightBase(LightReconstructBase):

    def __init__(self, concentric_with_model, priors, custom_prior):

        self._prior = priors
        self.concentric_with_model = concentric_with_model
        self.is_source_light = False
        if isinstance(custom_prior, list):
            self.custom_prior = custom_prior
        else:
            if custom_prior is None or custom_prior is False:
                self.custom_prior = []
            else:
                self.custom_prior = [custom_prior]
        super(LightBase, self).__init__()

    def surface_brightness(self, xgrid, ygrid, lensmodel, lensmodel_kwargs):

        lens_light_instance = self.lensLight

        try:
            surf_bright = lens_light_instance.surface_brightness(xgrid, ygrid, self.kwargs_light)

        except:
            shape0 = xgrid.shape
            surf_bright = lens_light_instance.surface_brightness(xgrid, ygrid, self.kwargs_light)
            surf_bright = surf_bright.reshape(shape0, shape0)

        return surf_bright

    @property
    def component_redshift(self):
        # not implemented for lens light instances
        return None

    @property
    def lensLight(self):
        return LightModel(self.light_model_list)
