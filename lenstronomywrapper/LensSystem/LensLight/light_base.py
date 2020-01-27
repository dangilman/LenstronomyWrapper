from lenstronomywrapper.LensSystem.light_reconstruct_base import LightReconstructBase, LightModel

class LightBase(LightReconstructBase):

    def __init__(self, concentric_with_model, priors):

        self._prior = priors
        self.concentric_with_model = concentric_with_model
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
    def priors(self):

        indexes = []
        priors = []
        for prior in self._prior:
            idx = 0
            indexes.append(idx)
            priors.append(prior)

        return indexes, priors

    @property
    def lensLight(self):
        return LightModel(self.light_model_list)
