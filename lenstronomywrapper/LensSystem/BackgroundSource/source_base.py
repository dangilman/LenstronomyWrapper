from lenstronomywrapper.LensSystem.light_reconstruct_base import LightReconstructBase, LightModel

class SourceBase(LightReconstructBase):

    def __init__(self, concentric_with_source, prior, source_x, source_y):

        self.concentric_with_source = concentric_with_source
        self._prior = prior
        self._source_x, self._source_y = source_x, source_y

        super(SourceBase, self).__init__()

    def surface_brightness(self, xgrid, ygrid, lensmodel, lensmodel_kwargs):

        source_light_instance = self.sourceLight

        try:
            beta_x, beta_y = lensmodel.ray_shooting(xgrid, ygrid, lensmodel_kwargs)
            surf_bright = source_light_instance.surface_brightness(beta_x, beta_y, self.kwargs_light)

        except:
            shape0 = xgrid.shape
            beta_x, beta_y = lensmodel.ray_shooting(xgrid.ravel(), ygrid.ravel(), lensmodel_kwargs)
            surf_bright = source_light_instance.surface_brightness(beta_x, beta_y, self.kwargs_light)
            surf_bright = surf_bright.reshape(shape0, shape0)

        return surf_bright

    @property
    def source_centroid(self):
        return self._source_x, self._source_y


    @property
    def sourceLight(self):
        return LightModel(self.light_model_list)
