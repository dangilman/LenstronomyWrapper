from lenstronomy.LightModel.light_model import LightModel

class LightReconstructBase(object):

    @property
    def n_models(self):
        return len(self.light_model_list)

    @property
    def concentric_with_lens_light(self):
        return self._concentric_with_lens_light

    @property
    def concentric_with_lens_model(self):
        return self._concentric_with_lens_model

    @property
    def lensLight(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def light_model_list(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def kwargs_light(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def param_init(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def param_sigma(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def param_lower(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def param_upper(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')
