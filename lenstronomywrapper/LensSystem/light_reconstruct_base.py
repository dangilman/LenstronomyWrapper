from lenstronomy.LightModel.light_model import LightModel

class LightReconstructBase(object):

    @property
    def n_models(self):
        return len(self.light_model_list)

    @property
    def reoptimize_sigma(self):
        kwargs = self.kwargs_light
        kw_sigma = []
        for kw in kwargs:
            new = {}
            for key in kw.keys():
                new[key] = kw[key] * 0.15
            kw_sigma.append(new)
        return kw_sigma

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
