class LightReconstructBase(object):

    def __init__(self):
        pass

    @property
    def source_centroid(self):
        raise Exception('Source centroid not definied for this class as there might be possibly more '
                        'than one component non-concentric.')
    @property
    def fixed_models(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def light_model_list(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def kwargs_light(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def sourceLight(self):
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
