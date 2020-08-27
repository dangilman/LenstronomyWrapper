class ReconstructBase(object):

    def __init__(self):
        pass

    @property
    def concentric_with_lens_light(self):
        raise Exception('linked lens light with lens model not implemented for this class')

    @property
    def concentric_with_lens_model(self):
        raise Exception('linked lens model with lens model not implemented for this class')

    @property
    def n_models(self):
        return len(self.light_model_list)

    @property
    def light_model_list(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def fixed_models(self):
        return NotImplementedError('Source reconstruction not yet implemented for this source class.')

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

    @property
    def lens_model_list(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def redshift_list(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')

    @property
    def kwargs(self):
        raise NotImplementedError('Source reconstruction not yet implemented for this source class.')
