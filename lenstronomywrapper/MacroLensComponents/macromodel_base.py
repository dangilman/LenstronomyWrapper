
class ComponentBase(object):

    def __init__(self, lens_model_names, redshifts, kwargs, convention_index):

        self.redshifts = redshifts
        self.lens_model_names = lens_model_names
        self.update_kwargs(kwargs)
        self.convention_index = convention_index

    def lenstronomy_args(self):

        return self.lens_model_names, self.redshifts, self.kwargs, self.convention_index

    def update_kwargs(self, kwargs):

        self.kwargs = kwargs

