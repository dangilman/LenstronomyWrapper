from lenstronomywrapper.LensSystem.lens_reconstruct_base import ReconstructBase

class ComponentBase(ReconstructBase):

    def __init__(self, lens_model_names, redshifts, kwargs, convention_index):

        self.zlens = redshifts[0]
        self.redshifts = redshifts
        self.lens_model_names = lens_model_names
        self.update_kwargs(kwargs)
        self.convention_index = convention_index

        self.x_center, self.y_center = kwargs[0]['center_x'], kwargs[0]['center_y']

        super(ComponentBase).__init__()

    def lenstronomy_args(self):

        return self.lens_model_names, self.redshifts, self._kwargs, [self.convention_index]*len(self.lens_model_names)

    def update_kwargs(self, kwargs):

        self._kwargs = kwargs

    @property
    def kwargs(self):
        return self._kwargs
