from lenstronomy.LensModel.lens_model import LensModel
from lenstronomywrapper.ExtendedSource.finte_source_magnification import ExtendedSource

class LensSystem(object):

    def __init__(self, zlens, zsource, macromodel, substructure_realization, optimization_function,
                  source_x=0, source_y=0):

        self.macromodel = macromodel
        self.zlens = zlens
        self.zsource = zsource

        self.update_realization(substructure_realization)
        self.optimization_function = optimization_function
        self.astropy = self.substructure_realization.lens_cosmo.cosmo.astropy
        self.update_source_position(source_x, source_y)

    def update_realization(self, realization):

        self.substructure_realization = realization

    def update_kwargs_macro(self, new_kwargs):

        self.macromodel.update_kwargs(new_kwargs[0:self.macromodel.n_lens_models])

    def update_source_position(self, source_x, source_y):

        self.source_x, self.source_y = source_x, source_y

    @property
    def realization(self):
        return self.substructure_realization

    def ray_shoot(self, xcoords, ycoords, lensModel=None, kwargs=None):

        if lensModel is None:
            lensModel, kwargs = self.get_lensmodel()

        betax, betay = lensModel.ray_shooting(xcoords, ycoords, kwargs)

        return betax, betay

    def get_lensmodel(self):

        names, redshifts, kwargs, numercial_alpha_class, convention_index = self.get_lenstronomy_args()
        lensModel = LensModel(names, lens_redshift_list=redshifts, z_lens=self.zlens, z_source=self.zsource,
                              multi_plane=True, numerical_alpha_class=numercial_alpha_class,
                              observed_convention_index=convention_index)
        return lensModel, kwargs

    def get_lenstronomy_args(self):

        lens_model_names, macro_redshifts, macro_kwargs, convention_index = self.macromodel.get_lenstronomy_args()
        realization = self.realization
        halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = realization.lensing_quantities()
        halo_redshifts = list(halo_redshifts)

        names = lens_model_names + halo_names
        redshifts = macro_redshifts + halo_redshifts
        kwargs = macro_kwargs + kwargs_halos

        return names, redshifts, kwargs, kwargs_lenstronomy, convention_index

    def optimize(self):

        return self.optimization_function.optimize()
