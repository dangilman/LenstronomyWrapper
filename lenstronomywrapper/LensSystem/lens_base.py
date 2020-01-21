from scipy.optimize import minimize
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import numpy as np

class LensBase(object):

    def __init__(self, macromodel, z_source, substructure_realization, pyhalo_cosmology):

        self.macromodel = macromodel
        self.zlens = macromodel.zlens
        self.zsource = z_source

        self.pyhalo_cosmology = pyhalo_cosmology
        self.astropy = pyhalo_cosmology.astropy

        self.lens_cosmo = LensCosmo(self.zlens, z_source, self.astropy)

        self.update_realization(substructure_realization)

        self.update_light_centroid(0, 0)

        self._saved_lensmodel, self._saved_kwargs_lens = None, None

    def fit(self, data_to_fit, optimization_class, verbose=False):

        optimizer = optimization_class(self)
        kwargs_lens_final, lens_model_full, return_kwargs = optimizer.\
            optimize(data_to_fit, verbose=verbose)

        return kwargs_lens_final, lens_model_full, return_kwargs

    def update_light_centroid(self, light_x, light_y):

        self.light_centroid_x = light_x
        self.light_centroid_y = light_y

    @property
    def realization(self):
        return self.substructure_realization

    def update_realization(self, realization):

        self.substructure_realization = realization

    def update_kwargs_macro(self, new_kwargs):

        self.macromodel.update_kwargs(new_kwargs[0:self.macromodel.n_lens_models])

    def get_kwargs_macro(self, include_substructure=True):

        return self.get_lenstronomy_args(include_substructure)[2]

    def get_lensmodel(self, include_substructure=True):

        names, redshifts, kwargs, numercial_alpha_class, convention_index = self.get_lenstronomy_args(include_substructure)
        lensModel = LensModel(names, lens_redshift_list=redshifts, z_lens=self.zlens, z_source=self.zsource,
                              multi_plane=True, numerical_alpha_class=numercial_alpha_class,
                              observed_convention_index=convention_index, cosmo=self.astropy)
        return lensModel, kwargs

    def get_lenstronomy_args(self, include_substructure=True):

        lens_model_names, macro_redshifts, macro_kwargs, convention_index = self.macromodel.get_lenstronomy_args()
        realization = self.realization
        if realization is not None and include_substructure:
            log_mlow = realization._logmlow
            halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = realization.lensing_quantities(log_mlow, log_mlow)
        else:
            halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = [], [], [], None
        halo_redshifts = list(halo_redshifts)
        names = lens_model_names + halo_names
        redshifts = macro_redshifts + halo_redshifts
        kwargs = macro_kwargs + kwargs_halos

        return names, redshifts, kwargs, kwargs_lenstronomy, convention_index

    def solve_lens_equation(self, lensmodel, kwargs_lens, precision_limit=10**-4):

        solver = LensEquationSolver(lensmodel)
        x_image, y_image = solver.findBrightImage(self.source_centroid_x, self.source_centroid_y, kwargs_lens,
                                                  search_window=4., precision_limit=precision_limit**2)
        return x_image, y_image

    @staticmethod
    def physical_location_deflector(lensmodel, kwargs, idx):

        kwargs_new = lensmodel.lens_model._convention(kwargs)
        return kwargs_new[idx]['center_x'], kwargs_new[idx]['center_y']
