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

        self.position_convention_halo = []

        self._numerical_alpha_class = None

        self.clear_static_lensmodel()

        self.pc_per_arcsec_zsource = 1000 * pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value ** -1

    def update_light_centroid(self, light_x, light_y):

        self.light_centroid_x = light_x
        self.light_centroid_y = light_y

    def update_realization(self, realization):

        self.realization = realization

    def set_position_convention_halo(self, idx_list):

        assert isinstance(idx_list, list)
        for idx in idx_list:
            assert idx > self.macromodel.n_lens_models - 1
        self.position_convention_halo = idx_list

    def update_kwargs_macro(self, new_kwargs):

        self.macromodel.update_kwargs(new_kwargs[0:self.macromodel.n_lens_models])

    def get_kwargs_macro(self, include_substructure=True):

        return self.get_lenstronomy_args(include_substructure)[2]

    def set_lensmodel_static(self, lensmodel, kwargs):

        self._static_lensmodel = True
        self._lensmodel_static = lensmodel
        self._kwargs_static = kwargs

    def clear_static_lensmodel(self):

        self._static_lensmodel = False
        self._lensmodel_static = None
        self._kwargs_static = None

    def get_lensmodel(self, include_substructure=True, substructure_realization=None, include_macromodel=True):

        if self._static_lensmodel and include_substructure is True:

            _, _, _, numercial_alpha_class, _ = self.get_lenstronomy_args(
                True)

            self._numerical_alpha_class = numercial_alpha_class

            return self._lensmodel_static, self._kwargs_static

        names, redshifts, kwargs, numercial_alpha_class, convention_index = self.get_lenstronomy_args(
            include_substructure, substructure_realization)

        if include_macromodel is False:
            n_macro = self.macromodel.n_lens_models
            names = names[n_macro:]
            kwargs = kwargs[n_macro:]
            redshifts = list(redshifts)[n_macro:]
            if isinstance(convention_index, list) or isinstance(convention_index, np.ndarray):
                convention_index = np.array(convention_index)[n_macro:] - n_macro
                convention_index = list(convention_index)

        self._numerical_alpha_class = numercial_alpha_class

        if convention_index is None:
            if self.position_convention_halo is None:
                pass
            else:
                convention_index = self.position_convention_halo
        else:
            convention_index += self.position_convention_halo

        lensModel = LensModel(names, lens_redshift_list=redshifts, z_lens=self.zlens, z_source=self.zsource,
                              multi_plane=True, numerical_alpha_class=numercial_alpha_class,
                              observed_convention_index=convention_index, cosmo=self.astropy)
        return lensModel, kwargs

    def get_lenstronomy_args(self, include_substructure=True, realization=None, z_mass_sheet_max=None):

        lens_model_names, macro_redshifts, macro_kwargs, convention_index = \
            self.macromodel.get_lenstronomy_args()

        if realization is None:
            realization = self.realization

        if realization is not None and include_substructure:

            halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = \
                realization.lensing_quantities(z_mass_sheet_max=z_mass_sheet_max)
        else:
            halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = [], [], [], None

        halo_redshifts = list(halo_redshifts)
        names = lens_model_names + halo_names
        redshifts = macro_redshifts + halo_redshifts
        kwargs = macro_kwargs + kwargs_halos

        return names, redshifts, kwargs, kwargs_lenstronomy, convention_index

    def solve_lens_equation(self, lensmodel, kwargs_lens, precision_limit=10**-4,
                            arrival_time_sort=False):

        solver = LensEquationSolver(lensmodel)
        x_image, y_image = solver.findBrightImage(self.source_centroid_x, self.source_centroid_y, kwargs_lens,
                                                  search_window=4.,
                                                  precision_limit=precision_limit**2, arrival_time_sort=arrival_time_sort)
        return x_image, y_image


    @staticmethod
    def physical_location_deflector(lensmodel, kwargs, idx):

        kwargs_new = lensmodel.lens_model._convention(kwargs)
        return kwargs_new[idx]['center_x'], kwargs_new[idx]['center_y']

    def lensed_position_from_physical(self, lensmodel, kwargs, x_phys, y_phys, z_stop):

        lensed_x, lensed_y, _, _ = lensmodel.ray_shooting_partial(0., 0., x_phys, y_phys, 0, z_stop, kwargs)
        Tz = self.pyhalo_cosmology.D_C_transverse(z_stop)
        return lensed_x/Tz, lensed_y/Tz
