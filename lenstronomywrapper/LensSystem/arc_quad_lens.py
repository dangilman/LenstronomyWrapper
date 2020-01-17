from lenstronomy.LensModel.lens_model import LensModel
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization

class ArcQuadLensSystem(object):

    def __init__(self, macromodel, lens_light_model, source_light_model, z_source, substructure_realization=None, pyhalo_cosmology=None):

        self.macromodel = macromodel
        self.zlens = macromodel.zlens
        self.zsource = z_source

        self.lens_light_model = lens_light_model
        self.source_light_model = source_light_model

        self.update_realization(substructure_realization)
        if pyhalo_cosmology is None:
            # the default cosmology in pyHalo, currently WMAP9
            pyhalo_cosmology = Cosmology()

        self.astropy = pyhalo_cosmology.astropy

        self.update_source_centroid(0, 0)

        self.kpc_per_arcsec_zsource = self.astropy.arcsec_per_kpc_proper(z_source).value ** -1

    def initialize(self, data_to_fit, opt_routine='fixed_powerlaw_shear', constrain_params=None, verbose=False):

        optimizer = BruteOptimization(self)

        _, _, _ = optimizer.optimize(data_to_fit, opt_routine, constrain_params, verbose, include_substructure=False)

        return

    def update_lens_light(self, new_kwargs_lens_light):

        self.lens_light_model._kwargs = new_kwargs_lens_light

    def update_source_light(self, new_kwargs_source_light):

        self.source_light_model._kwargs = new_kwargs_source_light

    def update_realization(self, realization):

        self.substructure_realization = realization

    def update_kwargs_macro(self, new_kwargs):

        self.macromodel.update_kwargs(new_kwargs[0:self.macromodel.n_lens_models])

    def physical_location_deflector(self, idx):

        try:
            lensmodel, kwargs = self.get_fit_lensmodel()
        except:
            lensmodel, kwargs = self.get_lensmodel()

        kwargs_new = lensmodel.lens_model._convention(kwargs)
        return kwargs_new[idx]['center_x'], kwargs_new[idx]['center_y']

    def update_source_centroid(self, source_x, source_y):

        self.source_centroid_x = source_x
        self.source_centroid_y = source_y

    @property
    def realization(self):
        return self.substructure_realization

    def ray_shoot(self, xcoords, ycoords, lensModel=None, kwargs=None):

        if lensModel is None:
            lensModel, kwargs = self.get_lensmodel()

        betax, betay = lensModel.ray_shooting(xcoords, ycoords, kwargs)

        return betax, betay

    def get_fit_lensmodel(self):

        if not hasattr(self, '_fit_lens_model'):
            raise Exception('can only call get_fit_lensmodel after fitting a lens_system to data.')
        elif not hasattr(self, '_fit_kwargs'):
            raise Exception('can only call get_fit_lensmodel after fitting a lens_system to data.')

        return self._fit_lens_model, self._fit_kwargs

    def set_fit_lensmodel(self, lens_model, kwargs_lensmodel):

        self._fit_lens_model = lens_model
        self._fit_kwargs = kwargs_lensmodel

    def get_lens_light(self):

        instance, kwargs = self.lens_light_model.lensLight, self.lens_light_model.kwargs_light
        return instance, kwargs

    def get_source_light(self):

        instance, kwargs = self.source_light_model.sourceLight, self.source_light_model.kwargs_light
        return instance, kwargs

    def get_lensmodel(self):

        names, redshifts, kwargs, numercial_alpha_class, convention_index = self.get_lenstronomy_args()
        lensModel = LensModel(names, lens_redshift_list=redshifts, z_lens=self.zlens, z_source=self.zsource,
                              multi_plane=True, numerical_alpha_class=numercial_alpha_class,
                              observed_convention_index=convention_index, cosmo=self.astropy)
        return lensModel, kwargs

    def get_lenstronomy_args(self, include_substructure=True):

        lens_model_names, macro_redshifts, macro_kwargs, convention_index = self.macromodel.get_lenstronomy_args()
        realization = self.realization
        if realization is not None and include_substructure:
            halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = realization.lensing_quantities()
        else:
            halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = [], [], [], None
        halo_redshifts = list(halo_redshifts)
        names = lens_model_names + halo_names
        redshifts = macro_redshifts + halo_redshifts
        kwargs = macro_kwargs + kwargs_halos

        return names, redshifts, kwargs, kwargs_lenstronomy, convention_index

    def fit(self, data_to_fit, optimization_class, verbose=False):

        optimizer = optimization_class(self)
        kwargs_lens_final, lens_model_full, return_kwargs = optimizer.optimize(data_to_fit, verbose=verbose)

        return kwargs_lens_final, lens_model_full, return_kwargs

    def quasar_magnification(self, x, y, source_class, lens_model=None, kwargs_lensmodel=None):

        if lens_model is None or kwargs_lensmodel is None:
            lens_model, kwargs_lensmodel = self.get_fit_lensmodel()

        return source_class.magnification(x, y, lens_model, kwargs_lensmodel)


