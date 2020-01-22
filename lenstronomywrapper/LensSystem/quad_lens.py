from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
from lenstronomywrapper.LensSystem.lens_base import LensBase
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar

class QuadLensSystem(LensBase):

    def __init__(self, macromodel, z_source, background_quasar_class, substructure_realization=None, pyhalo_cosmology=None):

        if pyhalo_cosmology is None:
            # the default cosmology in pyHalo, currently WMAP9
            pyhalo_cosmology = Cosmology()

        if background_quasar_class is None:
            kwargs_default = {'center_x': 0, 'center_y': 0, 'source_fwhm_pc': 30.}
            background_quasar_class = Quasar(kwargs_default)

        self.background_quasar = background_quasar_class
        pc_per_arcsec_zsource = 1000 * pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value ** -1
        self.background_quasar.setup(pc_per_arcsec_zsource)

        super(QuadLensSystem, self).__init__(macromodel, z_source, substructure_realization, pyhalo_cosmology)

    def initialize(self, data_to_fit, opt_routine='fixed_powerlaw_shear', constrain_params=None, verbose=False,
                   include_substructure=False, kwargs_optimizer={}):

        optimizer = BruteOptimization(self)

        kwargs_lens_final, lens_model_full, _ = optimizer.optimize(data_to_fit, opt_routine, constrain_params,
                                                                   verbose, False, kwargs_optimizer)

        if include_substructure and self.realization is not None:

            realization = self.realization

            realization.shift_background_to_source(self.source_centroid_x, self.source_centroid_y)

            optimizer = BruteOptimization(self, reoptimize=True)

            kwargs_lens_final, lens_model_full, _ = optimizer.optimize(data_to_fit, opt_routine, constrain_params,
                                                                       verbose, True, kwargs_optimizer)

        return

    def update_source_centroid(self, source_x, source_y):

        self.source_centroid_x = source_x
        self.source_centroid_y = source_y
        self.background_quasar.update_position(source_x, source_y)

    def quasar_magnification(self, x, y, lens_model=None, kwargs_lensmodel=None, normed=True):

        if lens_model is None or kwargs_lensmodel is None:
            lens_model, kwargs_lensmodel = self.get_lensmodel()

        return self.background_quasar.magnification(x, y, lens_model, kwargs_lensmodel, normed)

    def plot_images(self, x, y, lens_model=None, kwargs_lensmodel=None):

        if lens_model is None or kwargs_lensmodel is None:
            lens_model, kwargs_lensmodel = self.get_lensmodel()

        return self.background_quasar.plot_images(x, y, lens_model, kwargs_lensmodel)



