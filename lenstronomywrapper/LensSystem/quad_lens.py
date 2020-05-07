from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
from lenstronomywrapper.LensSystem.lens_base import LensBase
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths_system

class QuadLensSystem(LensBase):

    def __init__(self, macromodel, z_source, background_quasar_class,
                 substructure_realization=None, pyhalo_cosmology=None):

        """

        :param macromodel: an instance of MacroLensModel (see LensSystem.macrolensmodel)
        :param z_source: source redshift
        :param background_quasar_class: an instance of quasar (see LensSystem.BackgroundSource.quasar)
        :param substructure_realization: an instance of Realization (see pyhalo.single_realization)
        :param pyhalo_cosmology: an instance of Cosmology() from pyhalo
        """

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

    @classmethod
    def shift_background_auto(cls, lens_data_class, macromodel, zsource,
                              background_quasar, realization, cosmo=None, particle_swarm_init=False):

        """
        This method takes a macromodel and a substructure realization, fits a smooth model to the data
        with the macromodel only, and then shifts the halos in the realization such that they lie along
        a path traversed by the light. For simple 1-deflector lens models this path is close to a straight line
        between the observer and the source, but for more complicated lens models with satellite galaxies and
        LOS galaxies the path can be complicated and it is important to shift the line of sight halos;
        often, when line of sight galaxies are included, the source is not even close to being
        behind directly main deflector.

        :param macromodel: an instance of MacroLensModel (see LensSystem.macrolensmodel)
        :param z_source: source redshift
        :param background_quasar_class: an instance of quasar (see LensSystem.BackgroundSource.quasar)
        :param substructure_realization: an instance of Realization (see pyhalo.single_realization)
        :param cosmo: an instance of Cosmology() from pyhalo
        :param particle_swarm_init: whether or not to use a particle swarm algorithm when fitting the macromodel.
        You should use a particle swarm routine if you're starting the lens model from scratch

        This routine can be immediately followed by doing a lens model fit to the data, for example:

        1st:
        lens_system = QuadLensSystem.shift_background_auto(data, macromodel, zsource, background_quasar,
                            realization, particle_swarm_init=True)

        2nd:
        lens_system.initialize(data, include_substructure=True)
        # will fit the lens system while including every single halo in the computation

        More efficient optimization routines are detailed in Optimization.quad_optimization

        """
        lens_system_init = QuadLensSystem(macromodel, zsource, background_quasar, None,
                                          pyhalo_cosmology=cosmo)

        lens_system_init.initialize(lens_data_class, kwargs_optimizer={'particle_swarm': particle_swarm_init})

        source_x, source_y = lens_system_init.source_centroid_x, lens_system_init.source_centroid_y
        lens_center_x, lens_center_y = lens_system_init.macromodel.centroid

        ray_interp_x, ray_interp_y = interpolate_ray_paths_system(
            [lens_center_x], [lens_center_y], lens_system_init,
            include_substructure=False, terminate_at_source=True, source_x=source_x,
            source_y=source_y)

        realization = realization.shift_background_to_source(ray_interp_x[0], ray_interp_y[0])

        macromodel = lens_system_init.macromodel
        background_quasar = lens_system_init.background_quasar

        lens_system = QuadLensSystem(macromodel, zsource, background_quasar,
                                          realization, lens_system_init.pyhalo_cosmology)

        lens_system.update_source_centroid(source_x, source_y)

        return lens_system

    @classmethod
    def addRealization(cls, quad_lens_system, realization):

        """
        This routine creates a new instance of QuadLensSystem that is identical to quad_lens_system,
        but includes a different substructure realization
        """
        macromodel = quad_lens_system.macromodel
        z_source = quad_lens_system.zsource
        background_quasar = quad_lens_system.background_quasar
        pyhalo_cosmo = quad_lens_system.pyhalo_cosmology
        new_quad = QuadLensSystem(macromodel, z_source, background_quasar, realization, pyhalo_cosmo)
        source_x, source_y = quad_lens_system.source_centroid_x, quad_lens_system.source_centroid_y
        new_quad.update_source_centroid(source_x, source_y)
        return new_quad

    def get_smooth_lens_system(self):

        smooth_lens = QuadLensSystem(self.macromodel, self.zsource, self.background_quasar,
                                                  None, self.pyhalo_cosmology)

        smooth_lens.update_source_centroid(self.source_centroid_y, self.source_centroid_y)

        return smooth_lens

    def initialize(self, data_to_fit, opt_routine='fixed_powerlaw_shear', constrain_params=None, verbose=False,
                   include_substructure=False, kwargs_optimizer={}):

        """
        This routine fits a smooth macromodel profile defined by self.macromodel to the image positions in data_to_fit
        :param data_to_fit: an instanced of LensedQuasar (see LensSystem.BackgroundSource.lensed_quasar)

        """

        optimizer = BruteOptimization(self)
        kwargs_lens_final, lens_model_full, _ = optimizer.optimize(data_to_fit, opt_routine, constrain_params,
                                                                       verbose, include_substructure, kwargs_optimizer,
                                                                       )

        return

    def update_source_centroid(self, source_x, source_y):

        self.source_centroid_x = source_x
        self.source_centroid_y = source_y
        self.background_quasar.update_position(source_x, source_y)

    def quasar_magnification(self, x, y, lens_model=None, kwargs_lensmodel=None, normed=True):

        """
        Computes the magnifications (or flux ratios if normed=True)

        :param x: x image position
        :param y: y image position
        :param lens_model: an instance of LensModel (see lenstronomy.lens_model)
        :param kwargs_lensmodel: key word arguments for the lens_model
        :param normed: if True returns flux ratios
        """
        if lens_model is None or kwargs_lensmodel is None:
            lens_model, kwargs_lensmodel = self.get_lensmodel()

        return self.background_quasar.magnification(x, y, lens_model, kwargs_lensmodel, normed)

    def plot_images(self, x, y, lens_model=None, kwargs_lensmodel=None):

        if lens_model is None or kwargs_lensmodel is None:
            if self._static_lensmodel:
                lens_model, kwargs_lensmodel = self._lensmodel_static, self._kwargs_static
            else:
                raise Exception('must either specify the LensModel class instance and keywords,'
                                'or have a precomputed static lens model instance saved in this class.')
        return self.background_quasar.plot_images(x, y, lens_model, kwargs_lensmodel)



