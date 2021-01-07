from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
from lenstronomywrapper.LensSystem.lens_base import LensBase
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths_system
from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d

class QuadLensSystem(LensBase):

    def __init__(self, macromodel, z_source, substructure_realization=None, pyhalo_cosmology=None):

        """

        :param macromodel: an instance of MacroLensModel (see LensSystem.macrolensmodel)
        :param z_source: source redshift
        :param substructure_realization: an instance of Realization (see pyhalo.single_realization)
        :param pyhalo_cosmology: an instance of Cosmology() from pyhalo
        """

        if pyhalo_cosmology is None:
            # the default cosmology in pyHalo, currently WMAP9
            pyhalo_cosmology = Cosmology()

        self.pc_per_arcsec_zsource = 1000 * pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value ** -1

        super(QuadLensSystem, self).__init__(macromodel, z_source, substructure_realization, pyhalo_cosmology)

    @classmethod
    def fromArcQuadLens(cls, lens_system):

        macromodel = deepcopy(lens_system.macromodel)
        z_source = deepcopy(lens_system.zsource)
        substructure_realization, pyhalo_cosmology = deepcopy(lens_system.realization), \
                                                     deepcopy(lens_system.pyhalo_cosmology)

        system = QuadLensSystem(macromodel, z_source, substructure_realization,
                                pyhalo_cosmology)

        return system

    @classmethod
    def shift_background_auto(cls, lens_data_class, macromodel, zsource,
                              realization, cosmo=None, particle_swarm_init=False,
                              opt_routine='free_shear_powerlaw', constrain_params=None, verbose=False,
                              centroid_convention='IMAGES'):

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
        :param substructure_realization: an instance of Realization (see pyhalo.single_realization)
        :param cosmo: an instance of Cosmology() from pyhalo
        :param particle_swarm_init: whether or not to use a particle swarm algorithm when fitting the macromodel.
        You should use a particle swarm routine if you're starting the lens model from scratch
        :param opt_routine: the optimization routine to use... more documentation coming soon
        :param constrain_params: keywords to be passed to optimization routine
        :param verbose: whether to print stuff
        :param centroid_convention: the definition of the lens cone "center". There are two options:
        'IMAGES' - rendering area is taken to be the mean of the image coordinate at each lens plane
        'DEFLECTOR' - rendering area is computed by performing a ray tracing computation through the deflector mass
        centroid

        This routine can be immediately followed by doing a lens model fit to the data, for example:

        1st:
        lens_system = QuadLensSystem.shift_background_auto(data, macromodel, zsource,
                            realization, particle_swarm_init=True)

        2nd:
        lens_system.initialize(data, include_substructure=True)
        # will fit the lens system while including every single halo in the computation

        Other optimization routines are detailed in Optimization.quad_optimization

        """

        lens_system_init = QuadLensSystem(macromodel, zsource, None,
                                          pyhalo_cosmology=cosmo)

        lens_system_init.initialize(lens_data_class, opt_routine=opt_routine, constrain_params=constrain_params,
                                    kwargs_optimizer={'particle_swarm': particle_swarm_init}, verbose=verbose)

        source_x, source_y = lens_system_init.source_centroid_x, lens_system_init.source_centroid_y

        assert centroid_convention in ['IMAGES', 'DEFLECTOR']
        ray_interp_x, ray_interp_y = interpolate_ray_paths_system(
            lens_data_class.x, lens_data_class.y, lens_system_init,
            include_substructure=False, terminate_at_source=True, source_x=source_x,
            source_y=source_y)

        if centroid_convention == 'IMAGES':

            ### Now compute the centroid of the light cone as the coordinate centroid of the individual images
            z_range = np.linspace(0, lens_system_init.zsource, 100)
            distances = [lens_system_init.pyhalo_cosmology.D_C_transverse(zi) for zi in z_range]
            angular_coordinates_x = []
            angular_coordinates_y = []
            for di in distances:
                x_coords = [ray_x(di) for i, ray_x in enumerate(ray_interp_x)]
                y_coords = [ray_y(di) for i, ray_y in enumerate(ray_interp_y)]
                x_center = np.mean(x_coords)
                y_center = np.mean(y_coords)
                angular_coordinates_x.append(x_center)
                angular_coordinates_y.append(y_center)

            ray_interp_x = [interp1d(distances, angular_coordinates_x)]
            ray_interp_y = [interp1d(distances, angular_coordinates_y)]

        realization = realization.shift_background_to_source(ray_interp_x[0], ray_interp_y[0])

        macromodel = lens_system_init.macromodel

        lens_system = QuadLensSystem(macromodel, zsource,
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
        pyhalo_cosmo = quad_lens_system.pyhalo_cosmology
        new_quad = QuadLensSystem(macromodel, z_source, realization, pyhalo_cosmo)
        if hasattr(quad_lens_system, 'source_centroid_x'):
            source_x, source_y = quad_lens_system.source_centroid_x, quad_lens_system.source_centroid_y
            new_quad.update_source_centroid(source_x, source_y)
        return new_quad

    def get_smooth_lens_system(self):

        """
        Returns a lens system with only the smooth component of the lens model (i.e. no substructure)
        """
        smooth_lens = QuadLensSystem(self.macromodel, self.zsource,
                                                  None, self.pyhalo_cosmology)

        if hasattr(self, 'source_centroid_x'):
            smooth_lens.update_source_centroid(self.source_centroid_x, self.source_centroid_y)

        return smooth_lens

    def initialize(self, data_to_fit, opt_routine='free_shear_powerlaw',
                   constrain_params=None, verbose=False,
                   include_substructure=False, kwargs_optimizer={}):

        """
        This routine fits a smooth macromodel profile defined by self.macromodel to the image positions in data_to_fit
        :param data_to_fit: an instanced of LensedQuasar (see LensSystem.BackgroundSource.lensed_quasar)

        """

        optimizer = BruteOptimization(self)
        kwargs_lens_final, lens_model_full, _ = optimizer.optimize(
            data_to_fit, opt_routine, constrain_params, verbose,
            include_substructure, kwargs_optimizer
        )

        return

    def update_source_centroid(self, source_x, source_y):

        self.source_centroid_x = source_x
        self.source_centroid_y = source_y

    def quasar_magnification(self, x, y, background_source,
                             lens_model,
                             kwargs_lensmodel, normed=True,
                             retry_if_blended=0,
                             enforce_unblended=False,
                             adaptive=False, verbose=False, point_source=False,
                             grid_axis_ratio=1):

        """
        Computes the magnifications (or flux ratios if normed=True)

        :param x: x image position
        :param y: y image position
        :param background_quasar: an instance of the background source light profile
        :param lens_model: an instance of LensModel (see lenstronomy.lens_model)
        :param kwargs_lensmodel: key word arguments for the lens_model
        :param normed: if True returns flux ratios
        :param retry_if_blended: a integer that specifies how many times to try
        increasing the size of the ray tracing window if an image comes out blended together
        :param point_source: if True, computes the magnification of a point source
        """

        if point_source:
            mags = lens_model.magnification(x, y, kwargs_lensmodel)
            mags = abs(mags)
            if normed:
                mags *= max(mags) ** -1
            return mags, False

        background_source.setup(self.pc_per_arcsec_zsource)
        if not hasattr(self, 'source_centroid_x') or self.source_centroid_x is None:
            raise Exception('lens system must have a specified source coordinate in order to compute the magnification'
                            'from an extended source.')
        background_source.update_position(self.source_centroid_x, self.source_centroid_y)
        relative_angles = [np.arctan2(-xi, yi) for (xi, yi) in zip(x, y)]

        return background_source.magnification(x, y, lens_model,
                                                    kwargs_lensmodel,
                                                    normed, retry_if_blended,
                                                    enforce_unblended, adaptive, verbose,
                                                    grid_axis_ratio,
                                                    relative_angles)

    def plot_images(self, x, y, background_source, lens_model=None, kwargs_lensmodel=None,
                    adaptive=False):

        if lens_model is None or kwargs_lensmodel is None:
            if self._static_lensmodel:
                lens_model, kwargs_lensmodel = self._lensmodel_static, self._kwargs_static
            else:
                raise Exception('must either specify the LensModel class instance and keywords,'
                                'or have a precomputed static lens model instance saved in this class.')

        background_source.setup(self.pc_per_arcsec_zsource)
        if not hasattr(self, 'source_centroid_x') or self.source_centroid_x is None:
            raise Exception('lens system must have a specified source coordinate in order to compute the magnification'
                            'from an extended source.')
        background_source.update_position(self.source_centroid_x, self.source_centroid_y)

        return background_source.plot_images(x, y, lens_model, kwargs_lensmodel,
                                                      adaptive=adaptive)
