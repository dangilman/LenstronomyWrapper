from lenstronomywrapper.Optimization.quad_optimization.optimization_base import OptimizationBase
from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths
from copy import deepcopy

class DynamicOptimization(OptimizationBase):

    def __init__(self, lens_system, kwargs_rendering, global_log_mlow,
                 log_mass_cuts, aperture_sizes, refit,
                 particle_swarm, re_optimize, realization_type,
                 n_particles=35, simplex_n_iter=300, initial_pso=True):

        """
        :param lens_system: an instance of QuadLensSystem (see documentation in LensSystem.quad_lens)
        :param kwargs_rendering: key word arguments for the realization, see documentation in pyHalo
        :param global_log_mlow: the lowest-mass halos to render everywhere in the lens system
        Everything with mass below 10**global_log_mlow will be rendered iteratively around images in
        progressively smaller apertures defined by the next two arguments

        :param log_mass_cuts: list of floats corresponding to minimum halo masses to render
        :param aperture_sizes: list of floats corresponding to aperture size in which to render halos

        # The following control how lenstronomy goes about fitting the lens.
        :param refit: list of bool telling the code whether or not to re-fit a lens model with the new halos.
            - It may not be necessary to re-fit the lens model if the number of new halos added is small
            and/or they are not very massive

        :param particle_swarm: list of bools telling the optimizer routine (lenstronomy->Optimizer)
        whether or not to perform a particle swarm optimization
            - particle swarm algorithms are good at finding the minimum of a large-dimension parameter space
            - if particle_swarm = False, then refit key word is irrelvant

        :param re_optimize: list of bools telling the optimizer routine (lenstronomy->Optimizer) whether
        or not to treat each new re-fit of a lensmodel as a reoptimization
            - re-optimizing is fast if you're starting from a lens model you know is close to the 'truth'
            - Note: this key word only has an effect if you're doing a particle swarm optimization

        :param realization_type: specifies the kind of realization to generate (see documentation in pyHalo)
            - examples are 'composite_powerlaw', 'line_of_sight', 'main_lens'

        ###########################################
        EXAMPLE INPUT FOR A CDM MASS FUNCTION:

        minimum_mass_global = 8. # render 10^8 everywhere in the lensing volume
        log_mass_cuts_list = [7., 6]
        aperture_sizes_list = [0.2, 0.1]
        # add 10^7 halos in 0.2 arcsec apertures around each image along the ray path
        # add 10^6 halos in 0.1 arcsec apertures around each image along the ray path

        refit_list = [True, False]
        # re-optimize the lens model after adding 10^7 halos, but not after 10^6 because their
        # deflection angles are tiny
        particle_swarm_list = [True, False]
        # don't really need a particle swarm for low-mass halos
        (code will automatically do a particle swarm with the largest halos specified by minimum_mass_global)
        re_optimize_list = [True, False]
        #
        realization_type = 'composite_powerlaw'
        ###########################################

        ###########################################
        EXAMPLE INPUT FOR A DELTA FUNCTION MASS FUNCTION:

        log_mass_cuts_list = [delta_function_mass]
        aperture_sizes_list = [0.1] #arcsec
        refit_list = [True]
        particle_swarm_list = [False]
        re_optimize_list = [False]
        realization_type = 'line_of_sight'
        ###########################################
        """
        assert len(log_mass_cuts) == len(aperture_sizes)
        assert len(refit) == len(log_mass_cuts)

        # instantiate pyhaloDynamic
        self.pyhalo_dynamic = pyHaloDynamic(lens_system.zlens, lens_system.zsource)

        if len(log_mass_cuts) > 0:
            assert global_log_mlow > log_mass_cuts[0]

            for i in range(0, len(log_mass_cuts)-1):
                if log_mass_cuts[i] <= log_mass_cuts[i+1]:
                    raise Exception('Aperture masses must be decreasing, you provided: ' + str(log_mass_cuts))
            for i in range(0, len(log_mass_cuts)-1):
                if aperture_sizes[i] < aperture_sizes[i+1]:
                    print('WARNING: Aperture sizes are not monotically decreasing functions of halo mass.'
                          'This is probably not what you want to do....')

        self.log_mass_cuts = log_mass_cuts
        self.aperture_sizes = aperture_sizes
        self.refit = refit
        self.ps = particle_swarm
        self.re_optimize = re_optimize
        self.kwargs_rendering = deepcopy(kwargs_rendering)
        self.realization_type = realization_type

        self.n_particles = n_particles
        self.n_iterations = simplex_n_iter
        self.initial_pso = initial_pso

        self._global_log_mlow = global_log_mlow

        self.simplex_n_iter = simplex_n_iter

        super(DynamicOptimization, self).__init__(lens_system)

    def optimize(self, data_to_fit, opt_routine='free_shear_powerlaw',
                 constrain_params=None, verbose=False):

        """

        This class fits a lens model to four lensed image positions in the presence of dark matter halos.

        The algorithm adds halos dynamically, meaning that they are rendered around paths traversed by the
        light, in contrast to rendering the halos all at once everywhere.

        :param data_to_fit: instance of LensedQuasar data class (see LensData.lensed_quasar)
        :param opt_routine: optimization routine, can be fixed_powerlaw_shear or fixedshearpowerlaw
        :param constrain_params: penalize parameters if they don't have certain values
        (see documentation in lenstronomy.Optimizer)
        :param verbose: print things
        :return: optimized lens model and keyword arguments
        """

        brute = BruteOptimization(self.lens_system, self.n_particles, self.simplex_n_iter)

        # Fit a smooth model (macromodel + satellites) to the image positions

        kwargs_optimizer = {'particle_swarm': self.initial_pso}
        self.lens_system.initialize(data_to_fit, opt_routine, constrain_params,
                                    include_substructure=False, kwargs_optimizer=kwargs_optimizer,
                                    verbose=verbose)

        # set up initial realization with large halos generated everywhere
        realization_global, log_mhigh = self._initialize(verbose, data_to_fit)

        if verbose:
            print('fitting with log(mlow) = ' + str(self._global_log_mlow) + '.... ')
            print('n foreground halos: ',
                      realization_global.number_of_halos_before_redshift(self.lens_system.zlens))
            print('n subhalos: ',
                  realization_global.number_of_halos_at_redshift(self.lens_system.zlens))
            print('n background halos: ', realization_global.number_of_halos_after_redshift(self.lens_system.zlens))

        # Add the large halos, fit a lens model
        kwargs_lens_final, lens_model_full, source = brute.fit(
            data_to_fit, opt_routine, constrain_params=constrain_params, verbose=verbose,
                 include_substructure=True, realization=realization_global, re_optimize=False,
                 particle_swarm=True, pso_convergence_mean=150000)
        self.update_lens_system(source, kwargs_lens_final, lens_model_full, realization_global)

        if isinstance(self.kwargs_rendering, list):
            lens_plane_redshifts, _ = self.pyhalo_dynamic.lens_plane_redshifts(self.kwargs_rendering[0])
        else:
            lens_plane_redshifts, _ = self.pyhalo_dynamic.lens_plane_redshifts(self.kwargs_rendering)

        # Iterate through lower masses and smaller progressively smaller rendering apertures
        for (log_mlow, aperture_size, fit, particle_swarm, re_optimize) in zip(self.log_mass_cuts, self.aperture_sizes,
                                                                   self.refit, self.ps, self.re_optimize):

            # iteratively add lower mass halos in progressively smaller apertures around the lensed light rays

            assert log_mlow < log_mhigh

            x_interp_list, y_interp_list = self._get_interp(data_to_fit.x, data_to_fit.y,
                                                                          lens_plane_redshifts,
                                                            terminate_at_source=False)

            lens_centroid_x, lens_centroid_y = self.lens_system.macromodel.centroid

            if isinstance(self.kwargs_rendering, list):
                for i in range(0, len(self.kwargs_rendering)):
                    self.kwargs_rendering[i]['log_mlow'], self.kwargs_rendering[i]['log_mhigh'] = log_mlow, log_mhigh
            else:
                self.kwargs_rendering['log_mlow'], self.kwargs_rendering['log_mhigh'] = log_mlow, log_mhigh

            realization_global = self.pyhalo_dynamic.render_dynamic(self.realization_type, self.kwargs_rendering,
                       realization_global, lens_centroid_x, lens_centroid_y, x_interp_list, y_interp_list, aperture_size,
                       verbose, global_render=False)

            if verbose:
                print('log_mlow:', log_mlow)
                print('log_mhigh:', log_mhigh)
                print('aperture_size:', aperture_size)
                print('fitting:', fit)
                print('n foreground halos: ',
                      realization_global.number_of_halos_before_redshift(self.lens_system.zlens))
                print('n subhalos: ',
                      realization_global.number_of_halos_at_redshift(self.lens_system.zlens))
                print('n background halos: ', realization_global.number_of_halos_after_redshift(self.lens_system.zlens))

            if fit:

                brute = BruteOptimization(self.lens_system, self.n_particles, self.simplex_n_iter)
                kwargs_lens_final, lens_model_full, source = brute.fit(
                    data_to_fit, opt_routine, constrain_params=constrain_params, verbose=verbose,
                    include_substructure=True, realization=realization_global, re_optimize=re_optimize,
                    particle_swarm=particle_swarm, pso_convergence_mean=150000)
                self.update_lens_system(source, kwargs_lens_final, lens_model_full, realization_global)

            else:

                self.lens_system.clear_static_lensmodel()
                self.lens_system.update_realization(realization_global)
                lens_model_full, kwargs_lens_final = self.lens_system.get_lensmodel()
                self.update_lens_system(source, kwargs_lens_final, lens_model_full, realization_global)

            log_mhigh = log_mlow

        self.pyhalo_dynamic.reset(self.lens_system.zlens, self.lens_system.zsource)

        return self.return_results(source, kwargs_lens_final, lens_model_full, realization_global,
                                   {'realization_final': realization_global})

    def _initialize(self, verbose, data_to_fit):

        if verbose: print('initializing with log(mlow) = ' + str(self._global_log_mlow) + '.... ')

        macro_lens_model, kwargs_macro = self.lens_system.get_lensmodel(include_substructure=False)

        source_x, source_y = self.lens_system.source_centroid_x, self.lens_system.source_centroid_y
        x_interp_list, y_interp_list = self.pyhalo_dynamic.interpolate_ray_paths(data_to_fit.x, data_to_fit.y,
                                    lens_model=macro_lens_model, kwargs_lens=kwargs_macro, zsource=self.lens_system.zsource,
                                             terminate_at_source=True, source_x=source_x, source_y=source_y,
                                                                                 evaluate_at_mean=True)

        if isinstance(self.kwargs_rendering, list):
            print('generating global realization with first set of keywords in kwargs_rendering...')
            kwargs_init = deepcopy(self.kwargs_rendering[0])
        else:
            kwargs_init = deepcopy(self.kwargs_rendering)

        if 'log_mlow' in kwargs_init.keys():
            kwargs_init['log_mlow_subs'] = kwargs_init['log_mlow']
            kwargs_init['log_mhigh_subs'] = kwargs_init['log_mhigh']

        kwargs_init['log_mlow'] = self._global_log_mlow
        if 'log_mhigh' not in kwargs_init.keys():
            kwargs_init['log_mhigh'] = None

        aperture_radius = 0.5 * kwargs_init['cone_opening_angle']

        realization_global = self.pyhalo_dynamic.render_dynamic(self.realization_type, kwargs_init,
                                                                None, lens_centroid_x, lens_centroid_y,
                                                                x_interp_list, y_interp_list, aperture_radius,
                                                                verbose, include_mass_sheet_correction=True,
                                                                global_render=True)

        return realization_global, self._global_log_mlow

    def _get_interp(self, x_coords, y_coords, lens_plane_redshifts, terminate_at_source):

        lens_model, kwargs = self.lens_system.get_lensmodel()

        lens_model_list, lens_plane_redshifts, convention_index = \
            self.lenstronomy_args_from_lensmodel(lens_model)

        lensmodel_new = LensModel(lens_model_list, z_lens=self.pyhalo_dynamic.zlens,
                                  z_source=self.pyhalo_dynamic.zsource, lens_redshift_list=lens_plane_redshifts,
                                  cosmo=lens_model.cosmo, multi_plane=True,
                                  numerical_alpha_class=self.lens_system._numerical_alpha_class)

        source_x, source_y = self.lens_system.source_centroid_x, self.lens_system.source_centroid_y

        x_interp, y_interp = interpolate_ray_paths(x_coords, y_coords,
                              lensmodel_new, kwargs, self.pyhalo_dynamic.zsource,
                                                   terminate_at_source=terminate_at_source,
                                                   source_x=source_x, source_y=source_y)

        return x_interp, y_interp

    @staticmethod
    def lenstronomy_args_from_lensmodel(lensmodel):

        lens_model_list = lensmodel.lens_model_list
        redshift_list = lensmodel.redshift_list
        convention_index = lensmodel.lens_model._observed_convention_index
        return lens_model_list, redshift_list, convention_index
