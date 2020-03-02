from lenstronomywrapper.Optimization.quad_optimization.optimization_base import OptimizationBase
from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
from lenstronomywrapper.Utilities.data_util import image_separation_vectors_quad

class DynamicOptimization(OptimizationBase):

    def __init__(self, lens_system, pyhalo_dynamic, kwargs_rendering, global_log_mlow,
                 log_mass_cuts, aperture_sizes, refit, aperture_setting,
                 particle_swarm, re_optimize, realization_type,
                 n_particles=35, simplex_n_iter=300, reoptimize=False,
                 ):

        assert len(log_mass_cuts) == len(aperture_sizes)
        assert len(refit) == len(log_mass_cuts)

        self.log_mass_cuts = log_mass_cuts
        self.aperture_sizes = aperture_sizes
        self.aperture_setting = aperture_setting
        self.refit = refit
        self.ps = particle_swarm
        self.re_optimize = re_optimize
        self.kwargs_rendering = kwargs_rendering
        self.realization_type = realization_type

        self.n_particles = n_particles
        self.n_iterations = simplex_n_iter
        self.reoptimize = reoptimize

        self._global_log_mlow = global_log_mlow

        self.pyhalo_dynamic = pyhalo_dynamic

        self._brute = BruteOptimization(lens_system, n_particles, simplex_n_iter, log_mass_sheet_front=global_log_mlow,
                                        log_mass_sheet_back=global_log_mlow)

        super(DynamicOptimization, self).__init__(lens_system)

    def _setup(self, verbose):

        if verbose: print('initializing with log(mlow) = ' + str(self._global_log_mlow) + '.... ')
        macro_lens_model_init, kwargs_macro_init = self.lens_system.get_lensmodel(include_substructure=False)

        self.pyhalo_dynamic.set_macro_lensmodel(macro_lens_model_init, kwargs_macro_init, self._global_log_mlow)

        realization_global = self.pyhalo_dynamic.render(self.realization_type,
                                                        self.kwargs_rendering, verbose=verbose)[0]

        return realization_global, self._global_log_mlow

    def _auto_aperture_size(self, data_to_fit):

        raise Exception('not yet implemented')

    def optimize(self, data_to_fit, opt_routine='fixed_powerlaw_shear', constrain_params=None, verbose=False):

        self.lens_system.initialize(data_to_fit, opt_routine, constrain_params, verbose,
                   include_substructure=False)

        realization_global, log_mhigh = self._setup(verbose)

        if verbose:
            print('fitting with log(mlow) = ' + str(self._global_log_mlow) + '.... ')
            print('n foreground halos: ',
                      realization_global.number_of_halos_before_redshift(self.lens_system.zlens))
            print('n background halos: ', realization_global.number_of_halos_after_redshift(self.lens_system.zlens))

        kwargs_lens_final, _, lens_model_full, _, _, source = self._brute.fit(data_to_fit, 30, opt_routine, constrain_params,
                                                                     self.n_iterations, {}, verbose, re_optimize=False,
                                                                     particle_swarm=True, realization=realization_global)

        self.lens_system.clear_static_lensmodel()
        self.lens_system.update_kwargs_macro(kwargs_lens_final)
        self.lens_system.update_realization(realization_global)

        lens_model, kwargs_macro = self.lens_system.get_lensmodel()

        if self.aperture_setting == 'images':
            aperture_center_x, aperture_center_y = data_to_fit.x, data_to_fit.y

        else:
            aperture_center_x, aperture_center_y = self.aperture_setting['center_x'], self.aperture_setting['center_y']

        for (log_m_low, aperture_size, fit, particle_swarm, re_optimize) in zip(self.log_mass_cuts, self.aperture_sizes,
                                                                   self.refit, self.ps, self.re_optimize):

            if verbose:
                print('log_mlow:', log_m_low)
                print('log_mhigh:', log_mhigh)
                print('aperture_size:', aperture_size)
                print('fitting:', fit)

            assert log_m_low < log_mhigh
            realization_global = self.pyhalo_dynamic.render_dynamic(self.realization_type, self.kwargs_rendering,
                       aperture_center_x, aperture_center_y, aperture_size, log_m_low, log_mhigh, lens_model, kwargs_macro,
                                                                    realization_global, verbose)

            if verbose:
                print('n foreground halos: ',
                      realization_global.number_of_halos_before_redshift(self.lens_system.zlens))
                print('n background halos: ', realization_global.number_of_halos_after_redshift(self.lens_system.zlens))

            if fit:

                kwargs_lens_final, _, lens_model_full, _, _, source = self._brute.fit(data_to_fit, 30, opt_routine,
                                                                             constrain_params,
                                                                             self.n_iterations, {}, verbose, True,
                                                                             particle_swarm=True,
                                                                             realization=realization_global)
                self.lens_system.clear_static_lensmodel()
                self.lens_system.update_kwargs_macro(kwargs_lens_final)
                self.lens_system.update_realization(realization_global)
                lens_model, kwargs_macro = self.lens_system.get_lensmodel()

            else:

                self.lens_system.update_realization(realization_global)

                lens_model, kwargs_lens_final = self.lens_system.get_lensmodel()

            log_mhigh = log_m_low

        return self._return_results(source, kwargs_lens_final, lens_model, {'realization_final': realization_global})
