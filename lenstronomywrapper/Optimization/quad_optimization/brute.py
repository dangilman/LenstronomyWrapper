from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
from lenstronomywrapper.Optimization.quad_optimization.optimization_base import OptimizationBase
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths

class BruteOptimization(OptimizationBase):

    def __init__(self, lens_system, n_particles=None, simplex_n_iter=None, reoptimize=None,
                 log_mass_sheet_front=None, log_mass_sheet_back=None):

        settings = BruteSettingsDefault()

        if n_particles is None:
            n_particles = settings.n_particles
        if simplex_n_iter is None:
            n_iterations = settings.n_iterations
        if reoptimize is None:
            reoptimize = settings.reoptimize

        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.reoptimize = reoptimize

        self._log_mass_sheet_front = log_mass_sheet_front
        self._log_mass_sheet_back = log_mass_sheet_back

        # shoot a ray through the center to determine any global shifts in path
        ray_interp_x, ray_interp_y = interpolate_ray_paths([0.], [0.], lens_system,
                                                           include_substructure=False, realization=None)

        self.realization_initial = lens_system.realization.shift_background_to_source(ray_interp_x[0],
                                                                              ray_interp_y[0])

        super(BruteOptimization, self).__init__(lens_system)

    def optimize(self, data_to_fit, opt_routine='fixed_powerlaw_shear', constrain_params=None, verbose=False,
                 include_substructure=True, kwargs_optimizer={}):

        self._check_routine(opt_routine, constrain_params)

        kwargs_lens_final, _, lens_model_full, _, images, source = self._fit(data_to_fit, self.n_particles, opt_routine,
                                  constrain_params, self.n_iterations, {}, verbose,
                                      re_optimize=self.reoptimize, tol_mag=None,
                                          include_substructure=include_substructure,
                                        kwargs_optimizer=kwargs_optimizer, realization=self.realization_initial)

        return_kwargs = {'info_array': None,
                         'lens_model_raytracing': lens_model_full,
                         'realization_final': self.realization_initial}

        return self._return_results(source, kwargs_lens_final, lens_model_full, return_kwargs)

    def _fit(self, data_to_fit, nparticles, opt_routine, constrain_params, simplex_n_iter, optimizer_kwargs, verbose,
                            re_optimize=False, tol_mag=None, include_substructure=True,
                                            kwargs_optimizer={}, realization=None):

        """
        run_kwargs: {'optimizer_routine', 'constrain_params', 'simplex_n_iter'}
        filter_kwargs: {'re_optimize', 'particle_swarm'}
        """

        lens_model_list, redshift_list, kwargs_lens, numerical_alpha_class, convention_index = \
            self.lens_system.get_lenstronomy_args(include_substructure, realization=realization,
                                                  log_mass_sheet_front=self._log_mass_sheet_front,
                                                  log_mass_sheet_back=self._log_mass_sheet_back)

        run_kwargs = {'optimizer_routine': opt_routine, 'constrain_params': constrain_params,
                      'simplex_n_iterations': simplex_n_iter,
                      're_optimize': re_optimize, 'tol_mag': tol_mag, 'multiplane': True,
                      'z_main': self.lens_system.zlens, 'z_source': self.lens_system.zsource,
                      'astropy_instance': self.lens_system.astropy, 'verbose': verbose, 'pso_convergence_mean': 30000,
                      'observed_convention_index': convention_index, 'optimizer_kwargs': optimizer_kwargs,
                      }

        for key in kwargs_optimizer.keys():
            run_kwargs[key] = kwargs_optimizer[key]

        opt = Optimizer(data_to_fit.x, data_to_fit.y, redshift_list, lens_model_list, kwargs_lens, numerical_alpha_class,
                 magnification_target=data_to_fit.m, **run_kwargs)

        kwargs_lens_final, [source_x, source_y], [x_image, y_image] = opt.optimize(nparticles)
        lens_model_raytracing = opt.lensModel
        lens_model_full = opt._lensModel
        foreground_rays = opt.lensModel._foreground._rays

        return kwargs_lens_final, lens_model_raytracing, lens_model_full, foreground_rays, [x_image, y_image], \
               [source_x, source_y]

class BruteSettingsDefault(object):

    @property
    def reoptimize(self):
        return False

    @property
    def n_particles(self):
        return 35

    @property
    def n_iterations(self):
        return 300
