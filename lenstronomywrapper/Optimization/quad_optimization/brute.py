from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
from lenstronomywrapper.Optimization.quad_optimization.optimization_base import OptimizationBase

class BruteOptimization(OptimizationBase):

    def __init__(self, lens_system, n_particles=None, simplex_n_iter=None):

        """
        This class executes a lens model fit to the data using the Optimizer class in lenstronomy

        More sophisticated optimization routines are wrappers around this main class

        :param lens_system: the lens system class to optimize (instance of lenstronomywrapper.LensSystem.quad_lens
        :param n_particles: the number of particle swarm particles to use
        :param simplex_n_iter: the number of iterations for the downhill simplex routine
        :param reoptimize: whether to start the particle swarm particles close together if the initial
        guess for the lens model is close to the `true model'
        :param log_mass_sheet_front: the log(mass) used when subtracting foreground convergence sheets from the lens mdoel
        :param log_mass_sheet_back: same as ^ but for the background lens planes
        """

        settings = BruteSettingsDefault()

        if n_particles is None:
            n_particles = settings.n_particles
        if simplex_n_iter is None:
            simplex_n_iter = settings.n_iterations

        self.n_particles = n_particles
        self.n_iterations = simplex_n_iter

        self.realization_initial = lens_system.realization

        super(BruteOptimization, self).__init__(lens_system)

    def optimize(self, data_to_fit, opt_routine, constrain_params, verbose,
                 include_substructure, kwargs_optimizer):

        kwargs_lens_final, lens_model_full, [source_x, source_y] = self.fit(data_to_fit, opt_routine, constrain_params, verbose,
                        include_substructure, **kwargs_optimizer)

        return self.return_results(
            [source_x, source_y], kwargs_lens_final, lens_model_full,
            self.realization_initial, None
        )

    def fit(self, data_to_fit, opt_routine, constrain_params=None, verbose=False,
                 include_substructure=True, realization=None, opt_kwargs={}, tol_centroid=0.3, re_optimize=False,
                 particle_swarm=True, n_particles=None):

        self._check_routine(opt_routine, constrain_params)

        if n_particles is None:
            n_particles = self.n_particles

        run_kwargs = {'optimizer_routine': opt_routine, 'magnification_target': data_to_fit.m,
                      'multiplane': True, 'z_main': self.lens_system.zlens, 'z_source': self.lens_system.zsource,
                      'tol_centroid': tol_centroid, 'astropy_instance': self.lens_system.astropy, 'tol_mag': None,
                      'verbose': verbose, 're_optimize': re_optimize, 'particle_swarm': particle_swarm,
                      'pso_convergence_mean': 50000, 'constrain_params': constrain_params,
                      'simplex_n_iterations': self.n_iterations, 'optimizer_kwargs': opt_kwargs}

        kwargs_lens_final, lens_model_full, source = self._fit(data_to_fit,
                                    include_substructure, n_particles, realization, run_kwargs)

        return kwargs_lens_final, lens_model_full, source

    def _fit(self, data_to_fit, include_substructure, nparticles,
            realization, run_kwargs):

        """
        run_kwargs: {'optimizer_routine', 'constrain_params', 'simplex_n_iter'}
        filter_kwargs: {'re_optimize', 'particle_swarm'}
        """

        lens_model_list, redshift_list, kwargs_lens, numerical_alpha_class, convention_index = \
            self.lens_system.get_lenstronomy_args(include_substructure, realization=realization)

        opt = Optimizer(data_to_fit.x, data_to_fit.y, redshift_list, lens_model_list,
                        kwargs_lens, numerical_alpha_class, **run_kwargs)

        kwargs_lens_final, [source_x, source_y], _ = opt.optimize(nparticles)

        lens_model_full = opt._lensModel

        return kwargs_lens_final, lens_model_full, [source_x, source_y]

class BruteSettingsDefault(object):

    @property
    def reoptimize(self):
        return False

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 350
