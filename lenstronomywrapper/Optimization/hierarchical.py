from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
import numpy as np

class HierarchicalOptimization(object):

    def __init__(self, lens_system, settings=None):

        self.lens_system = lens_system
        if settings is None:
            settings = HierarchicalSettingsDefault()
        self.settings = settings

    def __call__(self, data_to_fit, opt_routine, constrain_params=None, verbose=False):

        realization = self.lens_system.realization
        foreground_realization, background_realization = self.split_realization(data_to_fit, realization)

        foreground_rays, N_foreground_halos, lens_model_raytracing, lens_model_full = \
            self._fit_foreground(data_to_fit, foreground_realization, opt_routine, constrain_params, verbose)

        info_array, lens_model_raytracing, lens_model_full = self._fit_background(
            data_to_fit, background_realization, foreground_rays, N_foreground_halos, opt_routine, lens_model_raytracing,
            lens_model_full, constrain_params, verbose)



    def split_realization(self, datatofit, realization):

        foreground = realization.filter(datatofit.x, datatofit.y, mindis_front=20,
                                        mindis_back=0, logmasscut_front=0,
                                        logabsolute_mass_cut_front=0,
                                        logmasscut_back=20,
                                        logabsolute_mass_cut_back=20, zmax=self.lens_system.zlens)

        background = realization.filter(datatofit.x, datatofit.y, mindis_front=0,
                                        mindis_back=20, logmasscut_front=20,
                                        logabsolute_mass_cut_front=20,
                                        logmasscut_back=0,
                                        logabsolute_mass_cut_back=0, zmin=self.lens_system.zlens)

        return foreground, background

    def _return_ray_path(self, x_opt, y_opt, lensModel, kwargs_lens):

        xpath, ypath = [], []

        for xi, yi in zip(x_opt, y_opt):
            _x, _y, redshifts, Tzlist = lensModel. \
                lens_model.ray_shooting_partial_steps(0, 0, xi,
                                                      yi, 0, self.lens_system.zsource, kwargs_lens)

            xpath.append(_x)
            ypath.append(_y)

        nplanes = len(xpath[0])

        x_path, y_path = [], []
        for ni in range(0, nplanes):
            arrx = np.array([xpath[0][ni], xpath[1][ni], xpath[2][ni], xpath[3][ni]])
            arry = np.array([ypath[0][ni], ypath[1][ni], ypath[2][ni], ypath[3][ni]])
            x_path.append(arrx)
            y_path.append(arry)

        return np.array(x_path), np.array(y_path), np.array(redshifts), np.array(Tzlist)

    def _fit(self, data_to_fit, nparticles, realization_filtered, run_kwargs, optimizer_kwargs, verbose):

        """
        run_kwargs: {'optimizer_routine', 'constrain_params', 'simplex_n_iter'}
        filter_kwargs: {'re_optimize', 'particle_swarm'}
        """

        lens_model_list, redshift_list, kwargs_lens, numerical_alpha_class, convention_index = self.lens_system.get_lenstronomy_args()

        opt = Optimizer(data_to_fit.x, data_to_fit.y, redshift_list, lens_model_list, kwargs_lens, numerical_alpha_class,
                 magnification_target=data_to_fit.m, multiplane=True, z_main=self.lens_system.zlens, z_source=self.lens_system.zsource,
                 astropy_instance=self.lens_system.astropy, verbose=verbose,
                 observed_convention_index=convention_index, optimizer_kwargs=optimizer_kwargs, **run_kwargs)

        kwargs_lens_final, [source_x, source_y], [x_image, y_image] = opt.optimize(nparticles)
        lens_model_raytracing = opt.lensModel
        lens_model_full = opt._lensModel
        foreground_rays = opt.lensModel._foreground._rays
        self.lens_system.update_source_position(source_x, source_y)

        return kwargs_lens_final, lens_model_raytracing, lens_model_full, foreground_rays, [x_image, y_image]

    def _fit_foreground(self, data_to_fit, realization_foreground, opt_routine, constrain_params=None, verbose=False):

        aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt = \
            self.settings.foreground_settings
        N_foreground_halos_last = 0

        for run in range(0, self.settings.foreground_runs):

            filter_kwargs = {'mindis_front': window_sizes[run], 'mindis_back': 100,
                             'source_x': self.lens_system.source_y, 'source_y': self.lens_system.source_x, 'logmasscut_front': globalmin_masses[run],
                             'logabsolute_mass_cut_front': aperture_masses[run], 'logmasscut_back': 12,
                             'logabsolute_mass_cut_back': 12, 'zmax': self.lens_system.zlens}
            if run == 0:
                realization_filtered = realization_foreground.filter(data_to_fit.x, data_to_fit.y, **filter_kwargs)
                if verbose: print('optimization ' + str(1))

            else:
                real = realization_foreground.filter(data_to_fit.x, data_to_fit.y, **filter_kwargs)
                if verbose: print('optimization ' + str(run + 1))
                realization_filtered = real.join(realization_filtered)

            N_foreground_halos = len(realization_filtered.halos)

            self.lens_system.update_realization(realization_filtered)

            if verbose:
                print('nhalos: ', len(realization_filtered.halos))
                print('aperture size: ', window_sizes[run])
                print('minimum mass in aperture: ', aperture_masses[run])
                print('minimum global mass: ', globalmin_masses[run])
                print('N foreground halos: ', N_foreground_halos)

            do_optimization = True
            if run > 0:
                if N_foreground_halos == 0:
                    do_optimization = False
                if N_foreground_halos == N_foreground_halos_last:
                    do_optimization = False
            if optimize_iteration[run] is False:
                do_optimization = False

            if do_optimization:
                if run == 0:
                    re_opt = False
                else:
                    re_opt = True

                run_kwargs = {'optimizer_routine': opt_routine, 'constrain_params': constrain_params,
                              'simplex_n_iter': self.settings.simplex_n_iter, 'particle_swarm': particle_swarm_reopt[run],
                              're_optimize': re_opt}
                optimizer_kwargs = {'re_optimize_scale': scale[run]}

                kwargs_lens_final, lens_model_raytracing, lens_model_full, foreground_rays, images = \
                    self._fit(data_to_fit, self.settings.nparticles, realization_filtered, run_kwargs, optimizer_kwargs,
                              verbose)

                self.lens_system.update_kwargs_macro(kwargs_lens_final)
                N_foreground_halos_last = N_foreground_halos

            else:

                N_foreground_halos_last = N_foreground_halos

        return foreground_rays, N_foreground_halos_last, lens_model_raytracing, lens_model_full

    def _fit_background(self, data_to_fit, realization_background, foreground_rays, N_foreground_halos,
                        opt_routine, lens_model_raytracing, lens_model_full, constrain_params=None, verbose=False):

        aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt = \
            self.settings.foreground_settings

        N_background_halos_last = 0

        backx, backy, background_Tzs, background_zs, reoptimized_realizations = [], [], [], [], []

        for run in range(0, self.settings.foreground_runs):

            filter_kwargs = {'mindis_back': window_sizes[run], 'mindis_front': 0,
                             'source_x': self.lens_system.source_x, 'source_y': self.lens_system.source_y,
                             'logmasscut_back': globalmin_masses[run],
                             'logabsolute_mass_cut_back': aperture_masses[run], 'logmasscut_front': 12,
                             'logabsolute_mass_cut_front': 12, 'zmin': self.lens_system.zlens}

            if run == 0:
                realization_background = realization_background.filter(data_to_fit.x, data_to_fit.y, **filter_kwargs)
                if verbose: print('optimization ' + str(1))

            else:

                filter_kwargs.update({'ray_x': path_x, 'ray_y': path_y, 'path_redshifts': path_redshifts,
                                          'path_Tzlist': path_Tzlist})
                real = realization_background.filter(data_to_fit.x, data_to_fit.y, **filter_kwargs)
                if verbose: print('optimization ' + str(run + 1))
                realization_filtered = real.join(realization_background)

            N_background_halos = len(realization_filtered.halos)

            self.lens_system.update_realization(realization_filtered)

            if verbose:
                print('nhalos: ', len(realization_filtered.halos))
                print('aperture size: ', window_sizes[run])
                print('minimum mass in aperture: ', aperture_masses[run])
                print('minimum global mass: ', globalmin_masses[run])
                print('N foreground halos: ', N_foreground_halos)
                print('N background halos: ', N_background_halos)

            do_optimization = True
            if run > 0:
                if N_background_halos == 0:
                    do_optimization = False
                if N_background_halos == N_background_halos_last:
                    do_optimization = False
            if optimize_iteration[run] is False:
                do_optimization = False

            if do_optimization:
                if run == 0:
                    re_opt = False
                else:
                    re_opt = True

                optimizer_kwargs = {'save_background_path': True,
                                  're_optimize_scale': scale[run],
                                  'precomputed_rays': foreground_rays}

                run_kwargs = {'optimizer_routine': opt_routine, 'constrain_params': constrain_params,
                              'simplex_n_iter': self.settings.simplex_n_iter,
                              'particle_swarm': particle_swarm_reopt[run],
                              're_optimize': re_opt}

                kwargs_lens_final, lens_model_raytracing, lens_model_full, foreground_rays, images = \
                    self._fit(data_to_fit, self.settings.nparticles, realization_filtered, run_kwargs, optimizer_kwargs,
                              verbose)

                path_x, path_y, path_redshifts, path_Tzlist = self._return_ray_path(data_to_fit.x, data_to_fit.y, lens_model_full,
                                                        kwargs_lens_final)


                backx.append(path_x)
                backy.append(path_y)
                background_Tzs.append(path_Tzlist)
                background_zs.append(path_redshifts)
                reoptimized_realizations.append(realization_filtered)
                N_background_halos_last = N_background_halos
                self.lens_system.update_kwargs_macro(kwargs_lens_final)

            else:
                reoptimized_realizations.append(realization_filtered)

        info_array = (backx, backy, background_Tzs, background_zs, reoptimized_realizations)

        return info_array, lens_model_raytracing, lens_model_full

class HierarchicalSettingsDefault(object):

    @property
    def nparticles(self):
        return 30

    @property
    def simplex_n_iter(self):
        return 200

    @property
    def foreground_runs(self):
        return 2

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [8, 7, 0]
        # add this everywhere
        globalmin_masses = [8, 8, 8]
        # window size
        window_sizes = [20, 0.4, 0.15]
        # controls starting points for re-optimizations
        scale = [1, 0.5, 0.1]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True, False]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt

    @property
    def background_runs(self):
        return 2

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [8, 7, 0]
        # add this everywhere
        globalmin_masses = [8, 8, 8]
        # window size
        window_sizes = [20, 0.4, 0.1]
        # controls starting points for re-optimizations
        scale = [1, 0.5, 0.1]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True, False]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt

