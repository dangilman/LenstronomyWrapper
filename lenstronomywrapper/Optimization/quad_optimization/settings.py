class HierarchicalSettingsLowMass(object):
    """
    Good for dealing with large numbers of tiny dark matter halos below 10^6
    """

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 250

    @property
    def n_iterations_background(self):
        return 3

    @property
    def n_iterations_foreground(self):
        return 3

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [8, 5.7, 5., 0]
        # add this everywhere
        globalmin_masses = [10., 7., 7., 7.]
        # window size
        window_sizes = [20, 0.3, 0.15, 0.1]
        # controls starting points for re-optimizations
        scale = [1, 0.5, 0.5, 0.5]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, True, False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True, True, False]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, True, True, False]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [8., 5.8, 5.5, 5.25, 5., 4.5, 0]
        # add this everywhere
        globalmin_masses = [10] * (len(aperture_masses) - 1)
        # window size
        window_sizes = [20, 0.35, 0.2, 0.1, 0.05, 0.025, 0.01]
        # controls starting points for re-optimizations
        scale = [1] + [0.4] * (len(aperture_masses)-1)
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, True, False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True, True, False]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, True, True, False]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration


class HierarchicalSettingsDefault(object):

    """
    Good for dealing with dark matter halos between 10^5 - 10^10 M_sun
    """

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 250

    @property
    def n_iterations_background(self):
        return 3

    @property
    def n_iterations_foreground(self):
        return 3

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
        optimize_iteration = [True, True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, False, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

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
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True, True, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration
