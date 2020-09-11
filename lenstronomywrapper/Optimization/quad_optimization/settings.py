class HierarchicalSettingsDeltaFunction(object):
    """
    Good for dealing with large numbers of tiny dark matter halos below 10^6
    """

    def __init__(self, delta_function_logmass, window_sizes):

        self.delta_function_logmass = delta_function_logmass
        self.window_sizes = window_sizes

    @property
    def log_mass_cut_global(self):
        return self.delta_function_logmass

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 300

    @property
    def n_iterations_background(self):
        return 1

    @property
    def n_iterations_foreground(self):
        return 1

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [self.delta_function_logmass]
        # add this everywhere
        globalmin_masses = [self.delta_function_logmass+1.]
        # window size
        window_sizes = self.window_sizes
        # controls starting points for re-optimizations
        scale = [1.0] * len(aperture_masses)
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True] * len(aperture_masses)
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True] * len(aperture_masses)
        if len(aperture_masses) > 1:
            particle_swarm_reopt += [False] * (len(aperture_masses) - 1)

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [self.delta_function_logmass]
        # add this everywhere
        globalmin_masses = [self.delta_function_logmass+1.]
        # window size
        window_sizes = self.window_sizes
        # controls starting points for re-optimizations
        scale = [1.0] * len(aperture_masses)
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True] * len(aperture_masses)
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True] * len(aperture_masses)
        if len(aperture_masses) > 1:
            particle_swarm_reopt += [False] * (len(aperture_masses)-1)

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration


class HierarchicalSettingsDefault(object):

    """
    Good for dealing with dark matter halos between 10^5 - 10^10 M_sun
    """

    @property
    def log_mass_cut_global(self):
        return 7.7

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 350

    @property
    def n_iterations_background(self):
        return 3

    @property
    def n_iterations_foreground(self):
        return 2

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global]*2
        # window size
        window_sizes = [100, 0.3]
        # controls starting points for re-optimizations
        scale = [1, 0.1]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 7., 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global, self.log_mass_cut_global, self.log_mass_cut_global]
        # window size
        window_sizes = [100, 0.3, 0.1]
        # controls starting points for re-optimizations
        scale = [1, 0.5, 0.25]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [False, False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True, True, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

class HierarchicalSettingsCDM(object):

    """
    Good for dealing with dark matter halos between 10^5 - 10^10 M_sun
    with small source sizes <10 pc
    """

    @property
    def log_mass_cut_global(self):
        return 7.5

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 350

    @property
    def n_iterations_background(self):
        return 3

    @property
    def n_iterations_foreground(self):
        return 2

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global]*2
        # window size
        window_sizes = [100, 0.3]
        # controls starting points for re-optimizations
        scale = [1, 0.1]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 6.7, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global, self.log_mass_cut_global, self.log_mass_cut_global]
        # window size
        window_sizes = [100, 0.3, 0.1]
        # controls starting points for re-optimizations
        scale = [1, 0.5, 0.25]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [False, False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True, False]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True, True, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration


class HierarchicalSettingsCDM2(object):

    """
    Good for dealing with dark matter halos between 10^5 - 10^10 M_sun
    with small source sizes <10 pc
    """

    @property
    def log_mass_cut_global(self):
        return 8.

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 350

    @property
    def n_iterations_background(self):
        return 3

    @property
    def n_iterations_foreground(self):
        return 2

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global]*2
        # window size
        window_sizes = [100, 0.3]
        # controls starting points for re-optimizations
        scale = [1, 0.1]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 7., 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global, self.log_mass_cut_global, self.log_mass_cut_global]
        # window size
        window_sizes = [100, 0.3, 0.1]
        # controls starting points for re-optimizations
        scale = [1, 0.5, 0.25]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [False, False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True, False]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True, True, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

class HierarchicalSettingsCDM3(object):

    """
    Good for dealing with dark matter halos between 10^5 - 10^10 M_sun
    with small source sizes <10 pc
    """

    @property
    def log_mass_cut_global(self):
        return 7.

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 350

    @property
    def n_iterations_background(self):
        return 2

    @property
    def n_iterations_foreground(self):
        return 2

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global]*2
        # window size
        window_sizes = [100, 0.3]
        # controls starting points for re-optimizations
        scale = [1, 0.1]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global, self.log_mass_cut_global]
        # window size
        window_sizes = [100, 0.1]
        # controls starting points for re-optimizations
        scale = [1, 0.25]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, False]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

class HierarchicalSettingsCDM4(object):

    """
    Good for dealing with dark matter halos between 10^5 - 10^10 M_sun
    with small source sizes <10 pc
    """

    @property
    def log_mass_cut_global(self):
        return 6.7

    @property
    def n_particles(self):
        return 30

    @property
    def n_iterations(self):
        return 350

    @property
    def n_iterations_background(self):
        return 2

    @property
    def n_iterations_foreground(self):
        return 2

    @property
    def foreground_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global]*2
        # window size
        window_sizes = [100, 0.3]
        # controls starting points for re-optimizations
        scale = [1, 0.1]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [True, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, True]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [False, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration

    @property
    def background_settings(self):
        # add this only within the window
        aperture_masses = [self.log_mass_cut_global, 0]
        # add this everywhere
        globalmin_masses = [self.log_mass_cut_global, self.log_mass_cut_global]
        # window size
        window_sizes = [100, 0.1]
        # controls starting points for re-optimizations
        scale = [1, 0.25]
        # determines whether to use PSO for re-optimizations
        particle_swarm_reopt = [False, False]
        # wheter to actually re-fit the lens model
        optimize_iteration = [True, False]
        # whether to re-optimize (aka start from a model very close to input model)
        re_optimize_iteration = [True, True]

        return aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, re_optimize_iteration
