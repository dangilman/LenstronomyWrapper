from lenstronomywrapper.Optimization.quad_optimization.settings import *
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths

class HierarchicalOptimizationLocal(object):

    def __init__(self, local_image_quad):

        self.local_image_quad = local_image_quad

    def fit(self, angular_scale, hessian_constraints, log_global_min, aperture_sizes, log_aperture_masses,
            refit, kwargs_evaluate={}, verbose=False):

        realization_init = self.local_image_quad.realization
        if verbose:
            print('Initial realization contains '+str(len(realization_init.halos))+' halos.')
        # first do the computation with the largest halos included everywhere
        realization_filtered = realization_init.filter_by_mass(10**log_global_min)
        nhalos = len(realization_filtered.halos)
        if verbose:
            print('Realization contains ' + str(nhalos) + ' halos more massive than 10^'+str(log_global_min))

        self.local_image_quad.clear_realization()
        self.local_image_quad.set_realization(realization_filtered)

        deltas = self.local_image_quad.evaluate(angular_scale, hessian_constraints, **kwargs_evaluate)

        srcx, srcy = self.local_image_quad._source_x, self.local_image_quad._source_y

        for aperture_size, logm_in_aperture, refit in zip(aperture_sizes, log_aperture_masses, refit):

            x_interp_list, y_interp_list = [], []

            for i, (local, ximg, yimg) in enumerate(zip(self.local_image_quad.class_list,
                                                        self.local_image_quad.x_image,
                                                        self.local_image_quad.y_image)):

                kwargs = self.local_image_quad.kwargs_shift_list[i] + self.local_image_quad.kwargs_local_special[i]
                x_interp, y_interp = interpolate_ray_paths([ximg], [yimg], local, kwargs, local.zsource,
                                                           terminate_at_source=True, source_x=srcx, source_y=srcy)
                x_interp_list += x_interp
                y_interp_list += y_interp

            filter_kwargs = {'aperture_radius_front': aperture_size,
                             'aperture_radius_back': aperture_size,
                             'mass_allowed_in_apperture_front': logm_in_aperture,
                             'mass_allowed_in_apperture_back': logm_in_aperture,
                             'mass_allowed_global_front': log_global_min,
                             'mass_allowed_global_back': log_global_min,
                             'interpolated_x_angle': x_interp_list,
                             'interpolated_y_angle': y_interp_list,
                             }

            real = realization_init.filter(**filter_kwargs)
            realization_filtered = realization_filtered.join(real)

            nhalos_added = len(realization_filtered.halos) - nhalos
            if verbose:
                print('added '+str(nhalos_added)+'.... ')

            self.local_image_quad.clear_realization()
            self.local_image_quad.set_realization(realization_filtered)

            if nhalos_added > 0 and refit:
                deltas = self.local_image_quad.evaluate(angular_scale, hessian_constraints, find_kwargs_special=refit,
                                                        **kwargs_evaluate)

        return deltas



