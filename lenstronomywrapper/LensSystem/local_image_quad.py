from lenstronomywrapper.LensSystem.localized_image_arc import LocalImageArc
import numpy as np
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths
from pyHalo.Cosmology.cosmology import Cosmology

class LocalImageQuad(object):

    def __init__(self, x_image_list, y_image_list, source_x, source_y,
                 lensmodel_macro, kwargs_lens_macro, z_lens, z_source,
                 macro_indicies_fixed=None, pyhalo_cosmology=None):

        class_list = []

        for (x, y) in zip(x_image_list, y_image_list):

            new = LocalImageArc(lensmodel_macro, z_lens,
                 z_source, pyhalo_cosmology, macro_indicies_fixed)

            class_list.append(new)

        if pyhalo_cosmology is None:
            # the default cosmology in pyHalo, currently WMAP9
            pyhalo_cosmology = Cosmology()

        self.x_image, self.y_image = x_image_list, y_image_list

        self._source_x, self._source_y = source_x, source_y
        self.class_list = class_list

        self.kwargs_lens_macro = kwargs_lens_macro
        self.lensmodel_macro_input = lensmodel_macro

        self.z_lens, self.z_source = z_lens, z_source
        self.source_x, self.source_y = source_x, source_y

        self._pyhalo_cosmology = pyhalo_cosmology

        self.pc_per_arcsec_zsource = 1000 * pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value ** -1

        self._realization_init = False

    def ray_shooting(self, xgrid, ygrid, image_index):

        kwargs = self.kwargs_shift_list[image_index] + self.kwargs_local_special[image_index]
        bx, by = self.class_list[image_index].ray_shooting(xgrid, ygrid, kwargs)
        return bx, by

    @property
    def kwargs_local(self):

        kwargs = []
        for i in range(0, len(self.x_image)):
            kwargs.append(self.kwargs_shift_list[i] + self.kwargs_local_special[i])

        return kwargs

    def clear_realization(self):

        for local_image in self.class_list:
            local_image.reset_realization()

    def set_realization(self, realization):

        if self._realization_init is False:
            error_message = 'First initialize realization by calling realization_init(realization). ' \
                            'This is necessary to align the halos in the realization along a path linking the' \
                            'the lens centroid with the background source coordinate.'
            raise Exception(error_message)

        lensmodel, kwargs, halo_redshifts = None, None, None
        for i, local_image in enumerate(self.class_list):

            local_image.set_realization(realization, lensmodel, kwargs, halo_redshifts)
            lensmodel = local_image.lensmodel_realization()
            kwargs = local_image.kwargs_halos
            halo_redshifts = local_image.halo_redshifts

    def realization_init(self, realization):

        self._realization_init = True

        coord_x = [self.kwargs_lens_macro[0]['center_x']]
        coord_y = [self.kwargs_lens_macro[0]['center_y']]

        ray_interp_x, ray_interp_y = realization.interpolate_ray_paths(coord_x, coord_y,
                    self.lensmodel_macro_input, self.kwargs_lens_macro, self.z_source,
                                                                       terminate_at_source=True,
                                                                       source_x=self.source_x,
                                                                       source_y=self.source_y)

        realization = realization.shift_background_to_source(ray_interp_x[0], ray_interp_y[0])

        self.realization = realization

        self.set_realization(realization)

    def magnification(self, source_model, adaptive=True, verbose=False, normed=True):

        mag, blended = source_model.magnification(self.x_image, self.y_image, self.class_list,
                      self.kwargs_local, normed=normed, adaptive=adaptive, verbose=verbose)

        return mag, blended

    def plot_images(self, x, y, source_model, adaptive=False):

        return source_model.plot_images(x, y, self.class_list, self.kwargs_local,
                                                      adaptive=adaptive)

    def evaluate(self, angular_scale, hessian_constraints, fixed_curvature=False, fixed_direction=False,
                 fixed_curvature_direction=False, verbose=False, find_kwargs_shift=True, find_kwargs_special=True):

        assert self._realization_init
        assert len(hessian_constraints) == len(self.class_list)

        if find_kwargs_special:
            self.kwargs_local_special = []

        if find_kwargs_shift:
            self.kwargs_shift_list = []

        delta_kappa = []
        delta_gamma1 = []
        delta_gamma2 = []

        for i, local_image in enumerate(self.class_list):

            x, y = self.x_image[i], self.y_image[i]

            if find_kwargs_shift:
                kwargs_shift = local_image.compute_kwargs_shift(x, y, self._source_x, self._source_y)
                self.kwargs_shift_list.append(kwargs_shift)

            if find_kwargs_special:
                constraints = (hessian_constraints[0][i], hessian_constraints[1][i], hessian_constraints[2][i], hessian_constraints[3][i])

                if local_image.kwargs_arc_estimate is not None:
                    kwargs_arc_estimate = local_image.kwargs_arc_estimate
                else:
                    kwargs_arc_estimate = None

                kwargs = local_image.solve_for_arc(constraints, self.kwargs_lens_macro,
                                                         x, y, self.kwargs_shift_list[i], angular_scale, fixed_curvature,
                                                         fixed_direction, fixed_curvature_direction,
                                                   verbose, kwargs_arc_estimate)

                kappa_model, gamma1_model, gamma2_model = local_image.kappa_gamma(x, y, self.kwargs_shift_list[i],
                                                                                 kwargs, angular_scale)

                self.kwargs_local_special.append(kwargs)

                fxx, fxy, fyx, fyy = constraints[0], constraints[1], constraints[2], constraints[3]
                kappa = 1. / 2 * (fxx + fyy)
                gamma1 = 1. / 2 * (fxx - fyy)
                gamma2 = 1. / 2 * (fxy + fyx)

                delta_kappa.append(kappa_model - kappa)
                delta_gamma1.append(gamma1_model - gamma1)
                delta_gamma2.append(gamma2_model - gamma2)

        return delta_kappa, delta_gamma1, delta_gamma2

    @staticmethod
    def grid_around_image(image_x, image_y, size, npix):
        x = y = np.linspace(-size, size, npix)
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx + image_x, yy + image_y
        return xx, yy, xx.shape


