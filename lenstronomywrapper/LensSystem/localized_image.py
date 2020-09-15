from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.lens_model import LensModel
from scipy.optimize import minimize
import numpy as np

class LocalImage(object):

    def __init__(self, lensmodel_macro, z_lens, z_source, pyhalo_cosmology=None):

        if pyhalo_cosmology is None:
            # the default cosmology in pyHalo, currently WMAP9
            pyhalo_cosmology = Cosmology()
            self.astropy = pyhalo_cosmology.astropy

        self.realization = None

        self.lensmodel_macro = lensmodel_macro

        self.zlens, self.zsource = z_lens, z_source

        self.T_z_lens = pyhalo_cosmology.astropy.comoving_transverse_distance(z_lens).value

        self.T_z_source = pyhalo_cosmology.astropy.comoving_transverse_distance(z_source).value

        self.pc_per_arcsec_zsrc = 1000 * pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value ** -1

        self._shift_lensmodel = LensModel(['SHIFT'], lens_redshift_list=[self.zlens], z_source=self.zsource,
                                          multi_plane=True, cosmo=self.astropy)

    def set_realization(self, realization):

        self.realization = realization

        _ = self.lensmodel_realization

    def source_coordinate(self, x_image, y_image, kwargs_lens):

        source_x, source_y = self.lensmodel_macro.ray_shooting(x_image, y_image, kwargs_lens)

        return source_x, source_y

    def ray_shooting(self, x, y, kwargs_shift, alpha_x_add=0, alpha_y_add=0):

        x, y, alpha_x, alpha_y = self.foreground_rayshooting(x, y)

        alpha_x += alpha_x_add
        alpha_y += alpha_y_add
        beta_x, beta_y = self._source_mapping(kwargs_shift, x, y, alpha_x, alpha_y, self.zlens, self.zsource)

        return beta_x, beta_y

    def alpha(self, x, y, kwargs_shift, alpha_x_add=0, alpha_y_add=0):

        beta_x, beta_y = self.ray_shooting(x, y, kwargs_shift, alpha_x_add, alpha_y_add)

        return x - beta_x, y - beta_y

    def map_to_source(self, x_image, y_image, source_x, source_y, window_size, npix, alpha_x_differential,
                      alpha_y_differential):

        grid_x, grid_y, shape0 = self.grid_around_image(x_image, y_image, window_size, npix)
        grid_x, grid_y = grid_x.ravel(), grid_y.ravel()

        _x, _y, _alpha_x, _alpha_y = self.foreground_rayshooting(x_image, y_image)

        alpha_shift = self._solve_deflection(_x, _y, _alpha_x, _alpha_y, source_x, source_y)

        _, _, alpha_x, alpha_y = self.foreground_rayshooting(grid_x, grid_y)
        alpha_x = alpha_x + alpha_x_differential.ravel()
        alpha_y = alpha_y + alpha_y_differential.ravel()

        kwargs_shift = [{'alpha_x': alpha_shift[0], 'alpha_y': alpha_shift[1]}]

        x, y = grid_x * self.T_z_lens, grid_y * self.T_z_lens
        beta_x, beta_y = self._source_mapping(kwargs_shift, x, y, alpha_x, alpha_y, self.zlens, self.zsource)

        return beta_x.reshape(shape0), beta_y.reshape(shape0)

    def foreground_rayshooting(self, x_image, y_image):

        x0, y0 = np.zeros_like(x_image), np.zeros_like(y_image)
        x, y, alpha_x, alpha_y = self.lensmodel_realization.lens_model.ray_shooting_partial(x0, y0, x_image, y_image, 0,
                                                                                            self.zlens,
                                                                                            self.kwargs_halos)

        return x, y, alpha_x, alpha_y

    def compute_kwargs_shift(self, x_image, y_image, source_x, source_y):

        x, y, alpha_x, alpha_y = self.foreground_rayshooting(x_image, y_image)

        out = self._solve_deflection(x, y, alpha_x, alpha_y, source_x, source_y)

        kwargs = [{'alpha_x': out[0], 'alpha_y': out[1]}]

        return kwargs

    def _solve_deflection(self, x, y, alpha_x, alpha_y, source_x, source_y):

        args = (self._shift_lensmodel, x, y, alpha_x, alpha_y, source_x, source_y, self.zlens, self.zsource)
        xinit = np.array([x_image, y_image])
        opt = minimize(self._func_to_min, xinit, args=args, method='Nelder-Mead')['x']
        return opt

    def _source_mapping(self, kwargs_shift, x, y, alpha_x, alpha_y, z_lens, z_source):

        x, y, alpha_x, alpha_y = self._shift_lensmodel.lens_model.ray_shooting_partial(x, y, alpha_x, alpha_y, z_lens,
                                                                                       z_lens, kwargs_shift,
                                                                                       include_z_start=True)

        bx, by, _, _ = self.lensmodel_realization.lens_model.ray_shooting_partial(x, y, alpha_x, alpha_y, z_lens,
                                                                                  z_source, self.kwargs_halos,
                                                                                  include_z_start=False)

        return bx / self.T_z_source, by / self.T_z_source

    def _func_to_min(self, alpha, shift_lensmodel, x, y, alpha_x, alpha_y, source_x, source_y, z_start, z_stop,
                     tol=0.0001):

        kwargs_shift = [{'alpha_x': alpha[0], 'alpha_y': alpha[1]}]

        bx, by = self._source_mapping(kwargs_shift, x, y, alpha_x, alpha_y, z_start, z_stop)

        dbx, dby = (bx - source_x) / tol, (by - source_y) / tol

        return dbx ** 2 + dby ** 2

    @property
    def lensmodel_realization(self):

        if not hasattr(self, '_lensmodel_realization'):
            assert hasattr(self, 'realization')

            realization_front, realization_back = self.realization.split_at_z(self.zlens)

            self._halo_names, self._halo_redshifts, self.kwargs_halos, self._kwargs_lenstronomy = \
                self.realization.lensing_quantities()

            self._lensmodel_realization = LensModel(self._halo_names, lens_redshift_list=list(self._halo_redshifts),
                                                    z_lens=self.zlens, z_source=self.zsource,
                                                    multi_plane=True, numerical_alpha_class=self._kwargs_lenstronomy,
                                                    cosmo=self.astropy)

        return self._lensmodel_realization

    def hessian_realization(self, x_image, y_image, scale):

        alpha_ra, alpha_dec = self.alpha(x_image, y_image, kwargs_shift, 0., 0.)

        alpha_ra_dx, alpha_dec_dx = self.alpha(x_image + scale, y_image, kwargs_shift, 0., 0.)
        alpha_ra_dy, alpha_dec_dy = self.alpha(x_image, y_image + scale, kwargs_shift, 0., 0.)

        dalpha_rara = (alpha_ra_dx - alpha_ra) / scale
        dalpha_radec = (alpha_ra_dy - alpha_ra) / scale
        dalpha_decra = (alpha_dec_dx - alpha_dec) / scale
        dalpha_decdec = (alpha_dec_dy - alpha_dec) / scale

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy

    def hessian_macro(self, x_image, y_image, kwargs_lens, scale):

        fxx, fxy, fyx, fyy = self.lensmodel_macro.hessian(x_image, y_image,
                                                          kwargs_lens, diff=scale)

        return fxx, fxy, fyx, fyy

    def effective_hessian(self, x_image, y_image, kwargs_macro, angular_scale):

        fxx_macro, fxy_macro, fyx_macro, fyy_macro = self.hessian_macro(x_image, y_image, kwargs_macro, angular_scale)
        fxx_sub, fxy_sub, fyx_sub, fyy_sub = self.hessian_realization(x_image, y_image, angular_scale)

        H_sub = np.array([[fxx_sub, fxy_sub], [fyx_sub, fyy_sub]])
        H_macro = np.array([[fxx_macro, fxy_macro], [fyx_macro, fyy_macro]])

        return H_macro - H_sub

    @staticmethod
    def kappa_gamma(fxx, fxy, fyx, fyy):

        gamma1 = 1. / 2 * (fxx - fyy)
        gamma2 = 0.5 * (fxy + fyx)
        kappa = 1. / 2 * (fxx + fyy)
        return kappa, gamma1, gamma2

    def differential_deflection_field(self, H, window_size, npix):

        delta_x, delta_y, shape0 = self.grid_around_image(0., 0., window_size, npix)
        coords = self.grids_to_coordinates(delta_x, delta_y)
        out = np.dot(coords, H)

        return out[:, 0].reshape(shape0), out[:, 1].reshape(shape0)

    def substructure_deflection_field(self, x_image, y_image, kwargs_shift,
                                      window_size, npix, alpha_x_macro_diff, alpha_y_macro_diff):

        xx, yy, shape0 = self.grid_around_image(x_image, y_image, window_size, npix)

        alpha_x, alpha_y = self.alpha(x_image, y_image, kwargs_shift, alpha_x_macro_diff, alpha_y_macro_diff)

        return alpha_x.reshape(shape0), alpha_y.reshape(shape0)

    @staticmethod
    def grid_around_image(image_x, image_y, size, npix):

        x = y = np.linspace(-size, size, npix)
        xx, yy = np.meshgrid(x, y)
        return xx + image_x, yy + image_y, xx.shape

    @staticmethod
    def grids_to_coordinates(xgrid, ygrid):

        coord = np.array([xgrid.ravel(), ygrid.ravel()]).T
        return coord
