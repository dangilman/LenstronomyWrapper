from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.lens_model import LensModel
from scipy.optimize import minimize
import numpy as np

class LocalImageBase(object):

    def __init__(self, lensmodel_macro, z_lens, z_source, pyhalo_cosmology,
                 macro_indicies_fixed, lens_model_special_names, lens_model_speical_redshifts):

        if pyhalo_cosmology is None:
            pyhalo_cosmology = Cosmology()
            self.astropy = pyhalo_cosmology.astropy

        self.reset_realization()

        self.zlens, self.zsource = z_lens, z_source

        self.T_z_lens = pyhalo_cosmology.astropy.comoving_transverse_distance(z_lens).value

        self.T_z_source = pyhalo_cosmology.astropy.comoving_transverse_distance(z_source).value

        self.pc_per_arcsec_zsrc = 1000 * pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value ** -1

        self.shift_lens_model = LensModel(['SHIFT'],
                                          lens_redshift_list=[self.zlens], z_source=self.zsource,
                                          multi_plane=True, cosmo=self.astropy)

        self.lens_model_special = LensModel(lens_model_special_names,
                                            lens_redshift_list=lens_model_speical_redshifts,
                                            z_source=self.zsource, multi_plane=True, cosmo=self.astropy)

        self._macro_indicies_fixed = macro_indicies_fixed

        self.lensmodel_macro_input = lensmodel_macro

        self.lensmodel_macro = self.get_fit_macro(lensmodel_macro)

        self.kwargs_arc_estimate = None
        self.kwargs_hessian_estimate = None

    def get_fit_macro(self, lensmodel_macro):

        lens_model_list = lensmodel_macro.lens_model_list

        redshift_list = lensmodel_macro.lens_model._multi_plane_base._lens_redshift_list

        self.redshift_list_macro = redshift_list

        if self._macro_indicies_fixed is None:
            return self.lensmodel_macro_input

        lens_list = []
        z_list = []

        for i, (z, model) in enumerate(zip(redshift_list, lens_model_list)):
            if i not in self._macro_indicies_fixed:
                lens_list.append(model)
                z_list.append(z)

        lensmodel = LensModel(lens_list, lens_redshift_list=z_list, multi_plane=True, z_source=self.zsource,
                              cosmo=self.astropy)

        return lensmodel

    def get_fixed_macro(self, lensmodel_macro):

        if self._macro_indicies_fixed is None:
            return [], []

        lens_model_list = lensmodel_macro.lens_model_list

        redshift_list = lensmodel_macro.lens_model._multi_plane_base._lens_redshift_list

        lens_list = []
        z_list = []

        for i, (z, model) in enumerate(zip(redshift_list, lens_model_list)):
            if i in self._macro_indicies_fixed:
                lens_list.append(model)
                z_list.append(z)

        return lens_list, z_list

    def set_realization(self, realization, lensmodel=None, kwargs_halos=None, halo_redshifts=None):

        self.realization = realization

        _ = self.lensmodel_realization(lensmodel, kwargs_halos, halo_redshifts)

    def reset_realization(self):

        self.realization = None

        self.kwargs_halos = None

        self._lensmodel_realization = None

    def lensmodel_realization(self, lensmodel=None, kwargs_halos=None, halo_redshifts=None):

        if lensmodel is not None:

            self._lensmodel_realization = lensmodel
            assert kwargs_halos is not None
            assert halo_redshifts is not None
            self.kwargs_halos = kwargs_halos
            self.halo_redshifts = halo_redshifts

        if not hasattr(self, '_lensmodel_realization') or self._lensmodel_realization is None:

            lens_list, redshift_list, kwargs_list, kwargs_lenstronomy = [], [], [], None

            macro_fixed, zmacro_fixed = self.get_fixed_macro(self.lensmodel_macro)
            lens_list += macro_fixed
            redshift_list += zmacro_fixed

            if self.realization is not None:
                halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = self.realization.lensing_quantities()
                lens_list += halo_names
                redshift_list += list(halo_redshifts)
                kwargs_list += kwargs_halos

            self.kwargs_halos = kwargs_list

            self.halo_redshifts = redshift_list

            self._lensmodel_realization = LensModel(lens_list, lens_redshift_list=redshift_list,
                                                    z_lens=self.zlens, z_source=self.zsource,
                                                    multi_plane=True, numerical_alpha_class=kwargs_lenstronomy,
                                                    cosmo=self.astropy)

        return self._lensmodel_realization

    @property
    def redshift_list(self):

        redshifts_halos = self.halo_redshifts
        redshifts_main = self.redshift_list_macro
        return redshifts_main + redshifts_halos

    def ray_shooting(self, x, y, kwargs):

        x0, y0, alpha_x, alpha_y = self.foreground_rayshooting(x, y, self.zlens)

        x0, y0, alpha_x, alpha_y = self.lens_model_special.lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y,
                                                                       self.zlens, self.zlens, kwargs,
                                                                       include_z_start=True)

        bx, by = self.background_rayshooting(x0, y0, alpha_x, alpha_y, self.zsource)

        return bx / self.T_z_source, by / self.T_z_source

    def ray_shooting_partial(self, x0, y0, alpha_x, alpha_y, zstart, zstop, kwargs):

        lensmodel_realization = self.lensmodel_realization()

        x0, y0, alpha_x, alpha_y = lensmodel_realization.lens_model.ray_shooting_partial(
            x0, y0, alpha_x, alpha_y, zstart, zstop, self.kwargs_halos)

        if zstop < self.zlens:
            return x0, y0, alpha_x, alpha_y

        x0, y0, alpha_x, alpha_y = self.lens_model_special.lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y,
                                                                                           self.zlens, self.zlens,
                                                                                           kwargs,
                                                                                           include_z_start=True)

        if zstop == self.zlens:
            return x0, y0, alpha_x, alpha_y

        x0, y0, alpha_x, alpha_y = lensmodel_realization.lens_model.ray_shooting_partial(
            x0, y0, alpha_x, alpha_y, self.zlens, zstop, self.kwargs_halos)

        return x0, y0, alpha_x, alpha_y

    def ray_shooting_differential(self, x, y, kwargs_shift, kwargs_special, foreground=None, to_zlens=False):

        if foreground is None:
            x0, y0, alpha_x, alpha_y = self.foreground_rayshooting(x, y, self.zlens)
        else:
            x0, y0, alpha_x, alpha_y = foreground[0], foreground[1], foreground[2], foreground[3]

        if kwargs_special is None:
            kwargs = kwargs_shift
            mod = self.shift_lens_model
        else:
            kwargs = kwargs_shift + kwargs_special
            mod = self.lens_model_special

        if to_zlens:
            bx = x0 / self.T_z_lens
            by = y0 / self.T_z_lens
            return bx, by

        x0, y0, alpha_x, alpha_y = mod.lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y,
                                                                       self.zlens, self.zlens, kwargs,
                                                                       include_z_start=True)

        bx, by = self.background_rayshooting(x0, y0, alpha_x, alpha_y, self.zsource)

        return bx / self.T_z_source, by / self.T_z_source

    def _ray_shooting_differential(self, shift, x, y, foreground):

        kwargs_shift = [{'alpha_x': shift[0], 'alpha_y': shift[1]}]
        return self.ray_shooting_differential(x, y, kwargs_shift, None, foreground)

    def alpha_differential(self, x, y, kwargs_shift, kwargs_special, **kwargs_rayshooting):

        beta_x, beta_y = self.ray_shooting_differential(x, y, kwargs_shift, kwargs_special,
                                                        **kwargs_rayshooting)

        return x - beta_x, y - beta_y

    def foreground_rayshooting(self, alpha_x, alpha_y, zstop):

        x0 = np.zeros_like(alpha_x)
        y0 = np.zeros_like(alpha_y)
        lensmodel_realization = self.lensmodel_realization()

        return lensmodel_realization.lens_model.ray_shooting_partial(
            x0, y0, alpha_x, alpha_y, 0., zstop, self.kwargs_halos)

    def background_rayshooting(self, x, y, alpha_x, alpha_y, zstop):

        lensmodel_realization = self.lensmodel_realization()
        x, y, alpha_x, alpha_y = lensmodel_realization.lens_model.ray_shooting_partial(
            x, y, alpha_x, alpha_y, self.zlens, zstop, self.kwargs_halos)

        return x, y

    def hessian(self, x, y, kwargs_shift, kwargs_special,
                scale, foreground_1=None, foreground_2=None, foreground_3=None):

        alpha_ra, alpha_dec = self.alpha_differential(x, y, kwargs_shift, kwargs_special,
                                                      foreground=foreground_1)

        alpha_ra_dx, alpha_dec_dx = self.alpha_differential(x + scale, y, kwargs_shift, kwargs_special,
                                                            foreground=foreground_2)
        alpha_ra_dy, alpha_dec_dy = self.alpha_differential(x, y + scale, kwargs_shift, kwargs_special,
                                                            foreground=foreground_3)

        dalpha_rara = (alpha_ra_dx - alpha_ra) / scale
        dalpha_radec = (alpha_ra_dy - alpha_ra) / scale
        dalpha_decra = (alpha_dec_dx - alpha_dec) / scale
        dalpha_decdec = (alpha_dec_dy - alpha_dec) / scale

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy

    def kappa_gamma(self, x, y, kwargs_shift, kwargs_special, scale):

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs_shift, kwargs_special, scale)

        kappa = 0.5 * (f_xx + f_yy)
        gamma1 = 0.5 * (f_xx - f_yy)
        gamma2 = 0.5 * (f_xy + f_yx)

        return kappa, gamma1, gamma2

    def _hessian_args_to_dict(self, f_xx, f_xy, f_yx, f_yy, x0, y0):

        return [{'f_xx': f_xx, 'f_xy': f_xy, 'f_yx': f_yx, 'f_yy': f_yy,
                 'ra_0': x0, 'dec_0': y0}]

    def compute_kwargs_shift(self, x_image, y_image, source_x, source_y):

        foreground = self.foreground_rayshooting(x_image, y_image, self.zlens)
        args = (x_image, y_image, source_x, source_y, foreground)
        xinit = np.array([x_image, y_image])
        out = minimize(self._func_to_min, xinit, args=args, method='Nelder-Mead')['x']

        kwargs = [{'alpha_x': out[0], 'alpha_y': out[1]}]

        return kwargs

    def _func_to_min(self, alpha, alpha_x, alpha_y, source_x, source_y, foreground, tol=0.0001):

        bx, by = self._ray_shooting_differential(alpha, alpha_x, alpha_y, foreground)

        dx, dy = bx - source_x, by - source_y

        dbx, dby = dx / tol, dy / tol

        return dbx ** 2 + dby ** 2

    @staticmethod
    def grid_around_image(image_x, image_y, size, npix):
        x = y = np.linspace(-size, size, npix)
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx + image_x, yy + image_y
        return xx, yy, xx.shape
