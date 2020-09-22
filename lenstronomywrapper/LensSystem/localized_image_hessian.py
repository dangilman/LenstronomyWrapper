from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.lens_model import LensModel
from scipy.optimize import minimize
import numpy as np
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
import matplotlib.pyplot as plt


class LocalImageHessian(object):

    def __init__(self, lensmodel_macro, z_lens, z_source, pyhalo_cosmology=None,
                 macro_indicies_fixed=None):

        if pyhalo_cosmology is None:
            # the default cosmology in pyHalo, currently WMAP9
            pyhalo_cosmology = Cosmology()
            self.astropy = pyhalo_cosmology.astropy

        self.realization = None

        self.zlens, self.zsource = z_lens, z_source

        self.T_z_lens = pyhalo_cosmology.astropy.comoving_transverse_distance(z_lens).value

        self.T_z_source = pyhalo_cosmology.astropy.comoving_transverse_distance(z_source).value

        self.pc_per_arcsec_zsrc = 1000 * pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value ** -1

        self.shift_hessian_lensmodel = LensModel(['SHIFT', 'HESSIAN'],
                                                 lens_redshift_list=[self.zlens, self.zlens], z_source=self.zsource,
                                                 multi_plane=True, cosmo=self.astropy)

        self.shift_lens_model = LensModel(['SHIFT'],
                                          lens_redshift_list=[self.zlens], z_source=self.zsource,
                                          multi_plane=True, cosmo=self.astropy)

        self.lensmodel_macro = lensmodel_macro

        self._macro_indicies_fixed = macro_indicies_fixed

        lensmodel_fit = self.get_fit_macro(lensmodel_macro)
        self._extension = LensModelExtensions(lensmodel_fit)

    def get_fit_macro(self, lensmodel_macro):

        lens_model_list = lensmodel_macro.lens_model_list

        redshift_list = lensmodel_macro.lens_model._multi_plane_base._lens_redshift_list

        if self._macro_indicies_fixed is None:
            return self.lensmodel_macro

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

    def set_realization(self, realization):

        self.realization = realization

        _ = self.lensmodel_realization

    @property
    def lensmodel_realization(self):

        if not hasattr(self, '_lensmodel_realization'):

            assert hasattr(self, 'realization')

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
            self._lensmodel_realization = LensModel(lens_list, lens_redshift_list=redshift_list,
                                                    z_lens=self.zlens, z_source=self.zsource,
                                                    multi_plane=True, numerical_alpha_class=kwargs_lenstronomy,
                                                    cosmo=self.astropy)

        return self._lensmodel_realization

    def ray_shooting_differential(self, x, y, kwargs_shift, kwargs_hessian, foreground=None, to_zlens=False):

        if foreground is None:
            x0, y0, alpha_x, alpha_y = self.foreground_rayshooting(x, y)
        else:
            x0, y0, alpha_x, alpha_y = foreground[0], foreground[1], foreground[2], foreground[3]

        if kwargs_hessian is None:
            kwargs = kwargs_shift
            mod = self.shift_lens_model
        else:
            kwargs = kwargs_shift + kwargs_hessian
            mod = self.shift_hessian_lensmodel

        if to_zlens:
            bx = x0 / self.T_z_lens
            by = y0 / self.T_z_lens
            return bx, by

        x0, y0, alpha_x, alpha_y = mod.lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y,
                                                                       self.zlens, self.zlens, kwargs,
                                                                       include_z_start=True)

        bx, by = self.background_rayshooting(x0, y0, alpha_x, alpha_y)

        return bx / self.T_z_source, by / self.T_z_source

    def _ray_shooting_differential(self, shift, x, y, foreground):

        kwargs_shift = [{'alpha_x': shift[0], 'alpha_y': shift[1]}]
        return self.ray_shooting_differential(x, y, kwargs_shift, None, foreground)

    def alpha_differential(self, x, y, kwargs_shift, kwargs_hessian, foreground=None):

        beta_x, beta_y = self.ray_shooting_differential(x, y, kwargs_shift, kwargs_hessian, foreground)

        return x - beta_x, y - beta_y

    def foreground_rayshooting(self, alpha_x, alpha_y):

        x0 = np.zeros_like(alpha_x)
        y0 = np.zeros_like(alpha_y)
        return self.lensmodel_realization.lens_model.ray_shooting_partial(
            x0, y0, alpha_x, alpha_y, 0., self.zlens, self.kwargs_halos)

    def background_rayshooting(self, x, y, alpha_x, alpha_y):

        x, y, alpha_x, alpha_y = self.lensmodel_realization.lens_model.ray_shooting_partial(
            x, y, alpha_x, alpha_y, self.zlens, self.zsource, self.kwargs_halos)

        return x, y

    def hessian(self, x, y, kwargs_shift, kwargs_arc, scale, foreground_1=None, foreground_2=None, foreground_3=None):

        alpha_ra, alpha_dec = self.alpha_differential(x, y, kwargs_shift, kwargs_arc, foreground=foreground_1)

        alpha_ra_dx, alpha_dec_dx = self.alpha_differential(x + scale, y, kwargs_shift, kwargs_arc,
                                                            foreground=foreground_2)
        alpha_ra_dy, alpha_dec_dy = self.alpha_differential(x, y + scale, kwargs_shift, kwargs_arc,
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

    def _hessian_args_to_dict(self, f_xx, f_xy, f_yx, f_yy, x0, y0):

        return [{'f_xx': f_xx, 'f_xy': f_xy, 'f_yx': f_yx, 'f_yy': f_yy,
                 'ra_0': x0, 'dec_0': y0}]

    def compute_kwargs_shift(self, x_image, y_image, source_x, source_y):

        foreground = self.foreground_rayshooting(x_image, y_image)
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

    def solve_for_hessian(self, hessian_constraints, x, y, kwargs_shift, scale):

        xinit = np.array([hessian_constraints[0], hessian_constraints[1],
                          hessian_constraints[2], hessian_constraints[3]])

        foreground1 = self.foreground_rayshooting(x, y)
        foreground2 = self.foreground_rayshooting(x + scale, y)
        foreground3 = self.foreground_rayshooting(x, y + scale)
        args = (hessian_constraints, x, y, kwargs_shift, scale, foreground1, foreground2, foreground3)

        opt = minimize(self._minimize_hessian, xinit, args=args, method='Nelder-Mead')

        opt = opt['x']
        x_eval, y_eval = self.ray_shooting_differential(x, y, kwargs_shift, None, to_zlens=True)
        out = [{'f_xx': opt[0], 'f_xy': opt[1], 'f_yx': opt[2], 'f_yy': opt[3],
                'ra_0': x_eval, 'dec_0': y_eval}]

        return out

    def _minimize_hessian(self, args, constraint, x, y, kwargs_shift, scale, foreground1, foreground2, foreground3,
                          penalty=1e-5):

        kwargs_hessian = {'f_xx': args[0], 'f_xy': args[1], 'f_yx': args[2], 'f_yy': args[3]}

        hessian = self.hessian(x, y, kwargs_shift, [kwargs_hessian], scale, foreground_1=foreground1, foreground_2=foreground2,
                               foreground_3=foreground3)

        dh = 0
        for i in range(0, 4):
            dh += (hessian[i] - constraint[i]) ** 2 / penalty ** 2
        return dh ** 0.5
