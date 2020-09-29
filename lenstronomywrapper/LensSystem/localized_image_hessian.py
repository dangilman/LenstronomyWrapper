from scipy.optimize import minimize
import numpy as np

from lenstronomywrapper.LensSystem.local_image_base import LocalImageBase

class LocalImageHessian(LocalImageBase):

    def __init__(self, lensmodel_macro, z_lens, z_source, pyhalo_cosmology=None,
                 macro_indicies_fixed=None):

        lens_model_special_names = ['SHIFT', 'HESSIAN']
        lens_model_speical_redshifts = [z_lens, z_lens]

        super(LocalImageHessian, self).__init__(lensmodel_macro, z_lens, z_source, pyhalo_cosmology,
                                            macro_indicies_fixed, lens_model_special_names,
                                                lens_model_speical_redshifts)

    def solve_for_hessian(self, hessian_constraints, x, y, kwargs_shift, scale, verbose=False):

        xinit = np.array([hessian_constraints[0], hessian_constraints[1],
                          hessian_constraints[2], hessian_constraints[3]])

        foreground1 = self.foreground_rayshooting(x, y, self.zlens)
        foreground2 = self.foreground_rayshooting(x + scale, y, self.zlens)
        foreground3 = self.foreground_rayshooting(x, y + scale, self.zlens)
        args = (hessian_constraints, x, y, kwargs_shift, scale, foreground1, foreground2, foreground3)

        opt = minimize(self._minimize_hessian, xinit, args=args, method='Nelder-Mead')
        if verbose: print(opt)
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
