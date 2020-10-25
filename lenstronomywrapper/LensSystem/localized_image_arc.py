from scipy.optimize import minimize
import numpy as np

from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

from lenstronomywrapper.LensSystem.local_image_base import LocalImageBase

class LocalImageArc(LocalImageBase):

    def __init__(self, lensmodel_macro, z_lens, z_source, pyhalo_cosmology=None,
                 macro_indicies_fixed=None):

        lens_model_special_names = ['SHIFT', 'CURVED_ARC']
        lens_model_speical_redshifts = [z_lens, z_lens]

        super(LocalImageArc, self).__init__(lensmodel_macro, z_lens, z_source, pyhalo_cosmology,
                                            macro_indicies_fixed, lens_model_special_names,
                                            lens_model_speical_redshifts)

    def _fixed_curvature_direction(self, kwargs_arc_macro, hessian_constraints,
                         arc_center_x, arc_center_y, x, y, kwargs_shift,
                         scale, foreground1, foreground2, foreground3, verbose):

        curvature = kwargs_arc_macro['curvature']
        direction = kwargs_arc_macro['direction']

        kwargs_keys = ['radial_stretch', 'tangential_stretch']
        args = (hessian_constraints, x, y, kwargs_shift, scale, arc_center_x, arc_center_y, curvature,
                direction, foreground1, foreground2, foreground3)

        xinit = np.array([kwargs_arc_macro[name] for name in kwargs_keys])
        opt = minimize(self._minimize_arc, xinit, args=args, method='Nelder-Mead')
        if verbose:
            print(opt)
        opt = opt['x']
        out = [{'radial_stretch': opt[0], 'tangential_stretch': opt[1],
                'curvature': curvature, 'direction': direction, 'center_x': arc_center_x, 'center_y': arc_center_y}]

        return out

    def _fixed_curvature(self, kwargs_arc_macro, hessian_constraints,
                         arc_center_x, arc_center_y, x, y, kwargs_shift,
                         scale, foreground1, foreground2, foreground3, verbose):

        curvature = kwargs_arc_macro['curvature']
        kwargs_keys = ['radial_stretch', 'tangential_stretch', 'direction']
        args = (hessian_constraints, x, y, kwargs_shift, scale, arc_center_x, arc_center_y, curvature,
                None, foreground1, foreground2, foreground3)

        xinit = np.array([kwargs_arc_macro[name] for name in kwargs_keys])
        opt = minimize(self._minimize_arc, xinit, args=args, method='Nelder-Mead')
        if verbose:
            print(opt)
        opt = opt['x']
        out = [{'radial_stretch': opt[0], 'tangential_stretch': opt[1],
                'curvature': curvature, 'direction': opt[2], 'center_x': arc_center_x, 'center_y': arc_center_y}]

        return out

    def _fixed_direction(self, kwargs_arc_macro, hessian_constraints,
                         arc_center_x, arc_center_y, x, y, kwargs_shift,
                         scale, foreground1, foreground2, foreground3, verbose):

        direction = kwargs_arc_macro['direction']
        kwargs_keys = ['radial_stretch', 'tangential_stretch', 'curvature']
        args = (hessian_constraints, x, y, kwargs_shift, scale, arc_center_x, arc_center_y, None,
                direction, foreground1, foreground2, foreground3)

        xinit = np.array([kwargs_arc_macro[name] for name in kwargs_keys])
        opt = minimize(self._minimize_arc, xinit, args=args, method='Nelder-Mead')
        if verbose:
            print(opt)
        opt = opt['x']
        out = [{'radial_stretch': opt[0], 'tangential_stretch': opt[1],
                'curvature': opt[2], 'direction': direction, 'center_x': arc_center_x, 'center_y': arc_center_y}]

        return out

    def _nothing_fixed(self, kwargs_arc_macro, hessian_constraints,
                         arc_center_x, arc_center_y, x, y, kwargs_shift,
                         scale, foreground1, foreground2, foreground3, verbose):

        kwargs_keys = ['radial_stretch', 'tangential_stretch', 'curvature', 'direction']
        args = (hessian_constraints, x, y, kwargs_shift, scale, arc_center_x, arc_center_y, None,
                None, foreground1, foreground2, foreground3)

        xinit = np.array([kwargs_arc_macro[name] for name in kwargs_keys])
        opt = minimize(self._minimize_arc, xinit, args=args, method='Nelder-Mead')
        if verbose:
            print(opt)
        opt = opt['x']
        out = [{'radial_stretch': opt[0], 'tangential_stretch': opt[1],
                'curvature': opt[2], 'direction': opt[3], 'center_x': arc_center_x, 'center_y': arc_center_y}]

        return out

    def solve_for_arc(self, hessian_constraints, kwargs_macro, x, y, kwargs_shift, scale,
                      fixed_curvature=False, fixed_direction=False, fixed_curvature_direction=False,
                      verbose=False, kwargs_arc_estimate=None):

        extension = LensModelExtensions(self.lensmodel_macro_input)

        if kwargs_arc_estimate is None:
            kwargs_arc_estimate = extension.curved_arc_estimate(x, y, kwargs_macro)

        arc_center_x, arc_center_y = self.ray_shooting_differential(x, y,
                                         kwargs_shift, None, to_zlens=True)

        foreground1 = self.foreground_rayshooting(x, y, self.zlens)
        foreground2 = self.foreground_rayshooting(x + scale, y, self.zlens)
        foreground3 = self.foreground_rayshooting(x, y + scale, self.zlens)

        if fixed_curvature:
            fun = self._fixed_curvature

        elif fixed_direction:
            fun = self._fixed_direction

        elif fixed_curvature_direction:
            fun = self._fixed_curvature_direction

        else:
            fun = self._nothing_fixed

        kwargs_arc = fun(kwargs_arc_estimate, hessian_constraints,
                         arc_center_x, arc_center_y, x, y, kwargs_shift,
                         scale, foreground1, foreground2, foreground3, verbose)
        self.kwargs_arc_estimate = kwargs_arc[0]

        return kwargs_arc


    def _minimize_arc(self, args, constraint, x, y, kwargs_shift, scale, center_x, center_y, curvature,
                      direction, foreground1, foreground2, foreground3, penalty=1e-8):

        kwargs_arc = {}

        kwargs_arc['radial_stretch'] = args[0]
        kwargs_arc['tangential_stretch'] = args[1]
        kwargs_arc['center_x'] = center_x
        kwargs_arc['center_y'] = center_y

        idx_next = 2

        if curvature is not None:
            kwargs_arc['curvature'] = curvature
        else:
            kwargs_arc['curvature'] = args[idx_next]
            idx_next += 1

        if direction is not None:
            kwargs_arc['direction'] = direction
        else:
            kwargs_arc['direction'] = args[idx_next]

        hessian = self.hessian(x, y, kwargs_shift, [kwargs_arc], scale, foreground_1=foreground1, foreground_2=foreground2,
                               foreground_3=foreground3)

        dh = 0
        for i in range(0, 4):
            dh += (hessian[i] - constraint[i])**2/penalty**2
        return dh ** 0.5
