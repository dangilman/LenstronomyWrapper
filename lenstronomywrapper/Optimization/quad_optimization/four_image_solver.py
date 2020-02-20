from lenstronomywrapper.Optimization.quad_optimization.optimization_base import OptimizationBase
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths
import numpy as np

from lenstronomy.LensModel.Solver.solver import Solver

class FourImageSolver(OptimizationBase):

    def __init__(self, lens_system):

        ray_interp_x, ray_interp_y = interpolate_ray_paths([0.], [0.], lens_system,
                                                           include_substructure=False, realization=None)

        self.realization_initial = lens_system.realization.shift_background_to_source(ray_interp_x[0],
                                                                                      ray_interp_y[0])

        super(FourImageSolver, self).__init__(lens_system)

    def optimize(self, data_to_fit, constrain_params, include_substructure=True):

        assert isinstance(constrain_params, dict)
        assert 'shear' in constrain_params.keys()

        # first solve with no substructure
        lensModel, kwargs = self.lens_system.get_lensmodel(include_substructure=include_substructure)

        gamma1, gamma2 = constrain_params['shear'], 0.
        kwargs[1]['gamma1'], kwargs[1]['gamma2'] = gamma1, gamma2
        solver = Solver('PROFILE_SHEAR', lensModel, 4)
        kwargs_fit, precision = solver.constraint_lensmodel(data_to_fit.x, data_to_fit.y, kwargs)
        self.lens_system.update_kwargs_macro(kwargs_fit)

        srcx, srcy = lensModel.ray_shooting(data_to_fit.x, data_to_fit.y, kwargs_fit)
        source_x, source_y = np.mean(srcx), np.mean(srcy)

        return_kwargs = {'info_array': None,
                         'lens_model_raytracing': lensModel,
                         'realization_final': self.realization_initial}

        return self._return_results([source_x, source_y], kwargs_fit, lensModel, return_kwargs)
