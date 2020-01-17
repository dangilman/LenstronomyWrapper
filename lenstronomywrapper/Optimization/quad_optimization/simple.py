from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomywrapper.Optimization.quad_optimization.optimization_base import OptimizationBase

class Simple(OptimizationBase):

    def __init__(self, lens_system, solver_type='PROFILE_SHEAR'):

        lens_model, kwargs_init = lens_system.get_lensmodel()
        self._solver = Solver4Point(lens_model, solver_type)
        self._kwargs_init = kwargs_init
        self._lens_model = lens_model
        super(Simple).__init__(lens_system)

    def optimize(self, data_to_fit, **kwargs):

        x_image, y_image = data_to_fit.x, data_to_fit.y

        kwargs_lens_final = self._solver.constraint_lensmodel(x_image, y_image, self._kwargs_init)

        beta_x, beta_y = self._lens_model.ray_shooting(x_image[0], y_image[0], kwargs_lens_final)

        source = [float(beta_x), float(beta_y)]

        return self._return_results(source, kwargs_lens_final, self._lens_model, None)
