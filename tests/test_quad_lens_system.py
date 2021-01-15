import pytest
import numpy as np
import numpy.testing as npt
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from pyHalo.single_realization import SingleHalo

from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.LensComponents.SIS import SISsatellite
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from pyHalo.Cosmology.cosmology import Cosmology
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar

class TestQuadLensSystem(object):

    def setup(self):

        zlens = 0.5
        zsource = 1.5
        self.zlens, self.zsource = zlens, zsource
        kwargs_epl_shear = [{'theta_E': 1., 'center_x': 0., 'center_y': 0., 'e1': 0.3, 'e2': -0.1, 'gamma': 2.},
                            {'gamma1': 0.05, 'gamma2': 0.}]
        kwargs_satellite = [{'theta_E': 0.1, 'center_x': 0.9, 'center_y': -1.2}]

        main_deflector = PowerLawShear(zlens, kwargs_epl_shear)
        satellite = SISsatellite(zlens + 0.2, kwargs_satellite)
        macromodel = MacroLensModel([main_deflector, satellite])

        self.halo_1 = SingleHalo(10 ** 9.3, -0.41, -0.89, 100, 'TNFW', zlens+0.05, zlens, zsource)
        self.halo_2 = SingleHalo(10 ** 8, 0.2, 0.9, 100, 'TNFW', 0.3, zlens, zsource)
        self.halo_3 = SingleHalo(10 ** 7, 0., 0., 100, 'TNFW', zsource-0.0001, zlens, zsource)
        self.realization = self.halo_1.join(self.halo_2).join(self.halo_3)


        cosmo = Cosmology()
        self.quad_lens_simple = QuadLensSystem(macromodel, zsource, substructure_realization=None, pyhalo_cosmology=None)
        self.quad_lens = QuadLensSystem(macromodel, zsource, substructure_realization=self.realization, pyhalo_cosmology=cosmo)

        self.lens_model_simple, self.kwargs_simple = self.quad_lens_simple.get_lensmodel()
        self.lens_model, self.kwargs = self.quad_lens.get_lensmodel()

        solver = LensEquationSolver(self.lens_model_simple)
        self.x_simple, self.y_simple = solver.image_position_from_source(0.05, 0.0, self.kwargs_simple)
        solver = LensEquationSolver(self.lens_model)
        self.x, self.y = solver.image_position_from_source(0.05, 0.0, self.kwargs)

        self.macromodel = macromodel

    def test_get_lensmodel(self):

        lens_model_simple, kwargs_simple = self.quad_lens_simple.get_lensmodel()
        lens_model_simple_2, kwargs_simple_2 = self.quad_lens.get_lensmodel(include_substructure=False)
        npt.assert_equal(len(lens_model_simple.lens_model_list), len(lens_model_simple_2.lens_model_list))

        lens_model, kwargs = self.quad_lens.get_lensmodel()
        lens_model_2, kwargs_2 = self.quad_lens.get_lensmodel(substructure_realization=self.halo_1)
        npt.assert_equal(len(lens_model.lens_model_list), len(lens_model_2.lens_model_list) + 2)
        lens_model_nomacro, kwargs_nomacro = self.quad_lens.get_lensmodel(include_macromodel=False,
                                                                          substructure_realization=self.halo_2)
        npt.assert_equal(len(lens_model_nomacro.lens_model_list), 1)
        npt.assert_string_equal(lens_model_nomacro.lens_model_list[0], 'TNFW')

        self.quad_lens.set_lensmodel_static(lens_model, kwargs)
        lens_model_static, kwargs_static = self.quad_lens.get_lensmodel()
        npt.assert_equal(len(lens_model_static.lens_model_list), len(lens_model.lens_model_list))

    def test_shift_background_auto(self):

        original_halo_positions_x = [halo.x for halo in self.realization.halos]
        original_halo_positions_y = [halo.y for halo in self.realization.halos]
        data = LensedQuasar(self.x, self.y, np.ones_like(self.x))
        quad_lens = QuadLensSystem.shift_background_auto(data, self.macromodel, self.zsource,
                                                         self.realization)
        new_halo_positions_x = [halo.x for halo in quad_lens.realization.halos]
        new_halo_positions_y = [halo.y for halo in quad_lens.realization.halos]

        lensmodel_nosubs, kw_nosubs = quad_lens.get_lensmodel(include_substructure=False)
        betax, betay = lensmodel_nosubs.ray_shooting(self.x, self.y, kw_nosubs)
        source_x_nosubs, source_y_nosubs = np.mean(betax), np.mean(betay)
        npt.assert_almost_equal(new_halo_positions_x[0], source_x_nosubs, 4)
        npt.assert_almost_equal(new_halo_positions_y[0], source_y_nosubs, 4)

        for ox, nx in zip(original_halo_positions_x[1:], new_halo_positions_x[1:]):
            npt.assert_equal(True, ox != nx)
        for oy, ny in zip(original_halo_positions_y[1:], new_halo_positions_y[1:]):
            npt.assert_equal(True, oy != ny)

        quad_lens.initialize(data, include_substructure=True)
        source_x, source_y = quad_lens.source_centroid_x, quad_lens.source_centroid_y
        lensmodel, kw = quad_lens.get_lensmodel()
        betax, betay = lensmodel.ray_shooting(self.x, self.y, kw)
        source_x_raytracing = np.mean(betax)
        source_y_raytracing = np.mean(betay)

        npt.assert_almost_equal(source_x_raytracing, source_x)
        npt.assert_almost_equal(source_y_raytracing, source_y)

    def test_get_smooth_lens(self):

        lens_system_smooth = self.quad_lens.get_smooth_lens_system()

        l, kw = lens_system_smooth.get_lensmodel()
        npt.assert_equal(len(kw), 3)

        lens_system_smooth.update_source_centroid(0.5, 1.)
        lens_system_smooth = lens_system_smooth.get_smooth_lens_system()
        l, kw = lens_system_smooth.get_lensmodel()
        npt.assert_equal(len(kw), 3)
        npt.assert_equal(lens_system_smooth.source_centroid_x, 0.5)

    def test_addRealization(self):

        lens_system = QuadLensSystem.addRealization(self.quad_lens_simple, self.realization)
        lm, kw = lens_system.get_lensmodel()
        npt.assert_equal(len(kw), len(self.kwargs))

    def test_update_source(self):

        self.quad_lens.update_source_centroid(1., 0.)
        source_x, source_y = self.quad_lens.source_centroid_x, self.quad_lens.source_centroid_y
        npt.assert_equal(source_x, 1.)
        npt.assert_equal(source_y, 0.)

    def test_quasar_magnification(self):

        data = LensedQuasar(self.x, self.y, np.ones_like(self.x))
        kwargs_source = {'source_fwhm_pc': 20.}
        background_quasar = Quasar(kwargs_source)

        lensmodel, kwargs_lens = self.quad_lens.get_lensmodel()

        npt.assert_raises(Exception, self.quad_lens.quasar_magnification, self.x, self.y, background_quasar,
                                                           lensmodel, kwargs_lens)

        self.quad_lens.initialize(data_to_fit=data, include_substructure=True)
        lensmodel, kwargs_lens = self.quad_lens.get_lensmodel()

        mag_slow, _ = self.quad_lens.quasar_magnification(self.x, self.y, background_quasar,
                                                           lensmodel, kwargs_lens, normed=True)

        mag_adaptive_slow, _ = self.quad_lens.quasar_magnification(self.x, self.y, background_quasar,
                                                           lensmodel, kwargs_lens, adaptive=True, normed=False)
        mag_adaptive_slow *= np.max(mag_adaptive_slow) ** -1

        mag_adaptive_fast, _ = self.quad_lens.quasar_magnification(self.x, self.y, background_quasar,
                                                                lensmodel, kwargs_lens, adaptive=True,
                                                                grid_axis_ratio=0.25, normed=False)
        mag_adaptive_fast *= np.max(mag_adaptive_fast) ** -1

        mag_point_source, _ = self.quad_lens.quasar_magnification(self.x, self.y, background_quasar,
                                                                   lensmodel, kwargs_lens, adaptive=True,
                                                                   grid_axis_ratio=0.25, normed=True,
                                                                  point_source=True)

        for mag_p, mag_s, mag_as, mag_a in zip(mag_point_source, mag_slow, mag_adaptive_slow, mag_adaptive_fast):
            npt.assert_almost_equal(mag_s, mag_p, 3)
            npt.assert_almost_equal(mag_as, mag_p, 3)
            npt.assert_almost_equal(mag_a, mag_p, 3)

if __name__ == '__main__':

    pytest.main()
