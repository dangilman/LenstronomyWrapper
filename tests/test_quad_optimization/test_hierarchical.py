import numpy.testing as npt
import pytest
from pyHalo.preset_models import CDM
from pyHalo.single_realization import realization_at_z
import numpy as np
from copy import deepcopy
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.LensComponents.SIS import SISsatellite
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.Optimization.quad_optimization.hierarchical import HierarchicalOptimization
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths_system
from lenstronomywrapper.LensSystem.LensComponents.multipole import Multipole

class TestHierarchicalOptimization(object):

    def setup(self):

        self.x, self.y = np.array([ 1.272,  0.306, -1.152, -0.384]), np.array([ 0.156, -1.092, -0.636, 1.026])
        self.m = np.array([0.96, 0.976, 1., 0.65])
        self.zlens, self.zsource = 0.45, 1.69
        self.zsatellite = 0.78
        self._realization = CDM(self.zlens, self.zsource, kwargs_model_other={'LOS_normalization': 1.})
        self.data = LensedQuasar(self.x, self.y, self.m)

    def test_free_shear_powerlaw(self):

        opt_routine = 'free_shear_powerlaw'

        constrain_params = None
        masscut_low = [6.7, 6.7]
        masscut_mid = [7.7, 7.5]
        rmax_foreground_low = [0.3, 0.15]
        rmax_background_low = [0.1, 0.1]
        rmax_foreground_mid = [100, 0.3]
        rmax_background_mid = [0.3, 0.3]

        for settings_class, decimals, masscut_low, masscut_mid, rmaxforelow, rmaxbacklow, rmaxforemid, rmaxbackmid in \
            zip(['default', 'default_CDM'], [5, 2], masscut_low, masscut_mid, rmax_foreground_low, rmax_background_low,
                                                          rmax_foreground_mid, rmax_background_mid):

            realization = deepcopy(self._realization)
            print('halos init: ', len(realization.halos))
            kwargs_epl_shear = [{'theta_E': 1., 'center_x': 0., 'center_y': 0., 'e1': 0.3, 'e2': -0.1, 'gamma': 2.},
                                {'gamma1': 0.05, 'gamma2': 0.}]
            kwargs_satellite = [{'theta_E': 0.35, 'center_x': -2.27, 'center_y': 1.98}]

            main_deflector = PowerLawShear(self.zlens, kwargs_epl_shear)
            satellite = SISsatellite(self.zsatellite, kwargs_satellite)
            macromodel = MacroLensModel([main_deflector, satellite])

            lens_system = QuadLensSystem.shift_background_auto(self.data, macromodel,
                                                                    self.zsource, realization,
                                                                    opt_routine=opt_routine,
                                                                    constrain_params=constrain_params)

            optimizer = HierarchicalOptimization(lens_system=lens_system, n_particles=30,
                                                 settings_class=settings_class)
            kwargs_lens_final, lens_model_full, kwargs_return = optimizer.\
                optimize(self.data, param_class_name=opt_routine, constrain_params=constrain_params,
                         verbose=False)

            betax, betay = lens_model_full.ray_shooting(self.x, self.y, kwargs_lens_final)
            betax, betay = np.mean(betax), np.mean(betay)
            npt.assert_almost_equal(betax, lens_system.source_centroid_x, decimals)
            npt.assert_almost_equal(betay, lens_system.source_centroid_y, decimals)

            unique_redshifts = np.unique(lens_model_full.redshift_list)
            distances = [lens_system.pyhalo_cosmology.D_C_transverse(zi) for zi in unique_redshifts]
            ray_paths_x, ray_paths_y = interpolate_ray_paths_system(self.x, self.y, lens_system,
                                                                    realization=lens_system.realization)

            for zi, di in zip(unique_redshifts, distances):
                real, _ = realization_at_z(lens_system.realization, zi)

                for halo in real.halos:
                    if halo.mass < masscut_low:
                        for i in range(0, 4):
                            xray, yray = ray_paths_x[i](di), ray_paths_y[i](di)
                            dr = np.hypot(xray - halo.x, yray - halo.y)
                            if halo.z <= self.zlens and dr < rmax_foreground_low:
                                break
                            elif halo.z > self.zlens and dr < rmax_background_low:
                                break
                        else:
                            npt.assert_equal(True, False)
                    elif halo.mass < masscut_mid:
                        for i in range(0, 4):
                            xray, yray = ray_paths_x[i](di), ray_paths_y[i](di)
                            dr = np.hypot(xray - halo.x, yray - halo.y)
                            if halo.z <= self.zlens and dr < rmax_foreground_mid:
                                break
                            elif halo.z > self.zlens and dr < rmax_background_mid:
                                break
                        else:
                            npt.assert_equal(True, False)

    def test_free_shear_powerlaw_multipole(self):

        opt_routine = 'free_shear_powerlaw_multipole'
        constrain_params = None
        masscut_low = [6.7, 6.7]
        masscut_mid = [7.7, 7.5]
        rmax_foreground_low = [0.3, 0.15]
        rmax_background_low = [0.1, 0.1]
        rmax_foreground_mid = [100, 0.3]
        rmax_background_mid = [0.3, 0.3]

        for settings_class, decimals, masscut_low, masscut_mid, rmaxforelow, rmaxbacklow, rmaxforemid, rmaxbackmid in \
            zip(['default', 'default_CDM'], [5, 2], masscut_low, masscut_mid, rmax_foreground_low, rmax_background_low,
                                                          rmax_foreground_mid, rmax_background_mid):

            realization = deepcopy(self._realization)
            kwargs_epl_shear = [{'theta_E': 1., 'center_x': 0., 'center_y': 0., 'e1': 0.3, 'e2': -0.1, 'gamma': 2.},
                                {'gamma1': 0.05, 'gamma2': 0.}]
            kwargs_satellite = [{'theta_E': 0.35, 'center_x': -2.27, 'center_y': 1.98}]
            kwargs_multipole = [{'m': 4., 'a_m': 0.005, 'phi_m': 0., 'center_x': 0., 'center_y': 0.}]

            main_deflector = PowerLawShear(self.zlens, kwargs_epl_shear)
            satellite = SISsatellite(self.zsatellite, kwargs_satellite)
            multipole = Multipole(self.zlens, kwargs_multipole)
            macromodel = MacroLensModel([main_deflector, multipole, satellite])

            lens_system = QuadLensSystem.shift_background_auto(self.data, macromodel,
                                                                    self.zsource, realization,
                                                                    opt_routine=opt_routine,
                                                                    constrain_params=constrain_params)

            optimizer = HierarchicalOptimization(lens_system=lens_system, n_particles=30,
                                                 settings_class=settings_class)
            kwargs_lens_final, lens_model_full, kwargs_return = optimizer.\
                optimize(self.data, param_class_name=opt_routine, constrain_params=constrain_params,
                         verbose=False)
            npt.assert_equal(lens_model_full.lens_model_list[2], 'MULTIPOLE')
            am = kwargs_lens_final[2]['a_m']
            npt.assert_equal(am, 0.005)

            betax, betay = lens_model_full.ray_shooting(self.x, self.y, kwargs_lens_final)
            betax, betay = np.mean(betax), np.mean(betay)
            npt.assert_almost_equal(betax, lens_system.source_centroid_x, decimals)
            npt.assert_almost_equal(betay, lens_system.source_centroid_y, decimals)

            unique_redshifts = np.unique(lens_model_full.redshift_list)
            distances = [lens_system.pyhalo_cosmology.D_C_transverse(zi) for zi in unique_redshifts]
            ray_paths_x, ray_paths_y = interpolate_ray_paths_system(self.x, self.y, lens_system,
                                                                    realization=lens_system.realization)

            for zi, di in zip(unique_redshifts, distances):
                real, _ = realization_at_z(lens_system.realization, zi)

                for halo in real.halos:
                    if halo.mass < masscut_low:
                        for i in range(0, 4):
                            xray, yray = ray_paths_x[i](di), ray_paths_y[i](di)
                            dr = np.hypot(xray - halo.x, yray - halo.y)
                            if halo.z <= self.zlens and dr < rmax_foreground_low:
                                break
                            elif halo.z > self.zlens and dr < rmax_background_low:
                                break
                        else:
                            npt.assert_equal(True, False)
                    elif halo.mass < masscut_mid:
                        for i in range(0, 4):
                            xray, yray = ray_paths_x[i](di), ray_paths_y[i](di)
                            dr = np.hypot(xray - halo.x, yray - halo.y)
                            if halo.z <= self.zlens and dr < rmax_foreground_mid:
                                break
                            elif halo.z > self.zlens and dr < rmax_background_mid:
                                break
                        else:
                            npt.assert_equal(True, False)

    def test_fixed_shear_powerlaw_multipole(self):

        opt_routine = 'fixed_shear_powerlaw_multipole'
        constrain_params = {'shear': 0.06}
        masscut_low = [6.7, 6.7]
        masscut_mid = [7.7, 7.5]
        rmax_foreground_low = [0.3, 0.15]
        rmax_background_low = [0.1, 0.1]
        rmax_foreground_mid = [100, 0.3]
        rmax_background_mid = [0.3, 0.3]

        for settings_class, decimals, masscut_low, masscut_mid, rmaxforelow, rmaxbacklow, rmaxforemid, rmaxbackmid in \
            zip(['default', 'default_CDM'], [5, 2], masscut_low, masscut_mid, rmax_foreground_low, rmax_background_low,
                                                          rmax_foreground_mid, rmax_background_mid):

            realization = deepcopy(self._realization)
            kwargs_epl_shear = [{'theta_E': 1., 'center_x': 0., 'center_y': 0., 'e1': 0.3, 'e2': -0.1, 'gamma': 2.},
                                {'gamma1': 0.05, 'gamma2': 0.}]
            kwargs_satellite = [{'theta_E': 0.35, 'center_x': -2.27, 'center_y': 1.98}]
            kwargs_multipole = [{'m': 4., 'a_m': 0.005, 'phi_m': 0., 'center_x': 0., 'center_y': 0.}]

            main_deflector = PowerLawShear(self.zlens, kwargs_epl_shear)
            satellite = SISsatellite(self.zsatellite, kwargs_satellite)
            multipole = Multipole(self.zlens, kwargs_multipole)
            macromodel = MacroLensModel([main_deflector, multipole, satellite])

            lens_system = QuadLensSystem.shift_background_auto(self.data, macromodel,
                                                                    self.zsource, realization,
                                                                    opt_routine=opt_routine,
                                                                    constrain_params=constrain_params)

            optimizer = HierarchicalOptimization(lens_system=lens_system, n_particles=30,
                                                 settings_class=settings_class)
            kwargs_lens_final, lens_model_full, kwargs_return = optimizer.\
                optimize(self.data, param_class_name=opt_routine, constrain_params=constrain_params,
                         verbose=False)
            npt.assert_equal(lens_model_full.lens_model_list[2], 'MULTIPOLE')
            am = kwargs_lens_final[2]['a_m']
            npt.assert_equal(am, 0.005)
            shear = np.hypot(kwargs_lens_final[1]['gamma1'], kwargs_lens_final[1]['gamma2'])
            npt.assert_almost_equal(shear, constrain_params['shear'])

            betax, betay = lens_model_full.ray_shooting(self.x, self.y, kwargs_lens_final)
            betax, betay = np.mean(betax), np.mean(betay)
            npt.assert_almost_equal(betax, lens_system.source_centroid_x, decimals)
            npt.assert_almost_equal(betay, lens_system.source_centroid_y, decimals)

            unique_redshifts = np.unique(lens_model_full.redshift_list)
            distances = [lens_system.pyhalo_cosmology.D_C_transverse(zi) for zi in unique_redshifts]
            ray_paths_x, ray_paths_y = interpolate_ray_paths_system(self.x, self.y, lens_system,
                                                                    realization=lens_system.realization)

            for zi, di in zip(unique_redshifts, distances):
                real, _ = realization_at_z(lens_system.realization, zi)

                for halo in real.halos:
                    if halo.mass < masscut_low:
                        for i in range(0, 4):
                            xray, yray = ray_paths_x[i](di), ray_paths_y[i](di)
                            dr = np.hypot(xray - halo.x, yray - halo.y)
                            if halo.z <= self.zlens and dr < rmax_foreground_low:
                                break
                            elif halo.z > self.zlens and dr < rmax_background_low:
                                break
                        else:
                            npt.assert_equal(True, False)
                    elif halo.mass < masscut_mid:
                        for i in range(0, 4):
                            xray, yray = ray_paths_x[i](di), ray_paths_y[i](di)
                            dr = np.hypot(xray - halo.x, yray - halo.y)
                            if halo.z <= self.zlens and dr < rmax_foreground_mid:
                                break
                            elif halo.z > self.zlens and dr < rmax_background_mid:
                                break
                        else:
                            npt.assert_equal(True, False)

    def test_fixed_shear_powerlaw(self):

        opt_routine = 'fixed_shear_powerlaw'
        constrain_params = {'shear': 0.06}
        masscut_low = [6.7, 6.7]
        masscut_mid = [7.7, 7.5]
        rmax_foreground_low = [0.3, 0.15]
        rmax_background_low = [0.1, 0.1]
        rmax_foreground_mid = [100, 0.3]
        rmax_background_mid = [0.3, 0.3]

        for settings_class, decimals, masscut_low, masscut_mid, rmaxforelow, rmaxbacklow, rmaxforemid, rmaxbackmid in \
            zip(['default', 'default_CDM'], [5, 2], masscut_low, masscut_mid, rmax_foreground_low, rmax_background_low,
                                                          rmax_foreground_mid, rmax_background_mid):

            realization = deepcopy(self._realization)
            kwargs_epl_shear = [{'theta_E': 1., 'center_x': 0., 'center_y': 0., 'e1': 0.3, 'e2': -0.1, 'gamma': 2.},
                                {'gamma1': 0.05, 'gamma2': 0.}]
            kwargs_satellite = [{'theta_E': 0.35, 'center_x': -2.27, 'center_y': 1.98}]

            main_deflector = PowerLawShear(self.zlens, kwargs_epl_shear)
            satellite = SISsatellite(self.zsatellite, kwargs_satellite)

            macromodel = MacroLensModel([main_deflector, satellite])

            lens_system = QuadLensSystem.shift_background_auto(self.data, macromodel,
                                                                    self.zsource, realization,
                                                                    opt_routine=opt_routine,
                                                                    constrain_params=constrain_params)

            optimizer = HierarchicalOptimization(lens_system=lens_system, n_particles=30,
                                                 settings_class=settings_class)
            kwargs_lens_final, lens_model_full, kwargs_return = optimizer.\
                optimize(self.data, param_class_name=opt_routine, constrain_params=constrain_params,
                         verbose=False)
            npt.assert_equal(lens_model_full.lens_model_list[2], 'SIS')
            shear = np.hypot(kwargs_lens_final[1]['gamma1'], kwargs_lens_final[1]['gamma2'])
            npt.assert_almost_equal(shear, constrain_params['shear'])

            betax, betay = lens_model_full.ray_shooting(self.x, self.y, kwargs_lens_final)
            betax, betay = np.mean(betax), np.mean(betay)
            npt.assert_almost_equal(betax, lens_system.source_centroid_x, decimals)
            npt.assert_almost_equal(betay, lens_system.source_centroid_y, decimals)

            unique_redshifts = np.unique(lens_model_full.redshift_list)
            distances = [lens_system.pyhalo_cosmology.D_C_transverse(zi) for zi in unique_redshifts]
            ray_paths_x, ray_paths_y = interpolate_ray_paths_system(self.x, self.y, lens_system,
                                                                    realization=lens_system.realization)

            for zi, di in zip(unique_redshifts, distances):
                real, _ = realization_at_z(lens_system.realization, zi)

                for halo in real.halos:
                    if halo.mass < masscut_low:
                        for i in range(0, 4):
                            xray, yray = ray_paths_x[i](di), ray_paths_y[i](di)
                            dr = np.hypot(xray - halo.x, yray - halo.y)
                            if halo.z <= self.zlens and dr < rmax_foreground_low:
                                break
                            elif halo.z > self.zlens and dr < rmax_background_low:
                                break
                        else:
                            npt.assert_equal(True, False)
                    elif halo.mass < masscut_mid:
                        for i in range(0, 4):
                            xray, yray = ray_paths_x[i](di), ray_paths_y[i](di)
                            dr = np.hypot(xray - halo.x, yray - halo.y)
                            if halo.z <= self.zlens and dr < rmax_foreground_mid:
                                break
                            elif halo.z > self.zlens and dr < rmax_background_mid:
                                break
                        else:
                            npt.assert_equal(True, False)

if __name__ == '__main__':

    pytest.main()
