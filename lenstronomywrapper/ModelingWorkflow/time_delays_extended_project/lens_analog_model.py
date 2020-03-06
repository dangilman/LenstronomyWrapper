from lenstronomywrapper.LensSystem.arc_quad_lens import ArcQuadLensSystem
from lenstronomywrapper.LensSystem.BackgroundSource.sersic_source import SersicSource
from lenstronomywrapper.LensSystem.LensLight.sersic_lens import SersicLens
from lenstronomywrapper.LensData.arc_plus_quad import ArcPlusQuad
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.LensSystem.LensComponents.satellite import SISsatellite
from lenstronomywrapper.LensSystem.light_model import LightModel
from lenstronomy.Plots.model_plot import ModelPlot
import random
from lenstronomywrapper.Utilities.data_util import approx_theta_E
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar
from lenstronomywrapper.Utilities.data_util import write_data_to_file
from lenstronomywrapper.LensSystem.LensSystemExtensions.chain_post_processing import ChainPostProcess
from lenstronomywrapper.LensSystem.LensSystemExtensions.lens_maps import ResidualLensMaps
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths
import numpy as np
from pyHalo.pyhalo import pyHalo
import os

import matplotlib.pyplot as plt

class AnalogModel(object):

    def __init__(self, lens_class_instance, kwargs_cosmology,
                 kwargs_quasar=None, makeplots=False, pyhalo=None,
                 free_convergence=False):

        if kwargs_quasar is None:
            kwargs_quasar = {'center_x': 0, 'center_y': 0, 'source_fwhm_pc': 25}

        self.free_convergence = free_convergence
        self._makeplots = makeplots
        self.lens = lens_class_instance
        self.kwargs_cosmology = kwargs_cosmology
        self.zlens, self.zsource = lens_class_instance.zlens, lens_class_instance.zsrc
        if pyhalo is None:
            pyhalo = pyHalo(self.zlens, self.zsource, cosmology_kwargs=self.kwargs_cosmology)
        self.pyhalo = pyhalo
        self.kwargs_quasar = kwargs_quasar

    def background_quasar_class(self):

        return Quasar(self.kwargs_quasar)

    def satellite_props(self, rein):

        amp = 600 * (rein / 0.3) ** 1.
        r = 0.2 * (rein / 0.3) ** 1.
        n = 3.
        amp = max(amp, 1200)
        r = max(r, 0.25)
        n = max(n, 2.5)

        return amp, r, n

    def sample_source(self, amp_macro):

        amp = abs(np.random.normal(amp_macro, 0.5*amp_macro))
        rein_eff = 0.5 * (amp/600)
        _, r, n = self.satellite_props(rein_eff)
        r *= 0.5
        return amp, r, n

    def gen_realization(self, realization_type, realization_kwargs):

        realization = self.pyhalo.render(realization_type, realization_kwargs)[0]
        return realization

    def flux_anomaly(self, f1, f2):

        return f1 - f2

    def time_anomaly(self, t1, t2):

        return t1 - t2

    def run(self, save_name_path, N_start, N, realization, arrival_time_sigma,
            image_positions_sigma, gamma_prior_scale, time_delay_likelihood, fix_D_dt, fit_smooth_kwargs,
            window_size, exp_time, background_rms, shapelet_nmax):

        observed_lens, modeled_lens, normalized_residuals, residual_convergence = [], [], [], []
        residual_mean_kappa = []
        time_delay_surface_true = []
        time_delay_surface_modeled = []

        for n in range(0, N):

            tbaseline, f, t, tdelay_model, macro_params, kw_fit, kw_setup= self.run_once(realization,
                                                                arrival_time_sigma,
                                                                image_positions_sigma,
                                                                gamma_prior_scale,
                                                                time_delay_likelihood,
                                                                fix_D_dt, window_size,
                                                                exp_time, background_rms,
                                                                shapelet_nmax,
                                                                **fit_smooth_kwargs)

            #h0_inf = kw_fit['H0_inferred']

            # info = [self.zlens, self.zsource, self.lens.x[0], self.lens.x[1], self.lens.x[2], self.lens.x[3],
            #         self.lens.y[0], self.lens.y[1], self.lens.y[2], self.lens.y[3]]

            observed_lens.append(kw_fit['observed_lens'])
            modeled_lens.append(kw_fit['modeled_lens'])
            normalized_residuals.append(kw_fit['normalized_residuals'])
            residual_convergence.append(kw_fit['residual_convergence'])
            residual_mean_kappa.append(kw_fit['mean_kappa'])
            time_delay_surface_true.append(kw_fit['time_delay_surface_true'])
            time_delay_surface_modeled.append(kw_fit['time_delay_surface_modeled'])

            if n == 0:
                baseline = tbaseline
                flux_anomalies = f
                time_anomalies = t
                # time_anomalies_geo = tgeo
                # time_anomalies_grav = tgrav
                #h0_inferred = h0_inf.ravel()
                time_delays_model = tdelay_model
                ddt_inferred = kw_fit['D_dt_samples'].ravel()
                macromodel_parameters = macro_params
                tsigma = arrival_time_sigma

            else:
                baseline = np.vstack((baseline, tbaseline))
                flux_anomalies = np.vstack((flux_anomalies, f))
                time_anomalies = np.vstack((time_anomalies, t))
                time_delays_model = np.vstack((time_delays_model, tdelay_model))
                # time_anomalies_geo = np.vstack((time_anomalies_geo, tgeo))
                # time_anomalies_grav = np.vstack((time_anomalies_grav, tgrav))
                ddt_inferred = np.append(ddt_inferred, kw_fit['D_dt_samples'].ravel())
                #h0_inferred = np.append(h0_inferred, h0_inf.ravel()).flatten()
                macromodel_parameters = np.vstack((macromodel_parameters, macro_params))
                tsigma = np.vstack((tsigma, arrival_time_sigma))

        fnames = ['tbaseline_', 'flux_anomaly_', 'time_anomaly_', 'time_delays_', 'ddt_inferred', 'macroparams_',
                  'time_delay_sigma_', 'kappares_']

        arrays = [baseline, flux_anomalies, time_anomalies, time_delays_model,
                  np.array(ddt_inferred), macromodel_parameters, tsigma, np.array(residual_mean_kappa)
                  ]

        for fname, arr in zip(fnames, arrays):
            write_data_to_file(save_name_path + fname + str(N_start) + '.txt', arr)

        for i in range(0, len(observed_lens)):

            np.savetxt(save_name_path + 'observed_' + str(N_start+i)+'.txt', X=observed_lens[i])
            np.savetxt(save_name_path + 'modeled_' + str(N_start+i)+'.txt', X=modeled_lens[i])
            np.savetxt(save_name_path + 'residuals_' + str(N_start + i) + '.txt', X=normalized_residuals[i])
            np.savetxt(save_name_path + 'kappa_' + str(N_start + i) + '.txt', X=residual_convergence[i])
            np.savetxt(save_name_path + 'tdelaymodeled_' + str(N_start + i) + '.txt', X=time_delay_surface_modeled[i])
            np.savetxt(save_name_path + 'tdelaytrue_' + str(N_start + i) + '.txt', X=time_delay_surface_true[i])

        return flux_anomalies, baseline, time_anomalies, ddt_inferred

    def save_append(self, filename, array_to_save):

        if os.path.exists(filename):
            x = np.loadtxt(filename)
            try:
                array_to_save = np.vstack((x, array_to_save))
            except:
                array_to_save = np.append(x, array_to_save)

        np.savetxt(filename, X=array_to_save, fmt='%.5f')

    def run_once(self, realization, arrival_time_sigma, image_sigma, gamma_prior_scale,
            time_delay_likelihood, fix_D_dt, window_size, exp_time, background_rms, shapelet_nmax,
                 **fit_smooth_kwargs):

        lens_system, data_class, return_kwargs_setup, kwargs_data_setup = \
            self.model_setup(realization, arrival_time_sigma, image_sigma, gamma_prior_scale, window_size, exp_time, background_rms, shapelet_nmax)

        return_kwargs_fit, kwargs_data_fit = self.fit_smooth(lens_system, data_class,
                                                             time_delay_likelihood, fix_D_dt, window_size, **fit_smooth_kwargs)

        macromodel_params = np.round(return_kwargs_fit['kwargs_lens_macro_fit'], 5)
        srcx, srcy = np.round(kwargs_data_fit['source_x'], 4), np.round(kwargs_data_fit['source_y'], 4)

        macromodel_params = np.hstack((macromodel_params, np.array([srcx, srcy]).reshape(len(srcx), 2)))

        key = 'flux_ratios'
        flux_anomaly = np.round(self.flux_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)

        key = 'time_delays'
        time_anomaly = np.round(self.time_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)
        time_delays_model = kwargs_data_fit[key]
        time_delay_baseline = kwargs_data_setup[key]

        return time_delay_baseline, flux_anomaly, time_anomaly, time_delays_model, \
               macromodel_params, return_kwargs_fit, return_kwargs_setup

    def compute_observables(self, lens_system):

        magnifications, _ = lens_system.quasar_magnification(self.lens.x, self.lens.y, normed=False)
        # lens_system_quad.plot_images(data_to_fit.x, data_to_fit.y)
        lensModel, kwargs_lens = lens_system.get_lensmodel()
        dtgeo, dtgrav = lensModel.lens_model.geo_shapiro_delay(self.lens.x, self.lens.y, kwargs_lens)
        arrival_times = dtgeo + dtgrav

        return magnifications, arrival_times, dtgeo, dtgrav

    def model_setup(self, realization, arrival_time_sigma, image_sigma, gamma_prior_scale,
                    window_size, exp_time, background_rms, shapelet_nmax):

        data_to_fit = LensedQuasar(self.lens.x, self.lens.y, self.lens.m)
        background_quasar = self.background_quasar_class()

        kwargs_macro = self.lens.kwargs_lens_init

        deflector_list = [PowerLawShear(self.zlens, kwargs_macro)]

        amp, rsersic, nsersic = self.satellite_props(approx_theta_E(self.lens.x, self.lens.y))

        kwargs_sersic_light = self.lens.kwargs_lens_light

        kwargs_sersic_source = self.lens.kwargs_source_light
        light_model_list = [SersicLens(kwargs_sersic_light, concentric_with_model=0)]

        r_sat_max = 0

        if self.lens.has_satellite:

            n_satellites = len(self.lens.satellite_mass_model)

            n_max = min(n_satellites, 2)

            for n in range(0, n_max):

                rein_sat = self.lens.satellite_kwargs[n]['theta_E']
                xsat = self.lens.satellite_kwargs[n]['center_x']
                ysat = self.lens.satellite_kwargs[n]['center_y']
                r_sat = np.sqrt(xsat ** 2 + ysat ** 2)
                r_sat_max = max(r_sat, r_sat_max)
                satellite_redshift = self.lens.satellite_redshift[n]
                prior_galaxy = [['theta_E', rein_sat, 0.1 * rein_sat], ['center_x', xsat, 0.05],
                          ['center_y', ysat, 0.05]]
                kwargs_init = [self.lens.satellite_kwargs[n]]

                satellite_galaxy = SISsatellite(satellite_redshift, kwargs_init=kwargs_init,
                                            prior=prior_galaxy)

                deflector_list += [satellite_galaxy]

                if self.lens.kwargs_satellite_light[n] is not None:
                    kwargs_light_satellite = [self.lens.kwargs_satellite_light[n]]

                    prior_sat_light = [['amp', amp, amp * 0.2],
                                   ['center_x', xsat, 0.05],
                                   ['center_y', ysat, 0.05],
                                   ['R_sersic', kwargs_light_satellite[0]['R_sersic'], 0.2 * kwargs_light_satellite[0]['R_sersic']],
                                   ['n_sersic', kwargs_light_satellite[0]['n_sersic'], 0.2 * kwargs_light_satellite[0]['n_sersic']]]


                    light_model_list += [SersicLens(kwargs_light_satellite, concentric_with_model=n+1,
                                                prior=prior_sat_light)]

        macromodel = MacroLensModel(deflector_list)

        if realization is not None:
            lens_system_quad = QuadLensSystem.shift_background_auto(data_to_fit, macromodel,
                                                                    self.zsource, background_quasar, realization,
                                                                    self.pyhalo._cosmology)

        else:
            lens_system_quad = QuadLensSystem(macromodel, self.zsource, background_quasar, None,
                                          pyhalo_cosmology=self.pyhalo._cosmology)

        lens_system_quad.initialize(data_to_fit, include_substructure=True, verbose=True,
                                    kwargs_optimizer={'particle_swarm': False})
        magnifications, arrival_times, dtgeo, dtgrav = self.compute_observables(lens_system_quad)

        arrival_time_uncertainties = []
        for t, delta_t in zip(arrival_times[1:]-arrival_times[0], arrival_time_sigma):
            arrival_time_uncertainties.append(abs(t*delta_t))

        source_model_list = [SersicSource(kwargs_sersic_source, concentric_with_source=True)]
        source_x, source_y = lens_system_quad.source_centroid_x, lens_system_quad.source_centroid_y
        if self.lens.identifier == 'lens0408':
            kwargs_sersic_source_2 = [{'amp': 1500, 'R_sersic': 0.1, 'n_sersic': 4., 'center_x': source_x - 0.6,
                                       'center_y': source_y + 0.8,
                                       'e1': 0.2, 'e2': -0.05}]
            kwargs_sersic_source_3 = [{'amp': 1000, 'R_sersic': 0.2, 'n_sersic': 2., 'center_x': source_x + 0.35,
              'center_y': source_y + 1.3,
              'e1': 0.01, 'e2': -0.01}]
            source_model_list += [SersicSource(kwargs_sersic_source_2),
                                  SersicSource(kwargs_sersic_source_3)]

        if shapelet_nmax is not None:
            kwargs_source_shapelet = [{'amp': 200., 'beta': 0.05,
                                       'n_max': int(shapelet_nmax),
                                       'center_x': 0., 'center_y': 0.}]
            source_model_list += [SersicSource(kwargs_source_shapelet, concentric_with_source=True)]

        if window_size is None:

            window_size_macro = 2. * lens_system_quad.macromodel.kwargs[0]['theta_E']
            if self.lens.has_satellite:
                if r_sat_max > window_size_macro/2:
                    window_size = r_sat_max * 1.2
                else:
                    window_size = window_size_macro
            else:
                window_size = window_size_macro

        light_model = LightModel(light_model_list)
        source_model = LightModel(source_model_list)
        lens_system = ArcQuadLensSystem.fromQuad(lens_system_quad, light_model,
                                                 source_model)

        data_kwargs = {'psf_type': 'GAUSSIAN', 'window_size': 2*window_size, 'deltaPix': 0.05, 'fwhm': 0.1,
                       'exp_time': exp_time, 'background_rms': background_rms}

        data_class = ArcPlusQuad(data_to_fit.x, data_to_fit.y, magnifications, lens_system, arrival_times,
                           arrival_time_uncertainties, image_sigma, data_kwargs=data_kwargs, no_bkg=False, noiseless=False,
                                 normed_magnifications=False)

        imaging_data = data_class.get_lensed_image()

        if realization is not None:
            try:
                log_mlow = realization.log_mlow
            except:
                print('realization instance has no attribute log_mlow; defaulting to 6.7')
                log_mlow = 6.7

            halo_model_names, redshift_list_halos, kwargs_halos, _ = \
                realization.lensing_quantities(log_mlow, log_mlow)
        else:
            halo_model_names, redshift_list_halos, kwargs_halos = [], [], []

        return_kwargs = {'imaging_data': imaging_data,
                         'kwargs_lens_macro': lens_system.macromodel.kwargs,
                         'lens_model_list_macro': lens_system.macromodel.lens_model_list,
                         'redshift_list_macro': lens_system.macromodel.redshift_list,
                         'lens_model_list_halos': halo_model_names,
                         'redshift_list_halos': redshift_list_halos,
                         'kwargs_lens_halos': kwargs_halos}

        return_kwargs_data = {'flux_ratios': magnifications[1:]/magnifications[0],
                              'time_delays': arrival_times[1:]-arrival_times[0],
                              'geo_delay': dtgeo[1:] - dtgeo[0],
                              'grav_delay': dtgrav[1:] - dtgrav[0],
                              'arrival_time_sigma': arrival_time_sigma}

        return lens_system, data_class, return_kwargs, return_kwargs_data

    def fit_smooth(self, arc_quad_lens, data, time_delay_likelihood, fix_D_dt, window_size,
                   n_particles=100, n_iterations=200, n_run=100, n_burn=600, walkerRatio=4):

        lensModel_full, kwargs_lens_full = arc_quad_lens.get_lensmodel()
        lens_system_simple = arc_quad_lens.get_smooth_lens_system()

        pso_kwargs = {'sigma_scale': 1.0, 'n_particles': n_particles, 'n_iterations': n_iterations}
        mcmc_kwargs = {'n_burn': n_burn, 'n_run': n_run, 'walkerRatio': walkerRatio, 'sigma_scale': 0.1}

        chain_list, kwargs_result, kwargs_model, multi_band_list, kwargs_special, param_class = \
            lens_system_simple.fit(data, pso_kwargs, mcmc_kwargs, time_delay_likelihood=time_delay_likelihood,
                                   fix_D_dt=fix_D_dt)

        lensModel, kwargs_lens = lens_system_simple.get_lensmodel()
        modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat")

        observed_lens = modelPlot._select_band(0)._data
        modeled_lens = modelPlot._select_band(0)._model
        normalized_residuals = modelPlot._select_band(0)._norm_residuals

        residual_maps = ResidualLensMaps(lensModel_full, lensModel, kwargs_lens_full, kwargs_lens)
        kappa = residual_maps.convergence(window_size, 200)

        tdelay_map_full, tdelay_map_simple = residual_maps.time_delay_surface_12(window_size, 200,
                                           self.lens.x[0], self.lens.y[0])

        # tdelay_res_geo, tdelay_res_grav = residual_maps.time_delay_surface_geoshapiro(window_size, 200,
        #                                    self.lens.x[0], self.lens.y[0])
        # tdelay_res_map = tdelay_res_geo + tdelay_res_grav

        D_dt_true = lens_system_simple.lens_cosmo.D_dt

        n_keep = 100
        chain_samples = chain_list[1][1]
        nsamples = int(chain_samples[:,-1].shape[0])

        keep_integers = random.sample(range(0, nsamples-1), n_keep)

        chain_samples = chain_samples[keep_integers, :]

        lensModel, _ = lens_system_simple.get_lensmodel()
        chain_process = ChainPostProcess(lensModel, chain_samples, param_class,
                                         background_quasar=lens_system_simple.background_quasar)

        flux_ratios, source_x, source_y = chain_process.flux_ratios(self.lens.x, self.lens.y)
        arrival_times, arrival_times_geo, arrival_times_grav = chain_process.time_delays(self.lens.x, self.lens.y)
        macro_params = chain_process.macro_params()

        return_kwargs = {'D_dt_true': D_dt_true,
                         'kwargs_lens_macro_fit': macro_params, 'mean_kappa': np.mean(kappa),
                         'residual_convergence': kappa, 'time_delay_surface_true': tdelay_map_full,
                         'time_delay_surface_modeled': tdelay_map_simple,
                         'observed_lens': observed_lens, 'modeled_lens': modeled_lens, 'normalized_residuals': normalized_residuals,
                         'D_dt_samples': chain_samples[:,-1], 'source_x': lens_system_simple.source_centroid_x,
                         'source_y': lens_system_simple.source_centroid_y, 'zlens': self.zlens,
                         'zsource': self.zsource}

        return_kwargs_data = {'flux_ratios': flux_ratios,
                              'time_delays': arrival_times,
                              'geo_delay': arrival_times_geo,
                              'grav_delay': arrival_times_grav,
                              'source_x': source_x,
                              'source_y': source_y}

        return return_kwargs, return_kwargs_data





















