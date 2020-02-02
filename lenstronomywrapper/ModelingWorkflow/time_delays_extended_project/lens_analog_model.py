from lenstronomywrapper.LensSystem.arc_quad_lens import ArcQuadLensSystem
from lenstronomywrapper.LensSystem.BackgroundSource.sersic_source import SersicSource
from lenstronomywrapper.LensSystem.LensLight.sersic_lens import SersicLens
from lenstronomywrapper.LensData.arc_plus_quad import ArcPlusQuad
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.LensComponents.powerlawshearconvergence import PowerLawShearConvergence
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.LensSystem.LensComponents.satellite import SISsatellite
from lenstronomywrapper.LensSystem.light_model import LightModel
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomywrapper.Utilities.data_util import approx_theta_E
from lenstronomywrapper.Utilities.lensing_util import solve_H0_from_Ddt
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomywrapper.Utilities.data_util import write_data_to_file
from lenstronomywrapper.LensSystem.LensSystemExtensions.chain_post_processing import ChainPostProcess

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

        amp = 600 * (rein / 0.3) ** 0.9
        r = 0.25 * (rein / 0.3) ** 0.5
        n = 3.25 * (rein / 0.3) ** 0.2
        amp = max(amp, 1200)
        r = max(r, 0.25)
        n = max(n, 2.5)

        return abs(np.random.normal(amp, 0.1 * amp)), abs(np.random.normal(r, 0.1 * r)), abs(
            np.random.normal(n, 0.1 * n))

    def sample_source(self, amp_macro):

        amp = abs(np.random.normal(amp_macro, 0.5*amp_macro))
        rein_eff = 0.5 * (amp/600)**(1./0.9)
        _, r, n = self.satellite_props(rein_eff)
        r *= 0.5
        return abs(np.random.normal(amp, 0.1 * amp)), abs(np.random.normal(r, 0.25 * r)), abs(
            np.random.normal(n, 0.2 * n))

    def gen_realization(self, realization_type, realization_kwargs):

        realization = self.pyhalo.render(realization_type, realization_kwargs)[0]
        return realization

    def flux_anomaly(self, f1, f2):

        return (f1 - f2).reshape(1, 3)

    def time_anomaly(self, t1, t2):

        return np.array(t1 - t2).reshape(1, 3)

    def run(self, save_name_path, N_start, N, realization_type, realization_kwargs, arrival_time_sigma,
            image_positions_sigma, gamma_prior_scale, time_delay_likelihood, fix_D_dt, fit_smooth_kwargs,
            window_size):

        for n in range(0, N):
            tbaseline, f, t, tgeo, tgrav, macro_params, kw_fit, kw_setup= self.run_once(realization_type,
                                                                realization_kwargs,
                                                                arrival_time_sigma,
                                                                image_positions_sigma,
                                                                gamma_prior_scale,
                                                                time_delay_likelihood,
                                                                fix_D_dt, window_size,
                                                                **fit_smooth_kwargs)

            h0_inf = np.mean(kw_fit['H0_inferred'])
            h0_inf_sigma = np.std(kw_fit['H0_inferred'])

            info = [self.zlens, self.zsource, self.lens.x[0], self.lens.x[1], self.lens.x[2], self.lens.x[3],
                    self.lens.y[0], self.lens.y[1], self.lens.y[2], self.lens.y[3]]

            if n == 0:
                baseline = tbaseline
                flux_anomalies = f
                time_anomalies = t
                time_anomalies_geo = tgeo
                time_anomalies_grav = tgrav
                h0_inferred = h0_inf
                h0_sigma = h0_inf_sigma
                macromodel_parameters = macro_params

            else:
                baseline = np.vstack((baseline, tbaseline))
                flux_anomalies = np.vstack((flux_anomalies, f))
                time_anomalies = np.vstack((time_anomalies, t))
                time_anomalies_geo = np.vstack((time_anomalies_geo, tgeo))
                time_anomalies_grav = np.vstack((time_anomalies_grav, tgrav))
                h0_inferred = np.vstack((h0_inferred, h0_inf))
                h0_sigma = np.vstack((h0_sigma, h0_inf_sigma))
                macromodel_parameters = np.vstack((macromodel_parameters, macro_params))

        fnames = ['tbaseline_', 'flux_anomaly_', 'time_anomaly_', 'time_anomaly_grav_',
                  'time_anomaly_geo_', 'geometry_', 'h0_inferred_', 'h0_sigma_', 'macroparams_']

        arrays = [baseline, flux_anomalies, time_anomalies, time_anomalies_grav, time_anomalies_geo, np.array(info),
                  np.array(h0_inferred), np.array(h0_sigma), macromodel_parameters]

        for fname, arr in zip(fnames, arrays):
                write_data_to_file(save_name_path + fname + str(N_start) + '.txt', arr)
        # else:
        #     for fname, arr in zip(fnames, arrays):
        #         self.save_append(save_name_path + fname + str(N_start) + '.txt', arr)


        return flux_anomalies, baseline, time_anomalies, time_anomalies_geo, time_anomalies_grav, h0_inferred, h0_sigma

    def save_append(self, filename, array_to_save):

        if os.path.exists(filename):
            x = np.loadtxt(filename)
            try:
                array_to_save = np.vstack((x, array_to_save))
            except:
                array_to_save = np.append(x, array_to_save)

        np.savetxt(filename, X=array_to_save, fmt='%.5f')

    def run_once(self, realization_type, realization_kwargs, arrival_time_sigma, image_sigma, gamma_prior_scale,
            time_delay_likelihood, fix_D_dt, window_size, realization=None, **fit_smooth_kwargs):

        lens_system, data_class, return_kwargs_setup, kwargs_data_setup = \
            self.model_setup(realization_type,realization_kwargs, arrival_time_sigma, image_sigma, gamma_prior_scale, window_size,
                             realization)

        return_kwargs_fit, kwargs_data_fit = self.fit_smooth(lens_system, data_class,
                                                             time_delay_likelihood, fix_D_dt, **fit_smooth_kwargs)

        macromodel_params = np.round(return_kwargs_fit['kwargs_lens_macro_fit'], 5)
        srcx, srcy = np.round(kwargs_data_fit['source_x'], 4), np.round(kwargs_data_fit['source_y'], 4)
        macromodel_params = np.append(macromodel_params, np.array([srcx, srcy]))

        L = len(macromodel_params)
        macromodel_params = macromodel_params.reshape(1, int(L))

        key = 'flux_ratios'
        flux_anomaly = np.round(self.flux_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)

        key = 'time_delays'
        time_anomaly = np.round(self.time_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)

        time_delay_baseline = kwargs_data_setup[key]
        time_delay_baseline = time_delay_baseline.reshape(1,3)

        key = 'geo_delay'
        time_anomaly_geo = np.round(self.time_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)

        key = 'grav_delay'
        time_anomaly_grav = np.round(self.time_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)

        ddt_samples = return_kwargs_fit['D_dt_samples']
        h0 = []
        for di in ddt_samples:
            h0.append(solve_H0_from_Ddt(self.zlens, self.zsource, di, self.pyhalo._cosmology.astropy))

        return_kwargs_fit['H0_inferred'] = np.round(np.array(h0), 3)

        return time_delay_baseline, flux_anomaly, time_anomaly, time_anomaly_geo, \
               time_anomaly_grav, macromodel_params, return_kwargs_fit, return_kwargs_setup

    def compute_observables(self, lens_system):

        magnifications = lens_system.quasar_magnification(self.lens.x, self.lens.y, normed=False)
        # lens_system_quad.plot_images(data_to_fit.x, data_to_fit.y)
        lensModel, kwargs_lens = lens_system.get_lensmodel()
        dtgeo, dtgrav = lensModel.lens_model.geo_shapiro_delay(self.lens.x, self.lens.y, kwargs_lens)
        arrival_times = dtgeo + dtgrav

        return magnifications, arrival_times, dtgeo, dtgrav

    def model_setup(self, realization_type, realization_kwargs, arrival_time_sigma, image_sigma, gamma_prior_scale,
                    window_size,
                    realization=None):

        data_to_fit = LensedQuasar(self.lens.x, self.lens.y, self.lens.m)
        background_quasar = self.background_quasar_class()

        kwargs_macro = [{'theta_E': 1., 'center_x': 0., 'center_y': 0, 'e1': 0.1, 'e2': 0.1, 'gamma': 2.},
                        {'gamma1': 0.02, 'gamma2': 0.01}]

        deflector_list = [PowerLawShear(self.zlens, kwargs_macro)]

        source_ellip = np.random.uniform(0.05, 0.4)
        source_phi = np.random.uniform(-np.pi, np.pi)
        source_e1, source_e2 = phi_q2_ellipticity(source_phi, 1-source_ellip)

        amp, rsersic, nsersic = self.satellite_props(approx_theta_E(self.lens.x, self.lens.y))
        kwargs_sersic_light = [{'amp': amp, 'R_sersic': rsersic, 'n_sersic': nsersic, 'center_x': None, 'center_y': None}]

        amp_source, r_source, n_source = self.sample_source(amp)
        kwargs_sersic_source = [{'amp': amp_source, 'R_sersic': r_source, 'n_sersic': n_source,
                                 'center_x': None, 'center_y': None,
                                 'e1': source_e1, 'e2': source_e2}]
        light_model_list = [SersicLens(kwargs_sersic_light, concentric_with_model=0)]
        source_model_list = [SersicSource(kwargs_sersic_source, concentric_with_source=True)]
        r_sat_max = 0

        if self.lens.has_satellite:

            n_satellites = len(self.lens.satellite_mass_model)
            n_max = 2
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

                amp, r_sersic, n_sersic = self.satellite_props(rein_sat)

                kwargs_light_satellite = [{'amp': amp,
                                           'R_sersic': r_sersic, 'n_sersic': n_sersic,
                                       'center_x': xsat,
                                       'center_y': ysat}]

                prior_sat_light = [['amp', amp, amp * 0.2],
                                   ['center_x', xsat, 0.05],
                                   ['center_y', ysat, 0.05],
                                   ['R_sersic', r_sersic, 0.2 * r_sersic],
                                   ['n_sersic', n_sersic, 0.2 * n_sersic]]

                deflector_list += [satellite_galaxy]
                light_model_list += [SersicLens(kwargs_light_satellite, concentric_with_model=n+1,
                                                prior=prior_sat_light)]

        macromodel = MacroLensModel(deflector_list)

        if realization is None:
            realization = self.gen_realization(realization_type, realization_kwargs)
        lens_system_quad = QuadLensSystem(macromodel, self.zsource, background_quasar, realization,
                                          pyhalo_cosmology=self.pyhalo._cosmology)

        lens_system_quad.initialize(data_to_fit, include_substructure=True, verbose=False)
        magnifications, arrival_times, dtgeo, dtgrav = self.compute_observables(lens_system_quad)

        lensModel, kwargs_lens = lens_system_quad.get_lensmodel()
        lens_analysis = LensProfileAnalysis(lensModel)
        gamma_effective = lens_analysis.profile_slope(kwargs_lens, kwargs_lens[0]['theta_E'])

        macromodel_prior = [['gamma', gamma_effective, gamma_prior_scale * gamma_effective]]
        lens_system_quad.macromodel.components[0].update_prior(macromodel_prior)

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

        data_kwargs = {'psf_type': 'GAUSSIAN', 'window_size': 2*window_size, 'deltaPix': 0.05, 'fwhm': 0.1}
        data_class = ArcPlusQuad(data_to_fit.x, data_to_fit.y, magnifications, lens_system, arrival_times,
                           arrival_time_sigma, image_sigma, data_kwargs=data_kwargs, no_bkg=False, noiseless=False,
                                 normed_magnifications=False)

        imaging_data = data_class.get_lensed_image()

        halo_model_names, redshift_list_halos, kwargs_halos, _ = \
            realization.lensing_quantities(realization_kwargs['log_mlow'], realization_kwargs['log_mlow'])

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
                              'grav_delay': dtgrav[1:] - dtgrav[0]}

        return lens_system, data_class, return_kwargs, return_kwargs_data

    def fit_smooth(self, arc_quad_lens, data, time_delay_likelihood, fix_D_dt,
                   n_particles=50, n_iterations=150, n_run=10, n_burn=5, walkerRatio=4):

        lens_system_simple = arc_quad_lens.get_smooth_lens_system()

        pso_kwargs = {'sigma_scale': 1.0, 'n_particles': n_particles, 'n_iterations': n_iterations}
        mcmc_kwargs = {'n_burn': n_burn, 'n_run': n_run, 'walkerRatio': walkerRatio, 'sigma_scale': 0.1}

        chain_list, kwargs_result, kwargs_model, multi_band_list, kwargs_special, param_class = \
            lens_system_simple.fit(data, pso_kwargs, mcmc_kwargs, time_delay_likelihood=time_delay_likelihood,
                                   fix_D_dt=fix_D_dt)

        #magnifications, arrival_times, dtgeo, dtgrav = self.compute_observables(lens_system_simple)
        D_dt_true = lens_system_simple.lens_cosmo.D_dt

        lensModel, _ = lens_system_simple.get_lensmodel()
        chain_process = ChainPostProcess(lensModel, chain_list[1][1], param_class,
                                         background_quasar=lens_system_simple.background_quasar)

        flux_ratios, source_x, source_y = chain_process.flux_ratios(self.lens.x, self.lens.y)
        arrival_times, arrival_times_geo, arrival_times_grav = chain_process.time_delays(self.lens.x, self.lens.y)
        macro_params = chain_process.macro_params()
        macro_params = np.mean(macro_params, axis=0)

        return_kwargs = {'D_dt_true': D_dt_true,
                         'kwargs_lens_macro_fit': macro_params,
                         'D_dt_samples': chain_list[1][1][:,-1], 'source_x': lens_system_simple.source_centroid_x,
                         'source_y': lens_system_simple.source_centroid_y, 'zlens': self.zlens,
                         'zsource': self.zsource}

        return_kwargs_data = {'flux_ratios': np.mean(flux_ratios, axis=0),
                              'time_delays': np.mean(arrival_times, axis=0),
                              'geo_delay': np.mean(arrival_times_geo, axis=0),
                              'grav_delay': np.mean(arrival_times_grav, axis=0),
                              'source_x': np.mean(source_x),
                              'source_y': np.mean(source_y)}

        return return_kwargs, return_kwargs_data





















