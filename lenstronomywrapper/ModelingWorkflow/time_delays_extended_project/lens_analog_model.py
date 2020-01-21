from lenstronomywrapper.LensSystem.arc_quad_lens import ArcQuadLensSystem
from lenstronomywrapper.LensSystem.BackgroundSource.sersic_source import SersicSource
from lenstronomywrapper.LensSystem.LensLight.sersic_lens import SersicLens
from lenstronomywrapper.LensData.arc_plus_quad import ArcPlusQuad
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.LensComponents.powerlawshearconvergence import PowerLawShearConvergence
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.LensSystem.LensComponents.satellite import SISsatellite
from lenstronomywrapper.LensSystem.light_model import LightModel
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomywrapper.Utilities.data_util import approx_theta_E
from lenstronomywrapper.Utilities.lensing_util import solve_H0_from_Ddt
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar

import numpy as np
from pyHalo.pyhalo import pyHalo
import os

import matplotlib.pyplot as plt

class AnalogModel(object):

    def __init__(self, lens_class_instance, kwargs_cosmology,
                 kwargs_quasar=None, makeplots=False, pyhalo=None):

        if kwargs_quasar is None:
            kwargs_quasar = {'center_x': 0, 'center_y': 0, 'source_fwhm_pc': 25}

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

    def flux_anomaly(self, mags1, mags2):

        f1, f2 = mags1[1:]/mags1[0], mags2[1:]/mags2[0]
        return np.array(f1 - f2)

    def time_anomaly(self, t1, t2):

        dt1 = t1[1:] - t1[0]
        dt2 = t2[1:] - t2[0]
        return np.array(dt1 - dt2)

    def run(self, save_name_path, N_start, N, realization_type, realization_kwargs, arrival_time_sigma,
            time_delay_likelihood, fix_D_dt, fit_smooth_kwargs):

        for n in range(0, N):
            tbaseline, f, t, tgeo, tgrav, kw_fit, kw_setup = self.run_once(realization_type,
                                                                realization_kwargs,
                                                                arrival_time_sigma,
                                                                time_delay_likelihood,
                                                                fix_D_dt, **fit_smooth_kwargs)

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
            else:
                baseline = np.vstack((baseline, tbaseline))
                flux_anomalies = np.vstack((flux_anomalies, f))
                time_anomalies = np.vstack((time_anomalies, t))
                time_anomalies_geo = np.vstack((time_anomalies_geo, tgeo))
                time_anomalies_grav = np.vstack((time_anomalies_grav, tgrav))
                h0_inferred = np.vstack((h0_inferred, h0_inf))
                h0_sigma = np.vstack((h0_sigma, h0_inf_sigma))

        fnames = ['tbaseline_', 'flux_anomaly_', 'time_anomaly_', 'time_anomaly_grav_',
                  'time_anomaly_geo_', 'geometry_', 'h0_inferred_', 'h0_sigma_']
        arrays = [baseline, flux_anomalies, time_anomalies, time_anomalies_grav, time_anomalies_geo, np.array(info),
                  np.array(h0_inferred), np.array(h0_sigma)]

        for fname, arr in zip(fnames, arrays):
            self.save_append(save_name_path + fname + str(N_start) + '.txt', arr)

        return flux_anomalies, baseline, time_anomalies, time_anomalies_geo, time_anomalies_grav, h0_inferred, h0_sigma

    def save_append(self, filename, array_to_save):

        if os.path.exists(filename):
            x = np.loadtxt(filename)
            try:
                array_to_save = np.vstack((x, array_to_save))
            except:
                array_to_save = np.append(x, array_to_save)

        np.savetxt(filename, X=array_to_save, fmt='%.5f')

    def run_once(self, realization_type, realization_kwargs, arrival_time_sigma,
            time_delay_likelihood, fix_D_dt, realization=None, **fit_smooth_kwargs):

        lens_system, data_class, return_kwargs_setup, kwargs_data_setup = \
            self.model_setup(realization_type,realization_kwargs, arrival_time_sigma, realization)

        return_kwargs_fit, kwargs_data_fit = self.fit_smooth(lens_system, data_class,
                                                             time_delay_likelihood, fix_D_dt, **fit_smooth_kwargs)

        key = 'mags'
        flux_anomaly = self.flux_anomaly(kwargs_data_fit[key], kwargs_data_setup[key])

        key = 'arrival_times'
        time_anomaly = self.time_anomaly(kwargs_data_fit[key], kwargs_data_setup[key])
        time_delay_baseline = kwargs_data_fit[key][1:] - kwargs_data_fit[key][0]

        key = 'geo_delay'
        time_anomaly_geo = self.time_anomaly(kwargs_data_fit[key], kwargs_data_setup[key])

        key = 'grav_delay'
        time_anomaly_grav = self.time_anomaly(kwargs_data_fit[key], kwargs_data_setup[key])

        ddt_samples = return_kwargs_fit['D_dt_samples']
        h0 = []
        for di in ddt_samples:
            h0.append(solve_H0_from_Ddt(self.zlens, self.zsource, di, self.pyhalo._cosmology.astropy))

        return_kwargs_fit['H0_inferred'] = np.array(h0)

        return time_delay_baseline, flux_anomaly, time_anomaly, time_anomaly_geo, time_anomaly_grav, return_kwargs_fit, return_kwargs_setup

    def compute_observables(self, lens_system):

        magnifications = lens_system.quasar_magnification(self.lens.x, self.lens.y, normed=False)
        # lens_system_quad.plot_images(data_to_fit.x, data_to_fit.y)
        lensModel, kwargs_lens = lens_system.get_lensmodel()
        dtgeo, dtgrav = lensModel.lens_model.geo_shapiro_delay(self.lens.x, self.lens.y, kwargs_lens)
        arrival_times = dtgeo + dtgrav

        return magnifications, arrival_times, dtgeo, dtgrav

    def model_setup(self, realization_type, realization_kwargs, arrival_time_sigma, realization=None):

        data_to_fit = LensedQuasar(self.lens.x, self.lens.y, self.lens.m)
        background_quasar = self.background_quasar_class()
        deflector_list = [PowerLawShearConvergence(self.zlens)]

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

            for n in range(0, n_satellites):

                rein_sat = self.lens.satellite_kwargs[n]['theta_E']
                xsat = self.lens.satellite_kwargs[n]['center_x']
                ysat = self.lens.satellite_kwargs[n]['center_y']
                r_sat = np.sqrt(xsat ** 2 + ysat ** 2)
                r_sat_max = max(r_sat, r_sat_max)
                satellite_redshift = self.lens.satellite_redshift[n]
                priors = [['theta_E', rein_sat, 0.02 * rein_sat], ['center_x', xsat, 0.05 * xsat],
                          ['center_y', ysat, 0.05 * ysat]]
                kwargs_init = [self.lens.satellite_kwargs[n]]

                satellite_galaxy = SISsatellite(satellite_redshift, kwargs_init=kwargs_init,
                                            prior=priors)

                amp, r_sersic, n_sersic = self.satellite_props(rein_sat)

                kwargs_light_satellite = [{'amp': amp,
                                           'R_sersic': r_sersic, 'n_sersic': n_sersic,
                                       'center_x': xsat,
                                       'center_y': ysat}]

                deflector_list += [satellite_galaxy]
                light_model_list += [SersicLens(kwargs_light_satellite, concentric_with_model=n+1)]

        macromodel = MacroLensModel(deflector_list)

        if realization is None:
            realization = self.gen_realization(realization_type, realization_kwargs)
        lens_system_quad = QuadLensSystem(macromodel, self.zsource, background_quasar, realization,
                                          pyhalo_cosmology=self.pyhalo._cosmology)
        lens_system_quad.initialize(data_to_fit, include_substructure=True, verbose=False)
        magnifications, arrival_times, dtgeo, dtgrav = self.compute_observables(lens_system_quad)

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

        data_kwargs = {'psf_type': 'GAUSSIAN', 'window_size': 2*window_size, 'deltaPix': 0.025}
        data_class = ArcPlusQuad(data_to_fit.x, data_to_fit.y, magnifications, lens_system, arrival_times,
                           arrival_time_sigma, data_kwargs=data_kwargs, no_bkg=False, noiseless=False,
                                 normed_magnifications=False)

        imaging_data = data_class.get_lensed_image()

        # if self._makeplots:
        #     plt.imshow(np.log10(imaging_data), origin='lower')
        #     plt.show()
        #     a=input('continue')

        halo_model_names, redshift_list_halos, kwargs_halos, _ = \
            realization.lensing_quantities(realization_kwargs['log_mlow'], realization_kwargs['log_mlow'])

        return_kwargs = {'imaging_data': imaging_data,
                         'kwargs_lens_macro': lens_system.macromodel.kwargs,
                         'lens_model_list_macro': lens_system.macromodel.lens_model_list,
                         'redshift_list_macro': lens_system.macromodel.redshift_list,
                         'lens_model_list_halos': halo_model_names,
                         'redshift_list_halos': redshift_list_halos,
                         'kwargs_lens_halos': kwargs_halos}

        return_kwargs_data = {'mags': magnifications,
                              'arrival_times': arrival_times,
                              'geo_delay': dtgeo,
                              'grav_delay': dtgrav}

        return lens_system, data_class, return_kwargs, return_kwargs_data

    def fit_smooth(self, arc_quad_lens, data, time_delay_likelihood, fix_D_dt,
                   n_particles=50, n_iterations=150, n_run=10, n_burn=5, walkerRatio=4):

        lens_system_simple = arc_quad_lens.get_smooth_lens_system()

        pso_kwargs = {'sigma_scale': 0.5, 'n_particles': n_particles, 'n_iterations': n_iterations}
        mcmc_kwargs = {'n_burn': n_burn, 'n_run': n_run, 'walkerRatio': walkerRatio, 'sigma_scale': .1}
        chain_list, kwargs_result, kwargs_model, multi_band_list, kwargs_special, param_class = \
            lens_system_simple.fit(data, pso_kwargs, mcmc_kwargs, time_delay_likelihood=time_delay_likelihood,
                                   fix_D_dt=fix_D_dt)

        magnifications, arrival_times, dtgeo, dtgrav = self.compute_observables(lens_system_simple)
        D_dt_true = lens_system_simple.lens_cosmo.D_dt
        D_dt_fit = kwargs_special['D_dt']

        return_kwargs = {'D_dt_true': D_dt_true, 'D_dt_fit': D_dt_fit,
                         'kwargs_lens_macro_fit': lens_system_simple.macromodel.kwargs,
                         'D_dt_samples': chain_list[1][1][:,-1], 'source_x': lens_system_simple.source_centroid_x,
                         'source_y': lens_system_simple.source_centroid_y, 'zlens': self.zlens,
                         'zsource': self.zsource}

        return_kwargs_data = {'mags': magnifications,
                              'arrival_times': arrival_times,
                              'geo_delay': dtgeo,
                              'grav_delay': dtgrav}

        return return_kwargs, return_kwargs_data





















