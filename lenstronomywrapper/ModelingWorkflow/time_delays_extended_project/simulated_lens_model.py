from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.lens_analog_model import *
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomy.Util.param_util import shear_polar2cartesian
from lenstronomywrapper.LensSystem.LensSystemExtensions.solver import iterative_rayshooting
from lenstronomywrapper.Utilities.data_util import image_separation_vectors_quad
import os

import numpy as np
from pyHalo.pyhalo import pyHalo


class Mock(object):

    def __init__(self, x, y, m, zlens, zsource):

        self.x = x
        self.y = y
        self.m = m
        self.zlens, self.zsrc = zlens, zsource
        self.has_satellite = False

class SimulatedModel(object):

    def __init__(self, realization_type, realization_kwargs, kwargs_cosmology={},
                 kwargs_quasar=None, zlens=None, zsource=None, free_convergence=False):

        self._realization_type = realization_type
        self.free_convergence = free_convergence
        self._realization_kwargs = realization_kwargs
        self._zlens = zlens
        self._zsource = zsource
        self.kwargs_cosmology = kwargs_cosmology
        self.kwargs_quasar = kwargs_quasar

    def setup(self):

        if self._zlens is None:
            zlens = np.round(max(0.2, np.random.normal(0.5, 0.1)), 2)
        if self._zsource is None:
            zsource = np.random.normal(zlens + 1.5, 0.5)
            zsource = np.round(max(zlens + 0.3, zsource), 2)

        if self.kwargs_quasar is None:
            self.kwargs_quasar = {'center_x': 0, 'center_y': 0, 'source_fwhm_pc': 25}

        pyhalo = pyHalo(zlens, zsource, cosmology_kwargs=self.kwargs_cosmology)

        mock_lens_class, realization, dtgeo, dtgrav, arrival_times = \
            self.gen_mock_data(self._realization_type, self._realization_kwargs, pyhalo)

        return realization, mock_lens_class, realization, pyhalo, zlens, zsource

    def run(self, save_name_path, N_start, N, arrival_time_sigma, image_positions_sigma, gamma_prior_scale,
            time_delay_likelihood, fix_D_dt, **fit_smooth_kwargs):

        assert os.path.exists(save_name_path)

        h0 = []
        h0_sigma = []
        for n in range(0, N):

            realization, mock_lens_class, realization, pyhalo, zlens, zsource = self.setup()

            self._analog_model = AnalogModel(mock_lens_class, self.kwargs_cosmology, self.kwargs_quasar, False,
                                             pyhalo, self.free_convergence)

            tbaseline, flux_anomaly, time_anomaly, time_anomaly_geo, time_anomaly_grav, macromodel_parameters, return_kwargs_fit, \
            return_kwargs_setup = self._analog_model.run_once(None, self._realization_kwargs, arrival_time_sigma,
                       image_positions_sigma, gamma_prior_scale, time_delay_likelihood, fix_D_dt, realization, **fit_smooth_kwargs)

            h0.append(np.mean(return_kwargs_fit['H0_inferred']))
            h0_sigma.append(np.std(return_kwargs_fit['H0_inferred']))

            x_image, y_image = mock_lens_class.x, mock_lens_class.y
            info = [zlens, zsource, x_image[0], x_image[1], x_image[2], x_image[3], y_image[0], y_image[1], y_image[2], y_image[3]]

            fnames = ['tbaseline_', 'flux_anomaly_', 'time_anomaly_', 'time_anomaly_grav_',
                      'time_anomaly_geo_', 'geometry_']
            arrays = [tbaseline, flux_anomaly, time_anomaly, time_anomaly_grav, time_anomaly_geo, np.array(info)]

            for fname, arr in zip(fnames, arrays):
                self.save_append(save_name_path + fname + str(N_start) + '.txt', arr)

        for fname, arr in zip(['h0_inferred_', 'h0_inferred_sigma'], [h0, h0_sigma]):
            self.save_append(save_name_path + fname + str(N_start) + '.txt', arr)

    def save_append(self, filename, array_to_save):

        if os.path.exists(filename):

            x = np.loadtxt(filename)
            try:
                array_to_save = np.vstack((x, array_to_save))
            except:
                array_to_save = np.append(x, array_to_save)

        np.savetxt(filename, X=array_to_save, fmt='%.5f')

    def sample_source(self, config):

        if config == 'cross':
            xmin, xmax = 0.03, 0.07
            ymin, ymax = 0.03, 0.07
        elif config == 'cusp_fold':
            xmin, xmax = 0.05, 0.1
            ymin, ymax = 0.05, 0.1
        else:
            raise Exception(config +' not recognized.')

        x, y = np.random.normal(xmin, xmax), np.random.normal(ymin, ymax)
        return x, y

    def background_quasar_class(self):

        return Quasar(self.kwargs_quasar)

    def gen_realization(self, pyhalo, realization_type, realization_kwargs):

        realization = pyhalo.render(realization_type, realization_kwargs)[0]
        return realization

    def gen_mock_data(self, realization_type, realization_kwargs, pyhalo):

        if np.random.rand() < 0.5:
            config = 'cusp_fold'
        else:
            config = 'cross'

        while True:

            kwargs_macro = self.macromodel_kwargs()

            zlens = pyhalo.zlens
            zsource = pyhalo.zsource

            srcx, srcy = self.sample_source(config)
            realization = self.gen_realization(pyhalo, realization_type, realization_kwargs)

            prior_gamma = [['gamma', kwargs_macro[0]['gamma'], 0.05*kwargs_macro[0]['gamma']]]
            macromodel = MacroLensModel([PowerLawShear(zlens, kwargs_macro, prior=prior_gamma)])
            lens_system = QuadLensSystem(macromodel, zsource, self.background_quasar_class(),
                                         realization)

            lens_system.update_source_centroid(srcx, srcy)

            lensModel_macro, kwargs_macro = lens_system.get_lensmodel(include_substructure=False)

            x_init, y_init = lens_system.solve_lens_equation(lensModel_macro, kwargs_macro, 10**-6)

            if len(x_init) != 4:
                continue

            image_seps = image_separation_vectors_quad(x_init, y_init)[0]
            sep = list(np.squeeze(image_seps))
            if min(sep) < 0.25:
                continue

            lensModel, kwargs = lens_system.get_lensmodel(include_substructure=True)
            x_image, y_image = iterative_rayshooting(srcx, srcy, x_init, y_init, lensModel, kwargs)

            image_seps = image_separation_vectors_quad(x_init, y_init)[0]
            sep = list(np.squeeze(image_seps))

            if min(sep) < 0.3:
                continue

            mags, arrival_times, dtgeo, dtgrav = self.compute_observables(lens_system, x_image, y_image)

            lens = Mock(x_image, y_image, mags, zlens, zsource)
            # plt.scatter(x_image, y_image)
            # plt.show()
            # a=input('continue')
            break

        return lens, realization, dtgeo, dtgrav, arrival_times

    def compute_observables(self, lens_system, x, y):

        magnifications = lens_system.quasar_magnification(x, y, normed=False)
        # lens_system_quad.plot_images(data_to_fit.x, data_to_fit.y)
        lensModel, kwargs_lens = lens_system.get_lensmodel()
        dtgeo, dtgrav = lensModel.lens_model.geo_shapiro_delay(x, y, kwargs_lens)
        arrival_times = dtgeo + dtgrav

        return magnifications, arrival_times, dtgeo, dtgrav

    def macromodel_kwargs(self):

        ellip = np.random.uniform(0.05, 0.4)
        phi = np.random.uniform(-np.pi, np.pi)
        e1, e2 = phi_q2_ellipticity(phi, 1 - ellip)
        gamma = np.random.normal(2., 0.05)
        shear = np.random.normal(0.06, 0.02)
        shear_pa = np.random.uniform(-np.pi, np.pi)
        shear_e1, shear_e2 = shear_polar2cartesian(shear_pa, shear)

        theta_E = abs(np.random.normal(1., 0.15))
        center_x, center_y = np.random.normal(0, 0.01), np.random.normal(0, 0.01)
        kwargs = [{'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y,
                   'e1': e1, 'e2': e2, 'gamma': gamma}, {'gamma1': shear_e1, 'gamma2': shear_e2},
                  {'kappa_ext': 0.}]
        return kwargs


