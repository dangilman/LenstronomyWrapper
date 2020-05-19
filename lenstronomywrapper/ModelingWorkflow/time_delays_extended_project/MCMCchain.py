from lenstronomy.Plots.model_plot import ModelPlot
import numpy as np
from lenstronomywrapper.LensSystem.LensSystemExtensions.chain_post_processing import ChainPostProcess
from lenstronomywrapper.LensSystem.LensSystemExtensions.lens_maps import ResidualLensMaps
import random


class MCMCchain(object):

    def __init__(self, save_name_path, lens_system_fit, lens, mcmc_samples, kwargs_result, kwargs_model,
                 multi_band_list, kwargs_special, param_class, lensModel, kwargs_lens,
                 lensModel_full, kwargs_lens_full, window_size, kwargs_data_setup):

        self.mcmc_samples = mcmc_samples
        self.kwargs_result = kwargs_result
        self.kwargs_model = kwargs_model
        self.multi_band_list = multi_band_list
        self.kwargs_special = kwargs_special
        self.param_class = param_class
        self.lensModel = lensModel
        self.kwargs_lens = kwargs_lens
        self.kwargs_data_setup = kwargs_data_setup

        self.save_name_path = save_name_path

        self.lens = lens

        self.lensModel_full = lensModel_full
        self.kwargs_lens_full = kwargs_lens_full

        self.lens_system_fit = lens_system_fit

        self.window_size = window_size

        self.modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat")

    def get_output(self, n_burn_frac, n_keep):

        assert n_burn_frac < 1
        logL = self.modelPlot._imageModel.likelihood_data_given_model(
            source_marg=False, linear_prior=None, **self.kwargs_result)
        ndata_points = self.modelPlot._imageModel.num_data_evaluate
        chi2_imaging = logL * 2 / ndata_points

        observed_lens = self.modelPlot._select_band(0)._data
        modeled_lens = self.modelPlot._select_band(0)._model
        normalized_residuals = self.modelPlot._select_band(0)._norm_residuals

        reconstructed_source, coord_transform = \
            self.modelPlot._select_band(0).source(numPix=250, deltaPix=0.025)

        reconstructed_source_log = np.log10(reconstructed_source)

        vmin, vmax = max(np.min(reconstructed_source_log), -5), min(np.max(reconstructed_source_log), 10)
        reconstructed_source_log[np.where(reconstructed_source_log < vmin)] = vmin
        reconstructed_source_log[np.where(reconstructed_source_log > vmax)] = vmax

        residual_maps = ResidualLensMaps(self.lensModel_full, self.lensModel, self.kwargs_lens_full, self.kwargs_lens)
        kappa = residual_maps.convergence(self.window_size, 250)

        tdelay_res_geo, tdelay_res_grav = residual_maps.time_delay_surface_geoshapiro(self.window_size, 250,
                                                                                      self.lens.x[0], self.lens.y[0])
        tdelay_res_map = tdelay_res_geo + tdelay_res_grav

        nsamples_total = int(len(self.mcmc_samples[:,0]))

        n_start = round(nsamples_total * (1 - n_burn_frac))

        if n_start < 0:
            raise Exception('n burn too large, length of array is '+str(nsamples_total))

        chain_samples = self.mcmc_samples[n_start:nsamples_total, :]
        keep_inds = random.sample(list(np.arange(1, chain_samples.shape[0])), n_keep)

        chain_samples = self.mcmc_samples[keep_inds, :]

        chain_process = ChainPostProcess(self.lensModel, chain_samples, self.param_class,
                                         background_quasar=self.lens_system_fit.background_quasar)

        flux_ratios, source_x, source_y = chain_process.flux_ratios(self.lens.x, self.lens.y)

        macro_params = chain_process.macro_params()

        arrival_times = chain_process.arrival_times(self.lens.x, self.lens.y)

        relative_arrival_times = np.empty((n_keep, 3))
        for row in range(0, n_keep):
            relative_arrival_times[row, :] = self.lens.relative_time_delays(arrival_times[row, :])

        return_kwargs_data = {'flux_ratios': flux_ratios,
                              'time_delays': relative_arrival_times,
                              'source_x': source_x,
                              'source_y': source_y}

        return_kwargs = {'chi2_imaging': chi2_imaging,
                         'kwargs_lens_macro_fit': macro_params, 'mean_kappa': np.mean(kappa),
                         'residual_convergence': kappa, 'time_delay_residuals': tdelay_res_map,
                         'reconstructed_source': reconstructed_source,
                         'observed_lens': observed_lens, 'modeled_lens': modeled_lens,
                         'normalized_residuals': normalized_residuals,
                         'source_x': self.lens_system_fit.source_centroid_x,
                         'source_y': self.lens_system_fit.source_centroid_y, 'zlens': self.lens_system_fit.zlens,
                         'zsource': self.lens_system_fit.zsource}

        return return_kwargs, return_kwargs_data, self.kwargs_data_setup
