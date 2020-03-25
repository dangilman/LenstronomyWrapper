import numpy as np
import lenstronomy.Util.constants as const
from scipy.optimize import minimize

class ChainPostProcess(object):

    def __init__(self, lensModel, mcmc_arg_samples, param_class, background_quasar=None):

        self.n_samples = int(np.shape(mcmc_arg_samples)[0])
        self.lensModel = lensModel
        self.param_class = param_class
        self.samples = mcmc_arg_samples
        self.background_quasar = background_quasar

    @staticmethod
    def _unpack_kwargs(kwargs_list):
        values, keys = [], []
        for kw in kwargs_list:
            for key in kw.keys():
                values.append(kw[key])
                keys.append(key)
        return np.array(values), keys

    def macro_params(self, n_keep='all'):

        kwargs_list = self.param_class.args2kwargs(tuple(self.samples[0, :]))
        kwargs_lens = kwargs_list['kwargs_lens']
        _, keys = self._unpack_kwargs(kwargs_lens)
        nparams = len(keys)

        macro_params = np.empty((self.n_samples, nparams))

        for n in range(0, self.n_samples):
            kwargs_list = self.param_class.args2kwargs(tuple(self.samples[n, :]))
            kwargs_lens = kwargs_list['kwargs_lens']
            values, _ = self._unpack_kwargs(kwargs_lens)
            macro_params[n, :] = values
        return macro_params

    def fermat_potential_difference(self, x_image, y_image):

        fermat_pot = np.empty((self.n_samples, 3))
        for n in range(0, self.n_samples):
            fermat_pot[n,:] = self._compute_fermat_potential_difference_single(x_image, y_image, self.samples[n,:])

        return fermat_pot

    def _eval_likelihood(self, Ddt_proposed, fermat_potential_modeled, time_delays_observed, sigma):

        chi2 = 0
        time_delays_modeled = []
        for i in range(0, 3):
            time_delays_modeled.append(const.delay_arcsec2days(fermat_potential_modeled[i], Ddt_proposed))

        for i in range(0, 3):
            chi2 += 0.5 * ((time_delays_modeled[i] - time_delays_observed[i]) / sigma[i]) ** 2

        return chi2

    def maximum_likelihood_Ddt(self, x_image, y_image, measured_time_delays, sigmas):

        fermat_potential_diff = self.fermat_potential_difference(x_image, y_image)
        tdelays = np.empty((self.n_samples, 3))
        ddt = []

        init = minimize(self._eval_likelihood, x0=np.array([3600]), args=(fermat_potential_diff[0, :],
                                                                       measured_time_delays,
                                                                       sigmas))['x'][0]

        for i in range(0, int(fermat_potential_diff.shape[0])):
            new_ddt = minimize(self._eval_likelihood, x0=np.array(init), args=(fermat_potential_diff[i, :],
                                                                       measured_time_delays,
                                                                       sigmas))['x'][0]
            ddt.append(new_ddt)
            tdelays[i,:] = const.delay_arcsec2days(fermat_potential_diff[i,:], new_ddt)

        return np.array(ddt), tdelays

    def time_delays(self, x_image, y_image, n_keep='all'):

        if n_keep == 'all':
            tdelays = np.empty((self.n_samples, 3))
            tdelays_geo = np.empty((self.n_samples, 3))
            tdelays_grav = np.empty((self.n_samples, 3))
            for n in range(0, self.n_samples):
                tdelays[n, :], tdelays_geo[n, :], tdelays_grav[n, :] = self._compute_time_delay_single(x_image, y_image,
                                                                                                       self.samples[n,
                                                                                                       :])

        elif isinstance(n_keep, int):
            nstart = self.n_samples - n_keep
            tdelays = np.empty((n_keep, 3))
            tdelays_geo = np.empty((self.n_samples, 3))
            tdelays_grav = np.empty((self.n_samples, 3))
            inds = np.arange(nstart, self.n_samples, 1)
            for n, index in enumerate(inds):
                tdelays[n, :], tdelays_geo[n, :], tdelays_grav[n, :] = self._compute_time_delay_single(x_image, y_image,
                                                                                                       self.samples[
                                                                                                       index, :])

        else:
            raise Exception('indexes not valid')

        return tdelays, tdelays_geo, tdelays_grav

    def arrival_times(self, x_image, y_image, n_keep='all'):

        times = np.empty((self.n_samples, 4))

        for n in range(0, self.n_samples):
            times[n, :] = self._compute_arrivaltime_single(x_image, y_image, self.samples[n,:])

        return times

    def flux_ratios(self, x_image, y_image, n_keep='all'):

        if n_keep == 'all':
            mags = np.empty((self.n_samples, 3))
            source_x = np.empty(self.n_samples)
            source_y = np.empty(self.n_samples)
            for n in range(0, self.n_samples):
                mags[n, :], srcx, srcy = self._compute_magnification_single(x_image, y_image, self.samples[n, :])
                source_x[n] = srcx
                source_y[n] = srcy

        elif isinstance(n_keep, int):
            nstart = self.n_samples - n_keep
            mags = np.empty((n_keep, 3))
            source_x = np.empty(n_keep)
            source_y = np.empty(n_keep)
            inds = np.arange(nstart, self.n_samples, 1)
            for n, index in enumerate(inds):
                mags[n, :], srcx, srcy = self._compute_magnification_single(x_image, y_image, self.samples[index, :])
                source_x[n] = srcx
                source_y[n] = srcy
        else:
            raise Exception('indexes must be type int or list/array')

        return mags, source_x, source_y

    def compute_arrival_times(self, x_image, y_image, mcmc_args):

        tdelays = np.empty((self.n_samples, 4))

        for n in range(0, self.n_samples):
            kwargs_list = self.param_class.args2kwargs(mcmc_args[n,:])
            kwargs_lens = kwargs_list['kwargs_lens']
            tgeo, tgrav = self.lensModel.lens_model.geo_shapiro_delay(x_image, y_image, kwargs_lens)
            tdelays[n, :] = tgeo + tgrav

        return tdelays

    def _compute_fermat_potential_difference_single(self, x_image, y_image, mcmc_args):

        kwargs_list = self.param_class.args2kwargs(tuple(mcmc_args))
        kwargs_lens = kwargs_list['kwargs_lens']
        pot = self.lensModel.fermat_potential(x_image, y_image, kwargs_lens)
        return pot[1:] - pot[0]

    def _compute_arrivaltime_single(self, x_image, y_image, mcmc_args):

        kwargs_list = self.param_class.args2kwargs(tuple(mcmc_args))
        kwargs_lens = kwargs_list['kwargs_lens']
        tgeo, tgrav = self.lensModel.lens_model.geo_shapiro_delay(x_image, y_image, kwargs_lens)
        t = tgeo + tgrav
        return t

    def _compute_time_delay_single(self, x_image, y_image, mcmc_args):

        kwargs_list = self.param_class.args2kwargs(tuple(mcmc_args))
        kwargs_lens = kwargs_list['kwargs_lens']
        tgeo, tgrav = self.lensModel.lens_model.geo_shapiro_delay(x_image, y_image, kwargs_lens)
        t = tgeo + tgrav
        return t[1:] - t[0], tgeo[1:] - t[0], tgrav[1:] - tgrav[0]

    def _compute_magnification_single(self, x_image, y_image, mcmc_args):

        if self.background_quasar is None:
            raise Exception('must provide instance of background quasar class to compute the flux ratios.')

        kwargs_list = self.param_class.args2kwargs(tuple(mcmc_args))
        kwargs_lens = kwargs_list['kwargs_lens']
        srcx, srcy = self.lensModel.ray_shooting(x_image, y_image, kwargs_lens)
        srcx, srcy = np.mean(srcx), np.mean(srcy)
        self.background_quasar.update_position(srcx, srcy)
        mags, _ = self.background_quasar.magnification(x_image, y_image, self.lensModel, kwargs_lens, normed=False)

        return mags[1:] / mags[0], srcx, srcy

