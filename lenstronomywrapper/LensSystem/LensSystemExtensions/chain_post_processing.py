import numpy as np


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

    def flux_ratios(self, x_image, y_image, n_keep='all'):

        if n_keep == 'all':
            mags = np.empty((self.n_samples, 3))
            for n in range(0, self.n_samples):
                mags[n, :] = self._compute_magnification_single(x_image, y_image, self.samples[n, :])

        elif isinstance(n_keep, int):
            nstart = self.n_samples - n_keep
            mags = np.empty((n_keep, 3))
            inds = np.arange(nstart, self.n_samples, 1)
            for n, index in enumerate(inds):
                mags[n, :] = self._compute_magnification_single(x_image, y_image, self.samples[index, :])

        else:
            raise Exception('indexes must be type int or list/array')

        return mags

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
        mags = self.background_quasar.magnification(x_image, y_image, self.lensModel, kwargs_lens, normed=False)

        return mags[1:] / mags[0]
