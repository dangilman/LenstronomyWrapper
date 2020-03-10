from lenstronomywrapper.Optimization.extended_optimization.sampler_init import SamplerInit
from lenstronomy.Workflow.fitting_sequence import FittingSequence

class SourceReconstruction(object):

    def __init__(self, lens_system, data_class, time_delay_likelihood=False, fix_D_dt=None):

        if time_delay_likelihood:
            D_dt_true = lens_system.lens_cosmo.ddt
            self._init = SamplerInit(lens_system, data_class, time_delay_likelihood, D_dt_true,
                                     data_class.relative_arrival_times, data_class.time_delay_sigma, fix_D_dt)
        else:
            self._init = SamplerInit(lens_system, data_class, time_delay_likelihood)

        self.lens_system = lens_system

        self._data_class = data_class

        self._lensmodel_init, _ = self.lens_system.get_lensmodel()

    def optimize(self, pso_kwargs=None, mcmc_kwargs=None, simplex_kwargs=None):

        chain_list, kwargs_result, kwargs_model, multi_band_list, param_class = \
                self._fit(pso_kwargs=pso_kwargs, mcmc_kwargs=mcmc_kwargs, simplex_kwargs=simplex_kwargs)

        kwargs_special = kwargs_result['kwargs_special']
        return chain_list, kwargs_result, kwargs_model, multi_band_list, kwargs_special, param_class

    def _fit(self, pso_kwargs=None, mcmc_kwargs=None, simplex_kwargs=None, reoptimize=False):

        kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, multi_band_list = \
            self._init.sampler_inputs(reoptimize)

        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood,
                                      kwargs_params)

        fitting_kwargs_list = []
        if pso_kwargs is not None:
            fitting_kwargs_list.append(['PSO', pso_kwargs])
        if simplex_kwargs is not None:
            fitting_kwargs_list.append(['SIMPLEX', simplex_kwargs])
        if mcmc_kwargs is not None:
            fitting_kwargs_list.append(['MCMC', mcmc_kwargs])

        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()
        print('fit kwargs:')
        print(kwargs_result['kwargs_lens'])

        kwargs_lens = kwargs_result['kwargs_lens']
        kwargs_source_light = kwargs_result['kwargs_source']
        kwargs_lens_light = kwargs_result['kwargs_lens_light']
        kwargs_ps = kwargs_result['kwargs_ps']

        for i, kw in enumerate(kwargs_lens):
            if 'ra_0' in kw.keys():
                del kwargs_lens[i]['ra_0']
            if 'dec_0' in kw.keys():
                del kwargs_lens[i]['dec_0']

        self._update_lens_system(kwargs_lens, kwargs_source_light, kwargs_lens_light, kwargs_ps)

        return chain_list, kwargs_result, kwargs_model, multi_band_list, fitting_seq.param_class

    def _update_lens_system(self, kwargs_lens, kwargs_source_light, kwargs_lens_light, kwargs_ps):

        self.lens_system.update_kwargs_macro(kwargs_lens)

        light_x, light_y = kwargs_lens_light[0]['center_x'], kwargs_lens_light[0]['center_y']
        source_x, source_y = kwargs_source_light[0]['center_x'], kwargs_source_light[0]['center_y']

        self.lens_system.update_light_centroid(light_x, light_y)
        self.lens_system.update_source_centroid(source_x, source_y)
        self._data_class.point_source.update_kwargs_ps(kwargs_ps)
        self.lens_system.set_lensmodel_static(self._lensmodel_init, kwargs_lens)

