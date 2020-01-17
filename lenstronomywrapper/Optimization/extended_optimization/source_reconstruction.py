from lenstronomywrapper.Optimization.extended_optimization.sampler_init import SamplerInit
from lenstronomy.Workflow.fitting_sequence import FittingSequence

class SourceReconstruction(object):

    def __init__(self, lens_system, data_class, time_delay_likelihood=False, D_dt_true=None, dt_measured=None, dt_sigma=None):

        self._init = SamplerInit(lens_system, data_class, time_delay_likelihood, D_dt_true, dt_measured, dt_sigma)

    def fit(self, pso_kwargs=None, mcmc_kwargs=None):

        kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, multi_band_list = \
            self._init.sampler_inputs()
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood,
                                      kwargs_params)
        if pso_kwargs is None:
            pso_kwargs = {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 200}
        if mcmc_kwargs is None:
            mcmc_kwargs = {'n_burn': 10, 'n_run': 10, 'walkerRatio': 4, 'sigma_scale': .1}
        fitting_kwargs_list = [['PSO', pso_kwargs],
                               ['MCMC', mcmc_kwargs]
                               ]

        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()

        return chain_list, kwargs_result
