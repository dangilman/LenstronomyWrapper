from lenstronomywrapper.Sampler.utilities import build_priors, simulation_setup
import numpy.testing as npt
import pytest
import numpy as np
from copy import deepcopy

class TestUtil(object):

    def setup(self):

        realization_kwargs = {'log_mass_sheet_min': 7.5, 'log_mass_sheet_max': 9.6,
                              'log_mlow': 6.8, 'log_mhigh': 9.6, 'opening_angle_factor': 6.0,
                              'kwargs_halo_mass_function': {'geometry_type': 'DOUBLE_CONE'}}

        params_to_vary = {'sigma_sub': {'prior_type': 'Uniform', 'low': 0.0, 'high': 0.1, 'group': 'realization'},
                                            'log_mc': {'prior_type': 'Uniform', 'low': 4.5, 'high': 10.0, 'group': 'realization'},
                                            'power_law_index': {'prior_type': 'Uniform', 'low': -1.95, 'high': -1.85, 'group': 'realization'},
                                            'LOS_normalization': {'prior_type': 'Uniform', 'low': 0.75, 'high': 1.25, 'group': 'realization'},
                                            'gamma_macro': {'prior_type': 'Uniform', 'low': 1.95, 'high': 2.2, 'group': 'macromodel'},
                                            'source_fwhm_pc': {'prior_type': 'Uniform', 'low': 25, 'high': 60.0, 'group': 'source'},
                                            'log_m_host': {'prior_type': 'Gaussian', 'mean': 13.3, 'sigma': 0.3, 'group': 'realization',
                                                           'positive_definite': True},
                                            'shear': {'prior_type': 'Uniform', 'low': 0.1, 'high': 0.28, 'group': 'macromodel', 'positive_definite': True},
                                            'zlens': {'prior_type': 'CustomPDF',
                                                      'parameter_values': [0.061757, 0.072423, 0.083195, 0.094076, 0.105066, 0.116167, 0.127379, 0.138704, 0.150142, 0.161695, 0.173365, 0.185151, 0.197056, 0.209081, 0.221226, 0.233494, 0.245884, 0.258399, 0.27104, 0.283808, 0.296704, 0.309729, 0.322886, 0.336174, 0.349596, 0.363153, 0.376846, 0.390677, 0.404646, 0.418756, 0.433008, 0.447402, 0.461942, 0.476627, 0.49146, 0.506442, 0.521574, 0.536859, 0.552297, 0.56789, 0.583639, 0.599547, 0.615615, 0.631844, 0.648236, 0.664793, 0.681516, 0.698407, 0.715467, 0.732699],
                                                      'probabilities': [0.02615367293314519, 0.03838602176725219, 0.05347645756511757, 0.0725306235841679, 0.09630886832562517, 0.12623407934656633, 0.16417518562622155, 0.20884805880315607, 0.26301023498988496, 0.3214934787583879, 0.38287282202034273, 0.45397191772355616, 0.542488486707241, 0.639484077204843, 0.7251307369119288, 0.7938328158521795, 0.8598735281771241, 0.9143369408113338, 0.9567347408331182, 0.9874969862892992, 1.0, 0.9900518786585224, 0.9536506592224443, 0.9028834939217891, 0.8479129212004911, 0.7868556931289844, 0.7136885493679468, 0.6312642654072518, 0.5448556417276241, 0.46266070574067525, 0.3912921200491005, 0.3311464094314153, 0.27933310406694906, 0.23641032481988106, 0.20044085233245912, 0.16754488746161728, 0.1374436390204859, 0.11158009005028767, 0.0903198755597547, 0.07278415467403583, 0.057768275325266585, 0.04469269941586026, 0.03454031885977097, 0.02691135345900558, 0.02085861076809542, 0.015429038586511548, 0.010818096705539966, 0.007348759452648944, 0.004766781014295698, 0.003029982902928531], 'group': 'cosmo'},
                                            'a_m_1': {'prior_type': 'Gaussian', 'mean': 0.0, 'sigma': 0.01, 'group': 'macromodel', 'positive_definite': False},
                                            'theta_E_2': {'prior_type': 'Gaussian', 'mean': 0.27, 'sigma': 0.05, 'group': 'macromodel', 'positive_definite': True},
                                            'center_x_2': {'prior_type': 'Gaussian', 'mean': -0.307, 'sigma': 0.05, 'group': 'macromodel'},
                                            'center_y_2': {'prior_type': 'Gaussian', 'mean': -1.153, 'sigma': 0.05, 'group': 'macromodel'}}

        self.keyword_arguments_gaussian = {'zlens': 0.31, 'zsource': 1.7, 'Nsamples': 800, 'verbose': False,
                                  'readout_steps': 25,
                                  'realization_kwargs': realization_kwargs,
                                  'params_to_vary': params_to_vary,
                                  'x_image': [0.838, -0.784, 0.048, -0.289],
                                  'y_image': [0.378, -0.211, -0.527, 0.528],
                                  'fluxes': [1.0, 1.0, 0.59, 0.79],
                                  'compute_args': {'cores_per_lens': 500, 'Ncores': 500},
                                  'astrometric_uncertainty': [0.003, 0.003, 0.003, 0.003],
                                  'keywords_optimizer': {'routine': 'hierarchical', 'settings_class': 'default_CDM'},
                                  'adaptive_mag': True,
                                           'secondary_lens_components':
                                      [['multipole', {'redshift': 'zlens', 'm': 4, 'center_x': 0.0, 'center_y': 0.0, 'phi_m': 0.0}],
                                       ['SIS', {'redshift': 'zlens'}]],
                                           'save_best_realization': True,
                                           'optimization_routine': 'fixed_shear_powerlaw_multipole',
                                           'grid_axis_ratio': 0.5, 'enforce_unblended': True,
                                           'preset_model': 'WDMLovell2020',
                                           'grid_rmax': 0.77, 'source_model': 'GAUSSIAN'}

        params_to_vary_dbl = deepcopy(params_to_vary)
        params_to_vary_dbl['dx_source_2'] = {'prior_type': 'Gaussian', 'mean': 0., 'sigma': 0.05, 'group': 'source'}
        params_to_vary_dbl['dy_source_2'] = {'prior_type': 'Uniform', 'low': -0.05, 'high': 0.05, 'group': 'source'}
        params_to_vary_dbl['size_scale_2'] = {'prior_type': 'Uniform', 'low': 0.0, 'high': 1.5, 'group': 'source'}
        params_to_vary_dbl['amp_scale_2'] = {'prior_type': 'Uniform', 'low': 0.0, 'high': 1.5, 'group': 'source'}
        self.keyword_arguments_dbl_gaussian = {'zlens': 0.31, 'zsource': 1.7, 'Nsamples': 800, 'verbose': False,
                                           'readout_steps': 25,
                                           'realization_kwargs': realization_kwargs,
                                           'params_to_vary': params_to_vary_dbl,
                                           'x_image': [0.838, -0.784, 0.048, -0.289],
                                           'y_image': [0.378, -0.211, -0.527, 0.528],
                                           'fluxes': [1.0, 1.0, 0.59, 0.79],
                                           'compute_args': {'cores_per_lens': 500, 'Ncores': 500},
                                           'astrometric_uncertainty': [0.003, 0.003, 0.003, 0.003],
                                           'keywords_optimizer': {'routine': 'hierarchical',
                                                                  'settings_class': 'default_CDM'},
                                           'adaptive_mag': True,
                                           'secondary_lens_components':
                                               [['multipole',
                                                 {'redshift': 'zlens', 'm': 4, 'center_x': 0.0, 'center_y': 0.0,
                                                  'phi_m': 0.0}],
                                                ['SIS', {'redshift': 'zlens'}]],
                                           'save_best_realization': True,
                                           'optimization_routine': 'fixed_shear_powerlaw_multipole',
                                           'grid_axis_ratio': 0.5, 'enforce_unblended': True,
                                           'preset_model': 'WDMLovell2020',
                                           'grid_rmax': 0.77, 'source_model': 'DOUBLE_GAUSSIAN'}

    def test_prior_setup(self):

        priors = build_priors(self.keyword_arguments_gaussian['params_to_vary'])
        (prior_list_realization, prior_list_macromodel, prior_list_source, prior_list_cosmo) = priors
        npt.assert_equal(len(prior_list_realization), 5)
        npt.assert_equal(len(prior_list_macromodel), 6)
        npt.assert_equal(len(prior_list_source), 1)
        npt.assert_almost_equal(len(prior_list_cosmo), 1)


    def test_sim_setup(self):

        keywords_master = self.keyword_arguments_gaussian

        priors = build_priors(self.keyword_arguments_gaussian['params_to_vary'])
        (prior_list_realization, prior_list_macromodel, prior_list_source, prior_list_cosmo) = priors
        kwargs_macro_ref = None
        setup1 = simulation_setup(keywords_master, prior_list_realization, prior_list_cosmo,
                                 prior_list_macromodel, prior_list_source, kwargs_macro_ref)

        priors = build_priors(self.keyword_arguments_dbl_gaussian['params_to_vary'])
        (prior_list_realization, prior_list_macromodel, prior_list_source, prior_list_cosmo) = priors
        setup2 = simulation_setup(self.keyword_arguments_dbl_gaussian, prior_list_realization, prior_list_cosmo,
                                 prior_list_macromodel, prior_list_source, kwargs_macro_ref)

        (kwargs_rendering1,
         realization_samples1,
         zlens1,
         zsource1,
         lens_source_sampled1, \
         macromodel1,
         macro_samples1,
         constrain_params1,
         background_source1,
         source_samples1,
         data_to_fit1, \
         optimization_settings1,
         optimization_routine1,
         params_sampled1) = setup1

        (kwargs_rendering2,
         realization_samples2,
         zlens2,
         zsource2,
         lens_source_sampled2, \
         macromodel2,
         macro_samples2,
         constrain_params2,
         background_source2,
         source_samples2,
         data_to_fit2, \
         optimization_settings2,
         optimization_routine2,
         params_sampled2) = setup2

        for kw in ['log_mass_sheet_min', 'log_mass_sheet_max', 'log_mlow', 'log_mhigh', 'opening_angle_factor', 'kwargs_halo_mass_function']:
            npt.assert_equal(True, kw in kwargs_rendering1.keys())
            npt.assert_equal(True, kw in kwargs_rendering2.keys())
            npt.assert_equal(kwargs_rendering1[kw], kwargs_rendering2[kw])

        npt.assert_equal(len(realization_samples1.keys()), len(prior_list_realization))
        for kw in realization_samples1.keys():
            npt.assert_equal(False, realization_samples1[kw]==realization_samples2[kw])

        npt.assert_equal(False, zlens1 == zlens2)
        npt.assert_equal(True, zsource1 == zsource2)
        npt.assert_equal(False, lens_source_sampled1['zlens'] == lens_source_sampled2['zlens'])
        npt.assert_equal(lens_source_sampled1['zlens'], zlens1)
        npt.assert_equal(macromodel1.zlens, zlens1)
        npt.assert_equal(macromodel2.zlens, zlens2)
        lens_model_list, redshift_list, kwargs_macro, _ = macromodel1.get_lenstronomy_args()
        npt.assert_string_equal(lens_model_list[0], 'EPL')
        npt.assert_string_equal(lens_model_list[1], 'SHEAR')
        npt.assert_string_equal(lens_model_list[2], 'MULTIPOLE')
        npt.assert_string_equal(lens_model_list[3], 'SIS')
        npt.assert_equal(redshift_list[0], zlens1)
        npt.assert_equal(redshift_list[1], zlens1)
        npt.assert_equal(redshift_list[2], zlens1)
        npt.assert_equal(redshift_list[3], zlens1)
        npt.assert_equal(kwargs_macro[0]['gamma'], macro_samples1['gamma'])
        npt.assert_equal(kwargs_macro[0]['gamma'], params_sampled1['gamma'])
        npt.assert_almost_equal(np.hypot(kwargs_macro[1]['gamma1'], kwargs_macro[1]['gamma2']), constrain_params1['shear'], 4)
        npt.assert_equal(background_source1._kwargs_init['source_fwhm_pc'], source_samples1['source_fwhm_pc'])
        npt.assert_equal(True, data_to_fit1.x[0] != data_to_fit2.x[0])
        for kw in params_sampled1.keys():
            if kw in kwargs_rendering1.keys():
                npt.assert_equal(kwargs_rendering1[kw], params_sampled1[kw])

        npt.assert_equal(background_source2._kwargs_init['source_fwhm_pc'], source_samples2['source_fwhm_pc'])
        npt.assert_equal(background_source2._kwargs_init['dx'], source_samples2['dx_source_2'])
        npt.assert_equal(background_source2._kwargs_init['dy'], source_samples2['dy_source_2'])
        npt.assert_equal(background_source2._kwargs_init['size_scale'], source_samples2['size_scale_2'])
        npt.assert_equal(background_source2._grid_rmax, self.keyword_arguments_dbl_gaussian['grid_rmax'])

if __name__ == '__main__':

    pytest.main()
