import numpy as np
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.LensComponents.SIS import SISsatellite
from lenstronomywrapper.LensSystem.LensComponents.multipole import Multipole

from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.Utilities.data_util import approx_theta_E
from lenstronomywrapper.Utilities.misc import write_fluxes, write_params, write_macro, write_sampling_rate, write_delta_hessian

from lenstronomywrapper.Sampler.prior_sample import PriorDistribution

from copy import deepcopy

def readout(readout_path, kwargs_macro, fluxes, parameters, header, write_header, write_mode,
            sampling_rate, readout_macro,
            readout_flux_only=False, flux_file_extension=''):

    if readout_flux_only:
        write_fluxes(readout_path + 'fluxes' + flux_file_extension+'.txt', fluxes=fluxes, mode=write_mode)

    else:
        write_params(parameters, readout_path + 'parameters.txt', header, mode=write_mode,
                     write_header=write_header)
        write_fluxes(readout_path + 'fluxes' + flux_file_extension+'.txt', fluxes=fluxes, mode=write_mode)
        if readout_macro:
            write_macro(readout_path + 'macro.txt', kwargs_macro, mode=write_mode, write_header=write_header)
        write_sampling_rate(readout_path + 'sampling_rate.txt', sampling_rate)

def load_keywords(path_to_folder, job_index):

    with open(path_to_folder + '/paramdictionary_1.txt', 'r') as f:
        keywords_init = eval(f.read())

    cores_per_lens = keywords_init['compute_args']['cores_per_lens']

    Nlens = keywords_init['compute_args']['Ncores'] / keywords_init['compute_args']['cores_per_lens']

    Nlens = int(np.round(Nlens))

    data_id = []

    Nlens = int(np.round(Nlens))

    for d in range(0, int(Nlens)):
        data_id += [d + 1] * cores_per_lens

    f_index = data_id[job_index - 1]

    print('executing commands from: ', '/paramdictionary_' + str(f_index) + '.txt')

    with open(path_to_folder + '/paramdictionary_' + str(f_index) + '.txt', 'r') as f:
        keywords = eval(f.read())

    return keywords

def realization_keywords(keywords_init, prior_list_realization):

    keywords = deepcopy(keywords_init['realization_kwargs'])

    samples = {}

    keywords_sampled = {}

    for param_name in prior_list_realization.keys():

        sample = prior_list_realization[param_name]()
        keywords_sampled[param_name] = sample
        samples[param_name] = sample

    keywords.update(keywords_sampled)

    if 'mc_model' in keywords.keys():

        mc_model = {'custom': True}

        if 'log10c0' in keywords_sampled.keys():
            params = ['log10c0', 'zeta', 'beta']
        else:
            params = ['c0', 'zeta', 'beta']

        for name in params:
            if name in keywords_sampled.keys():
                mc_model[name] = keywords_sampled[name]
            else:
                assert name in keywords.keys()
                mc_model[name] = keywords[name]

        keywords['mc_model'] = mc_model

    return keywords, samples

def build_priors(params_to_vary):

    prior_list_realization = {}
    prior_list_macromodel = {}
    prior_list_source = {}
    prior_list_cosmo = {}

    for param_name in params_to_vary.keys():

        kwargs_reduced = {}
        prior_type = params_to_vary[param_name]['prior_type']
        prior_group = params_to_vary[param_name]['group']

        for name in params_to_vary[param_name].keys():
            if name == 'prior_type' or name == 'group':
                continue
            kwargs_reduced[name] = params_to_vary[param_name][name]

        new = PriorDistribution(prior_type, kwargs_reduced)

        if prior_group == 'realization':
            prior_list_realization[param_name] = new
        elif prior_group == 'macromodel':
            prior_list_macromodel[param_name] = new
        elif prior_group == 'source':
            prior_list_source[param_name] = new
        elif prior_group == 'cosmo':
            prior_list_cosmo[param_name] = new
        else:
            raise Exception('prior group '+str(prior_group)+' not recognized')

    return prior_list_realization, prior_list_macromodel, prior_list_source, prior_list_cosmo

def load_lens_source(prior_list_cosmo, keywords):

    samples = {}

    if 'zlens' in prior_list_cosmo.keys():
        zlens = prior_list_cosmo['zlens']()
        zlens = np.round(zlens, 2)
        samples['zlens'] = zlens
    else:
        zlens = keywords['zlens']

    return zlens, keywords['zsource'], samples

def load_background_source(prior_list_source, keywords):

    samples = {}

    assert 'source_model' in keywords.keys()

    if keywords['source_model'] == 'GAUSSIAN':
        assert 'source_fwhm_pc' in prior_list_source.keys()
        source_fwhm_pc = prior_list_source['source_fwhm_pc']()
        samples['source_fwhm_pc'] = source_fwhm_pc
        if 'source_fwhm_pc_2' in prior_list_source.keys():
            samples['source_fwhm_pc_2'] = prior_list_source['source_fwhm_pc_2']()

    elif keywords['source_model'] == 'DOUBLE_GAUSSIAN':

        assert 'source_fwhm_pc' in prior_list_source.keys()
        assert 'dx_source_2' in prior_list_source.keys()
        assert 'dy_source_2' in prior_list_source.keys()
        assert 'size_scale_2' in prior_list_source.keys()
        assert 'amp_scale_2' in prior_list_source.keys()

        name_out = ['source_fwhm_pc', 'dx', 'dy', 'size_scale', 'amp_scale']
        for name, pname in zip(['source_fwhm_pc', 'dx_source_2', 'dy_source_2', 'size_scale_2', 'amp_scale_2'], name_out):
            samples[pname] = prior_list_source[name]()

    else:
        raise Exception('source model must be specifed and be either GAUSSIAN OR DOUBLE_GAUSSIAN')

    return samples

def load_double_background_quasar(prior_list_source, keywords):

    samples = {}

    n_sources = len(prior_list_source)

    if n_sources == 1:
        kwargs_quasar = {'center_x': 0., 'center_y': 0., 'source_fwhm_pc': None}
        if 'source_fwhm_pc' not in keywords.keys():
            assert 'source_fwhm_pc' in prior_list_source.keys()
            kwargs_quasar['source_fwhm_pc'] = prior_list_source['source_fwhm_pc']()
        quasar = Quasar(kwargs_quasar)
        samples['source_fwhm_pc'] = kwargs_quasar['source_fwhm_pc']
        return quasar, samples

    else:
        raise Exception('only single sources implemented, not '+str(n_sources))

def load_powerlaw_ellipsoid_macromodel(zlens, prior_list_macromodel,
                                       kwargs_macro_ref, secondary_lens_components,
                                       keywords, x_image, y_image):

    samples = {}

    constrain_params = None

    theta_E_approx = approx_theta_E(x_image, y_image)
    e1, e2 = np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)

    if kwargs_macro_ref is None:
        kwargs_init = [{'theta_E': theta_E_approx, 'e1': e1, 'e2': e2, 'center_x': 0., 'center_y': 0.,
                    'gamma': 2.}, {'gamma1': 0.04, 'gamma2': 0.02}]
    else:
        kwargs_init = deepcopy(kwargs_macro_ref[0:2])

    if 'gamma_macro' in prior_list_macromodel.keys():
        gamma_macro = prior_list_macromodel['gamma_macro']()
        kwargs_init[0]['gamma'] = gamma_macro
        samples['gamma'] = gamma_macro

    if 'shear' in prior_list_macromodel.keys():
        shear = prior_list_macromodel['shear']()
        gamma1, gamma2 = shear / np.sqrt(2), shear / np.sqrt(2)
        kwargs_init[1]['gamma1'], kwargs_init[1]['gamma2'] = gamma1, gamma2
        constrain_params = {'shear': shear}
        samples['shear'] = shear

    if 'q_min' in keywords.keys():
        constrain_params['q_min'] = keywords['q_min']

    main_deflector = PowerLawShear(zlens, kwargs_init)
    component_list = [main_deflector]

    secondary_models, samples_secondary = load_seccondary_lens_components(prior_list_macromodel,
                                                 secondary_lens_components,
                                                 zlens)

    for name_secondary in samples_secondary.keys():
        samples[name_secondary] = samples_secondary[name_secondary]

    component_list += secondary_models

    return MacroLensModel(component_list), samples, constrain_params

def load_seccondary_lens_components(prior_list_macro,
                                    secondary_lens_components,
                                    z_main):

    lens_component_list = []
    params_sampled = {}

    if secondary_lens_components is None:
        return lens_component_list, params_sampled
    else:
        assert isinstance(secondary_lens_components, list)

    for idx, comp in enumerate(secondary_lens_components):

        component_name = comp[0]
        component_kwargs = comp[1]
        kwargs_model = {}

        if component_name == 'SIS':

            component_class = SISsatellite
            names = ['theta_E', 'center_x', 'center_y']

        elif component_name == 'multipole':

            component_class = Multipole
            names = ['m', 'a_m', 'phi_m', 'center_x', 'center_y']

        else:
            raise Exception('component name '+str(component_name) +
                            ' not recognized.')

        names_input = [ni + '_' + str(idx + 1) for ni in names]

        for input, output in zip(names_input, names):

            if input in prior_list_macro.keys():
                value = prior_list_macro[input]()
                kwargs_model[output] = value
                params_sampled[input] = value
            else:

                assert output in component_kwargs.keys(), 'error: did not find expected parameter '+str(output)
                kwargs_model[output] = component_kwargs[output]

        z_name = 'z_' + str(idx + 1)
        if z_name in prior_list_macro.keys():
            z_comp = prior_list_macro[z_name]()
            params_sampled[z_name] = z_comp
        else:
            assert 'redshift' in component_kwargs.keys()
            if component_kwargs['redshift'] == 'zlens':
                z_comp = z_main
            else:
                z_comp = component_kwargs['redshift']

        kwargs_component = {'redshift': z_comp, 'kwargs_init': [kwargs_model]}

        new_component = component_class(**kwargs_component)
        lens_component_list.append(new_component)

    return lens_component_list, params_sampled

def load_optimization_settings(keywords):

    if keywords['keywords_optimizer']['routine'] == 'dynamic':

        keywords_opt = {}

        for name in ['global_log_mlow', 'log_mass_cuts', 'aperture_sizes', 'refit', 'particle_swarm',
                     're_optimize', 'realization_type']:

            keywords_opt[name] = keywords['keywords_optimizer'][name]

        if 'n_particles' not in keywords['keywords_optimizer'].keys():
            n_particles = 35
        else:
            n_particles = keywords['keywords_optimizer']['n_particles']

        if 'simplex_n_iter' not in keywords['keywords_optimizer'].keys():
            simplex_n_iter = 300
            keywords['simplex_n_iter'] = simplex_n_iter
        else:
            simplex_n_iter = keywords['keywords_optimizer']['simplex_n_iter']

        keywords_opt['n_particles'] = n_particles
        keywords_opt['n_particles'] = simplex_n_iter

    elif keywords['keywords_optimizer']['routine'] == 'hierarchical':

        keywords_opt = {}

    else:
        raise Exception('optimization routine '+
                        keywords['optimization_routine']+ ' not recognized.')

    return keywords_opt

def load_data_to_fit(keywords):

    x_image, y_image, flux_ratios = keywords['x_image'], \
                                    keywords['y_image'], \
                                    keywords['fluxes']

    x_image, y_image, flux_ratios = np.array(x_image), np.array(y_image), np.array(flux_ratios)
    sigma = keywords['astrometric_uncertainty']

    x_image_sigma = np.random.normal(0, sigma)
    y_image_sigma = np.random.normal(0, sigma)

    data_to_fit = LensedQuasar(x_image + x_image_sigma, y_image + y_image_sigma, flux_ratios)

    return data_to_fit

def simulation_setup(keyword_arguments, prior_list_realization, prior_list_cosmo, prior_list_macromodel,
                     prior_list_source, kwargs_macro_ref=None):

    params_sampled = {}

    ######## Sample keyword arguments for the substructure realization ##########

    kwargs_rendering, realization_samples = realization_keywords(keyword_arguments, prior_list_realization)
    params_sampled.update(realization_samples)

    ######## Sample keyword arguments for the lensing volume ##########
    zlens, zsource, lens_source_sampled = load_lens_source(prior_list_cosmo, keyword_arguments)
    params_sampled.update(lens_source_sampled)

    ######## Sample keyword arguments for the background source ##########
    source_samples = load_background_source(prior_list_source,
                                                               keyword_arguments)
    params_sampled.update(source_samples)

    ################## Set up the data to fit ####################
    data_to_fit = load_data_to_fit(keyword_arguments)

    ######## Sample keyword arguments for the macromodel ##########
    macromodel, macro_samples, constrain_params = \
        load_powerlaw_ellipsoid_macromodel(zlens, prior_list_macromodel, kwargs_macro_ref,
                                           keyword_arguments['secondary_lens_components'],
                                           keyword_arguments, data_to_fit.x, data_to_fit.y)
    params_sampled.update(macro_samples)

    ################ Get the optimization settings ################
    optimization_settings = load_optimization_settings(keyword_arguments)

    ################ Perform a fit with only a smooth model ################
    optimization_routine = keyword_arguments['optimization_routine']

    return kwargs_rendering, realization_samples, zlens, zsource, lens_source_sampled, \
           macromodel, macro_samples, constrain_params, source_samples, data_to_fit, \
           optimization_settings, optimization_routine, params_sampled



