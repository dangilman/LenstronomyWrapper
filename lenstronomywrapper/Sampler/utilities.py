import numpy as np
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.LensComponents.satellite import SISsatellite

from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar

from lenstronomywrapper.Utilities.misc import write_fluxes, write_params, write_macro

from lenstronomywrapper.Sampler.prior_sample import PriorDistribution

from copy import deepcopy

def readout(readout_path, kwargs_macro, fluxes, parameters, header, write_header, write_mode):

    write_params(parameters, readout_path + 'parameters.txt', header, mode=write_mode,
                 write_header=write_header)
    write_fluxes(readout_path + 'fluxes.txt', fluxes=fluxes, mode=write_mode)
    write_macro(readout_path + 'macro.txt', kwargs_macro, mode=write_mode, write_header=write_header)


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
        if prior_group == 'cosmo':
            prior_list_cosmo[param_name] = new

    return prior_list_realization, prior_list_macromodel, prior_list_source, prior_list_cosmo

def load_lens_source(prior_list_cosmo, keywords):

    samples = {}

    if 'zlens' in prior_list_cosmo.keys():
        zlens = prior_list_cosmo['zlens']()
        samples['zlens'] = zlens
    else:
        zlens = keywords['zlens']

    return zlens, keywords['zsource'], samples


def build_kwargs_macro_powerlaw_ellipsoid(prior_list_macromodel):

    samples = {}

    constrain_params = None

    e1, e2 = np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)

    kwargs_init = [{'theta_E': 1., 'e1': e1, 'e2': e2, 'center_x': 0., 'center_y': 0.,
                    'gamma': 2.}, {'gamma1': 0.04, 'gamma2': 0.02}]

    if 'gamma_macro' in prior_list_macromodel.keys():
        gamma_macro = prior_list_macromodel['gamma_macro']()
        kwargs_init[0]['gamma'] = gamma_macro
        samples['gamma'] = gamma_macro

    if 'shear' in prior_list_macromodel.keys():
        shear = prior_list_macromodel['shear']()
        gamma1, gamma2 = shear/np.sqrt(2), shear/np.sqrt(2)
        kwargs_init[1]['gamma1'], kwargs_init[1]['gamma2'] = gamma1, gamma2
        constrain_params = {'shear': shear}
        samples['shear'] = shear

    return kwargs_init, constrain_params, samples

def load_background_quasar(prior_list_source, keywords):

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


def load_powerlaw_ellipsoid_macromodel(zlens, prior_list_macromodel, kwargs_macro_ref):

    samples = {}

    constrain_params = None

    e1, e2 = np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)

    if kwargs_macro_ref is None:
        kwargs_init = [{'theta_E': 1., 'e1': e1, 'e2': e2, 'center_x': 0., 'center_y': 0.,
                    'gamma': 2.}, {'gamma1': 0.04, 'gamma2': 0.02}]
    else:
        kwargs_init = deepcopy(kwargs_macro_ref)

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

    deflector = PowerLawShear(zlens, kwargs_init)
    component_list = [deflector]

    if constrain_params is None:
        opt_routine = 'fixed_powerlaw_shear'
    else:
        opt_routine = 'fixedshearpowerlaw'

    return MacroLensModel(component_list), samples, constrain_params, opt_routine

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

    x_image += x_image_sigma
    y_image += y_image_sigma

    data_to_fit = LensedQuasar(x_image, y_image, flux_ratios)

    return data_to_fit


