from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.scripts import *
import os

def run(Nstart, lens_class, gamma_prior, fname, log_mlow, window_size, exp_time, background_rms, N=1,
        subtract_exact_mass_sheets=False, name_append='', fix_Ddt=False,
        fit_smooth_kwargs=None, window_scale=10, do_sampling=True, realization_kwargs=None):

    position_sigma = [0.005]*4

    arrival_time_sigma = [abs(delta_ti / max(abs(ti), 0.1)) for delta_ti, ti in
                          zip(lens_class.delta_time_delay, lens_class.relative_arrival_times)]

    arrival_time_sigma = np.round(arrival_time_sigma, 5)

    mdef = 'TNFW'

    realization_kwargs_base = {'mdef_main': mdef, 'mdef_los': mdef, 'mass_func_type': 'POWER_LAW',
                          'log_mlow': log_mlow, 'log_mhigh': 9., 'power_law_index': -1.9,
                          'r_tidal': '0.5Rs',
                          'cone_opening_angle': window_scale * window_size,
                          'log_mass_sheet_min': log_mlow,
                          'log_mass_sheet_max': 9.,
                          'opening_angle_factor': window_scale * window_size,
                          'subtract_exact_mass_sheets': subtract_exact_mass_sheets,
                          'subtract_subhalo_mass_sheet': True}

    realization_kwargs.update(realization_kwargs_base)

    kwargs_cosmo = {'cosmo_kwargs': {'H0': 73.3}}

    try:
        base_path = os.getenv('HOME') + '/../../../../u/flashscratch/g/gilmanda'
        assert os.path.exists(base_path)
    except:
        base_path = os.getenv('HOME') + '/Code'
        assert os.path.exists(base_path)

    if fix_Ddt:
        name_append += '_fixDdt'
        time_delay_likelihood = False
    else:
        time_delay_likelihood = True

    if Nstart < 51:
        print('SAMPLING control...... ')
        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        N0 = Nstart
        if not os.path.exists(save_name_path_base + '/chains_control' + name_append + '/'):
            create_directory(save_name_path_base + '/chains_control' + name_append + '/')

        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo,
                                 chain_directory=save_name_path_base + '/chains_control'+name_append+'/',
                                              class_idx=N0)

        save_name_path = base_path + '/tdelay_output/raw/' + fname + '/control' + name_append + '/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)
        shapelet_nmax = None
        use_realization = False

    elif Nstart < 101:

        print('SAMPLING control...... ')
        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        if not os.path.exists(save_name_path_base + '/chains_control_shapelets' + name_append + '/'):
            create_directory(save_name_path_base + '/chains_control_shapelets' + name_append + '/')

        N0 = Nstart - 50
        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo,
                                pickle_directory=save_name_path_base + '/realizations'+name_append+'/',
                                chain_directory=save_name_path_base + '/chains_control_shapelets'+name_append+'/',
                                              class_idx=N0)
        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/control_shapelets' + name_append + '/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

        shapelet_nmax = 8
        use_realization = False

    elif Nstart < 301:

        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/los_plus_subs' + name_append + '/'

        if not os.path.exists(save_name_path):
            create_directory(save_name_path)
        if not os.path.exists(save_name_path_base + '/realizations'+name_append+'/'):
            create_directory(save_name_path_base + '/realizations'+name_append+'/')
        if not os.path.exists(save_name_path_base + '/chains' + name_append + '/'):
            create_directory(save_name_path_base + '/chains' + name_append + '/')


        print('SAMPLING LOS plus subs...... ')
        N0 = Nstart - 100
        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo,
                                    pickle_directory=save_name_path_base + '/realizations'+name_append+'/',
                                              chain_directory=save_name_path_base + '/chains' + name_append + '/',
                                              class_idx=N0, do_sampling=do_sampling)

        use_realization = True

        shapelet_nmax = None

    elif Nstart < 501:
        print('SAMPLING LOS plus subs with shapelets...... ')
        N0 = Nstart - 300

        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/los_plus_subs' + name_append + '_shapelets/'

        if not os.path.exists(save_name_path_base + '/realizations'+name_append+'/'):
            create_directory(save_name_path_base + '/realizations'+name_append+'/')
        if not os.path.exists(save_name_path_base + '/chains_shapelets' + name_append + '/'):
            create_directory(save_name_path_base + '/chains_shapelets' + name_append + '/')

        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo,
                                  pickle_directory=save_name_path_base + '/realizations'+name_append+'/',
                                  chain_directory=save_name_path_base + '/chains_shapelets' + name_append + '/',
                                              class_idx=N0, do_sampling=do_sampling)

        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

        use_realization = True

        shapelet_nmax = 8

    print('N0:', N0)
    print('filename: ', save_name_path)
    print('arrival time uncertainties: ', arrival_time_sigma)
    run_real(lens_analog_model_class, gamma_prior, save_name_path, N, N0, use_realization,
             arrival_time_sigma, position_sigma, realization_kwargs, fix_Ddt, window_size, exp_time, background_rms,
             time_delay_like=time_delay_likelihood, fit_smooth_kwargs=fit_smooth_kwargs, shapelet_nmax=shapelet_nmax)

