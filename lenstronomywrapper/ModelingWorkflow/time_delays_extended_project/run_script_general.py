from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.scripts import *
import os
import dill

def run(Nstart, lens_class, fname, log_mlow, window_size, exp_time, background_rms, N=1,
        subtract_exact_mass_sheets=False, name_append='', fix_Ddt=False,
        fit_smooth_kwargs=None, window_scale=10):

    position_sigma = [0.005]*4

    arrival_time_sigma = [abs(delta_ti / max(abs(ti), 0.1)) for delta_ti, ti in
                          zip(lens_class.delta_time_delay, lens_class.relative_arrival_times)]

    arrival_time_sigma = np.round(arrival_time_sigma, 5)

    mdef = 'TNFW'
    realization_kwargs = {'mdef_main': mdef, 'mdef_los': mdef, 'mass_func_type': 'POWER_LAW',
                          'log_mlow': log_mlow, 'log_mhigh': 9., 'power_law_index': -1.9,
                          'parent_m200': 10 ** 13, 'r_tidal': '0.5Rs',
                          'cone_opening_angle': window_scale * window_size,
                          'log_mass_sheet_min': log_mlow,
                          'sigma_sub': 0.02,
                          'log_mass_sheet_max': 9.,
                          'opening_angle_factor': window_scale * window_size,
                          'subtract_exact_mass_sheets': subtract_exact_mass_sheets,
                          'subtract_subhalo_mass_sheet': True}

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
        N0 = Nstart
        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo)
        save_name_path = base_path + '/tdelay_output/raw/' + fname + '/control' + name_append + '/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)
        shapelet_nmax = None
        realization = None
        #fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 200, 'n_run': 150, 'walkerRatio': 4, 'n_burn': 300}
        #fit_smooth_kwargs = {'n_particles': 1, 'n_iterations': 1, 'n_run': 3, 'walkerRatio': 4, 'n_burn': 0}

    elif Nstart < 101:

        print('SAMPLING control...... ')
        N0 = Nstart - 50
        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo)
        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/control_shapelets' + name_append + '/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

        shapelet_nmax = 8
        realization = None
        #fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 250, 'n_run': 150, 'walkerRatio': 4, 'n_burn': 300}
        #fit_smooth_kwargs = {'n_particles': 1, 'n_iterations': 1, 'n_run': 3, 'walkerRatio': 4, 'n_burn': 0}

    elif Nstart < 301:

        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/los_plus_subs' + name_append + '/'

        if not os.path.exists(save_name_path):
            create_directory(save_name_path)
        if not os.path.exists(save_name_path_base + '/realizations'+name_append+'/'):
            create_directory(save_name_path_base + '/realizations'+name_append+'/')

        print('SAMPLING LOS plus subs...... ')
        #fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 250, 'n_run': 150, 'walkerRatio': 4, 'n_burn': 300}
        #fit_smooth_kwargs = {'n_particles': 1, 'n_iterations': 1, 'n_run': 2, 'walkerRatio': 4, 'n_burn': 0}
        N0 = Nstart - 100
        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo,
                                    pickle_directory=save_name_path_base + '/realizations'+name_append+'/',
                                              class_idx=N0, log_mlow=log_mlow)
        #### CREATE REALIZATION ####
        lens_system_file_name = base_path + '/tdelay_output/raw/' + fname + \
                                '/realizations'+name_append+'/macromodel_' + str(N0)
        if os.path.exists(lens_system_file_name):
            file = open(lens_system_file_name, 'rb')
            system = dill.load(file)
            realization = system.realization
        else:
            realization = True

        shapelet_nmax = None

    elif Nstart < 501:
        print('SAMPLING LOS plus subs with shapelets...... ')
        N0 = Nstart - 300
        #fit_smooth_kwargs = {'n_particles': 50, 'n_iterations': 50, 'n_run': 2, 'walkerRatio': 4, 'n_burn': 0}
        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/los_plus_subs' + name_append + '_shapelets/'
        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo,
                                  pickle_directory=save_name_path_base + '/realizations'+name_append+'/',
                                              class_idx=N0, log_mlow=log_mlow)

        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

        realization = True

        shapelet_nmax = 8

    print('N0:', N0)
    print('filename: ', save_name_path)
    print('arrival time uncertainties: ', arrival_time_sigma)
    run_real(lens_analog_model_class, save_name_path, N, N0, realization,
             arrival_time_sigma, position_sigma, realization_kwargs, fix_Ddt, window_size, exp_time, background_rms,
             time_delay_like=time_delay_likelihood, fit_smooth_kwargs=fit_smooth_kwargs, shapelet_nmax=shapelet_nmax)

