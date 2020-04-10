from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.scripts import *
import os
import dill
from pyHalo.single_realization import RealiztionFromFile

def run(Nstart, lens_class, fname, log_mlow, window_size, exp_time, background_rms, N=1,
        subtract_exact_mass_sheets=False, name_append='', fix_Ddt=False,
        fit_smooth_kwargs=None):

    if fit_smooth_kwargs is None:
        # default
        fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 250, 'n_run': 150, 'walkerRatio': 4, 'n_burn': 300}

    position_sigma = [0.005]*4
    gamma_prior_scale = None

    arrival_time_sigma = [abs(delta_ti / max(abs(ti), 0.1)) for delta_ti, ti in
                          zip(lens_class.delta_time_delay, lens_class.relative_arrival_times)]

    arrival_time_sigma = np.round(arrival_time_sigma, 5)

    mdef = 'TNFW'
    realization_kwargs = {'mdef_main': mdef, 'mdef_los': mdef,
                          'log_mlow': log_mlow, 'log_mhigh': 10., 'power_law_index': -1.9,
                          'parent_m200': 10 ** 13, 'r_tidal': '0.5Rs',
                          'cone_opening_angle': 10 * window_size, 'opening_angle_factor': 10 * window_size,
                          'subtract_exact_mass_sheets': subtract_exact_mass_sheets,
                          'subtract_subhalo_mass_sheet': True, 'subhalo_mass_sheet_scale': 1}

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
        if not os.path.exists(save_name_path_base + '/realizations/'):
            create_directory(save_name_path_base + '/realizations/')

        print('SAMPLING LOS plus subs...... ')
        #fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 250, 'n_run': 150, 'walkerRatio': 4, 'n_burn': 300}
        #fit_smooth_kwargs = {'n_particles': 1, 'n_iterations': 1, 'n_run': 2, 'walkerRatio': 4, 'n_burn': 0}
        N0 = Nstart - 100
        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo,
                                              pickle_directory=save_name_path_base + '/realizations/',
                                              class_idx=N0, log_mlow=log_mlow)
        #### CREATE REALIZATION ####
        lens_system_file_name = base_path + '/tdelay_output/raw/' + fname + '/realizations/macromodel_' + str(N0)
        if os.path.exists(lens_system_file_name):
            file = open(lens_system_file_name, 'rb')
            system = dill.load(file)
            realization = system.realization

        else:
            SHMF_norm = 0.02
            LOS_norm = 1.
            realization_kwargs['sigma_sub'] = SHMF_norm
            realization_kwargs['LOS_normalization'] = LOS_norm
            realization = lens_analog_model_class.pyhalo.render('composite_powerlaw',
                                                                        realization_kwargs)[0]
            realization.log_mlow = log_mlow

        #realization = RealiztionFromFile(realization_file_name)
        #realization.log_mlow = log_mlow

        shapelet_nmax = None

    elif Nstart < 501:
        print('SAMPLING LOS plus subs with shapelets...... ')
        #fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 250,
        #                     'n_run': 150, 'walkerRatio': 4, 'n_burn': 300}
        #fit_smooth_kwargs = {'n_particles': 2, 'n_iterations': 2, 'n_run': 3, 'walkerRatio': 4, 'n_burn': 0}
        N0 = Nstart - 300
        save_name_path_base = base_path + '/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/los_plus_subs' + name_append + '_shapelets/'
        lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo,
                                              pickle_directory=save_name_path_base + '/realizations/',
                                              class_idx=N0, log_mlow=log_mlow)

        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

        #realization_file_name = save_name_path_base + '/realizations/realization_'+str(N0) + name_append
        #assert os.path.exists(realization_file_name + '_kwargslist.txt')

        #realization = RealiztionFromFile(realization_file_name)
        #realization.log_mlow = log_mlow
        realization = True

        shapelet_nmax = 8

    print('N0:', N0)
    print('filename: ', save_name_path)
    print('arrival time uncertainties: ', arrival_time_sigma)
    run_real(lens_analog_model_class, save_name_path, N, N0, realization,
             arrival_time_sigma, position_sigma, gamma_prior_scale, fix_Ddt, window_size, exp_time, background_rms,
             time_delay_like=time_delay_likelihood, fit_smooth_kwargs=fit_smooth_kwargs, shapelet_nmax=shapelet_nmax)

