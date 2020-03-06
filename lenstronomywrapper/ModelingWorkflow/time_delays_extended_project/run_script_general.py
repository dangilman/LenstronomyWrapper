from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.scripts import *
import os

def run(Nstart, lens_class, fname, log_mlow, window_size, exp_time, background_rms, N=1,
        subtract_exact_mass_sheets=False, name_append=''):

    position_sigma = [0.005]*4
    fix_D_dt = False
    gamma_prior_scale = None

    arrival_time_sigma = [abs(delta_ti / max(abs(ti), 0.1)) for delta_ti, ti in
                          zip(lens_class.delta_time_delay, lens_class.relative_arrival_times)]

    arrival_time_sigma = np.round(arrival_time_sigma, 5)

    fit_smooth_kwargs = {'n_particles': 120, 'n_iterations': 200, 'n_run': 150, 'walkerRatio': 4, 'n_burn': 800}
    fit_smooth_kwargs = {'n_particles': 1, 'n_iterations': 1, 'n_run': 1, 'walkerRatio': 4, 'n_burn': 0}

    kwargs_cosmo = {'cosmo_kwargs': {'H0': 73.3}}
    lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo)

    if Nstart < 101:
        print('SAMPLING control...... ')
        N0 = Nstart
        realization = None
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/control' + name_append + '/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)
        shapelet_nmax = None
        fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 150, 'n_run': 150, 'walkerRatio': 4, 'n_burn': 500}

    elif Nstart < 301:
        print('SAMPLING LOS plus subs...... ')
        N0 = Nstart - 100
        save_name_path_base = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/los_plus_subs' + name_append + '/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)
        realization_file_name = save_name_path_base + '/realizations/realization_'+str(N0) + '.txt'
        realization = RealiztionFromFile(realization_file_name)
        realization.log_mlow = log_mlow
        shapelet_nmax = None

    elif Nstart < 501:
        print('SAMPLING LOS plus subs with shapelets...... ')
        N0 = Nstart - 300
        save_name_path_base = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname
        save_name_path = save_name_path_base + '/los_plus_subs' + name_append + '_shapelets/'

        if not os.path.exists(save_name_path):
            create_directory(save_name_path)
        realization_file_name = save_name_path_base + '/realizations/realization_' + str(N0) + '.txt'
        realization = RealiztionFromFile(realization_file_name)
        realization.log_mlow = log_mlow
        shapelet_nmax = 10

    run_real(lens_analog_model_class, save_name_path, N, N0, realization,
             arrival_time_sigma, position_sigma, gamma_prior_scale, fix_D_dt, window_size, exp_time, background_rms,
             time_delay_like=True, fit_smooth_kwargs=fit_smooth_kwargs, shapelet_nmax=shapelet_nmax)
