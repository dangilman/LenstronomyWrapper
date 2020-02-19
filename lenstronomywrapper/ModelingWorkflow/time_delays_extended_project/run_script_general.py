from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.scripts import *
import os

def run(Nstart, lens_class, fname, log_mlow, window_size, gamma_macro, N=2):

    opening_angle = 10 * window_size
    position_sigma = [0.005]*4
    fix_D_dt = False
    gamma_prior_scale = None

    arrival_time_sigma = [abs(delta_ti / max(abs(ti), 0.1)) for delta_ti, ti in
                          zip(lens_class.delta_time_delay, lens_class.relative_arrival_times)]
    arrival_time_sigma = np.round(arrival_time_sigma, 5)

    fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 200, 'n_run': 60, 'walkerRatio': 4, 'n_burn': 700}
    #fit_smooth_kwargs = {'n_particles': 1, 'n_iterations': 1, 'n_run': 5, 'walkerRatio': 4, 'n_burn': 1}
    if Nstart < 501:
        print('SAMPLING control...... ')
        N0 = Nstart
        SHMF_norm = 0.
        LOS_norm = 0.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/control/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    elif Nstart < 1001:
        print('SAMPLING subs only...... ')
        N0 = Nstart - 500
        SHMF_norm = 0.02
        LOS_norm = 0.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/subsonly/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    elif Nstart < 1501:
        print('SAMPLING LOS only...... ')
        N0 = Nstart - 1000
        SHMF_norm = 0.0
        LOS_norm = 1.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/LOSonly/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    elif Nstart < 2001:
        print('SAMPLING LOS+subs...... ')
        N0 = Nstart - 1500
        SHMF_norm = 0.02
        LOS_norm = 1.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/LOS_plus_subs/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    else:
        raise Exception('out of range.')

    run_real(lens_class, save_name_path, N, N0, SHMF_norm, LOS_norm, log_mlow, opening_angle,
             arrival_time_sigma, position_sigma, gamma_prior_scale, fix_D_dt, window_size, gamma_macro,
             time_delay_like=True, fit_smooth_kwargs=fit_smooth_kwargs)
