from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.scripts import *
import os

def run(Nstart, lens_class, fname, log_mlow, window_size, gamma_macro, N=2):

    opening_angle = 8 * window_size
    position_sigma = [0.005]*4
    fix_D_dt = False
    gamma_prior_scale = None

    arrival_time_sigma = [delta_ti / ti for delta_ti, ti in
                          zip(lens_class.delta_time_delay, lens_class.relative_arrival_times)]
    arrival_time_sigma = np.round(arrival_time_sigma, 5)

    fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 200, 'n_run': 50, 'walkerRatio': 4, 'n_burn': 650}

    if Nstart < 501:
        print('SAMPLING subs1...... ')
        N0 = Nstart
        SHMF_norm = 0.15
        LOS_norm = 0.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/subs1/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    elif Nstart < 1001:
        print('SAMPLING subs2...... ')
        N0 = Nstart - 500
        SHMF_norm = 0.3
        LOS_norm = 0.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/subs2/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    elif Nstart < 1501:
        print('SAMPLING subs3...... ')
        N0 = Nstart - 1000
        SHMF_norm = 0.6
        LOS_norm = 0.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/subs3/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    elif Nstart < 2001:
        print('SAMPLING subs4...... ')
        N0 = Nstart - 1500
        SHMF_norm = 1.2
        LOS_norm = 0.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/subs4/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    elif Nstart < 2501:
        print('SAMPLING LOS...... ')
        N0 = Nstart - 2000
        SHMF_norm = 0.
        LOS_norm = 1.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/LOSonly/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    elif Nstart < 3001:
        print('SAMPLING LOS+subs...... ')
        N0 = Nstart - 2500
        SHMF_norm = 0.6
        LOS_norm = 1.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/LOSplussubs/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    else:
        raise Exception('out of range.')

    run_real(lens_class, save_name_path, N, N0, SHMF_norm, LOS_norm, log_mlow, opening_angle,
             arrival_time_sigma, position_sigma, gamma_prior_scale, fix_D_dt, window_size, gamma_macro,
             time_delay_like=True, fit_smooth_kwargs=fit_smooth_kwargs)
