from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.scripts import *
import os

def run(Nstart, lens_class, fname, log_mlow, window_size, exp_time, background_rms, N=1,
        subtract_exact_mass_sheets=False, name_append=''):

    opening_angle = 10 * window_size
    position_sigma = [0.005]*4
    fix_D_dt = False
    gamma_prior_scale = None

    arrival_time_sigma = [abs(delta_ti / max(abs(ti), 0.1)) for delta_ti, ti in
                          zip(lens_class.delta_time_delay, lens_class.relative_arrival_times)]

    arrival_time_sigma = np.round(arrival_time_sigma, 5)

    fit_smooth_kwargs = {'n_particles': 120, 'n_iterations': 200, 'n_run': 200, 'walkerRatio': 4, 'n_burn': 1000}
    #fit_smooth_kwargs = {'n_particles': 10, 'n_iterations': 1, 'n_run': 1, 'walkerRatio': 4, 'n_burn': 1}

    if Nstart < 201:
        print('SAMPLING control...... ')
        arrival_time_sigma *= 1.
        N0 = Nstart
        SHMF_norm = 0.
        LOS_norm = 0.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/control' + name_append + '/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    else:
        print('SAMPLING LOS plus subs...... ')
        N0 = Nstart - 200
        SHMF_norm = 0.02
        LOS_norm = 1.
        save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/los_plus_subs' + name_append + '/'
        if not os.path.exists(save_name_path):
            create_directory(save_name_path)

    run_real(lens_class, save_name_path, N, N0, SHMF_norm, LOS_norm, log_mlow, opening_angle,
             arrival_time_sigma, position_sigma, gamma_prior_scale, fix_D_dt, window_size, exp_time, background_rms,
             time_delay_like=True, fit_smooth_kwargs=fit_smooth_kwargs,
             subtract_exact_mass_sheets=subtract_exact_mass_sheets)
