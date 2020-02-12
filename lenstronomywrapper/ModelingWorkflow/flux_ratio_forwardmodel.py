from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar
from lenstronomywrapper.Optimization.quad_optimization.hierarchical import HierarchicalOptimization
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.Utilities.parameter_util import *
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear

def setup(z_lens, z_source, source_size_pc, source_x, source_y, kwargs_macro):

    kwargs_quasar = {'source_fwhm_pc': source_size_pc, 'center_x': None, 'center_y': None}

    macromodel = MacroLensModel([PowerLawShear(z_lens, kwargs_macro)])
    quasar = Quasar(kwargs_quasar)
    lens_system = QuadLensSystem(macromodel, z_source, quasar)
    lens_system.update_source_centroid(source_x, source_y)

    lensmodel, kwargs = lens_system.get_lensmodel()
    x_image, y_image = lens_system.solve_lens_equation(lensmodel, kwargs)
    magnifications = lens_system.quasar_magnification(x_image, y_image, lensmodel, kwargs)

    return x_image, y_image, magnifications

def forward_model(x_image, y_image, magnfications, macromodel_class, source_size_pc,
                  pyhalo_instance, pyhalo_kwargs, realization_type, opt_routine, constrain_params,
                  verbose, optimizer_class=None, kwargs_optimizer_init={}):

    kwargs_optimizer = {'opt_routine': opt_routine,
                        'constrain_params': constrain_params}
    data_class = LensedQuasar(x_image, y_image, magnfications)
    flux_ratios_observed = data_class.flux_ratios(0)

    realization = pyhalo_instance.render(realization_type, pyhalo_kwargs)[0]

    kwargs_quasar = {'center_x': None, 'center_y': None, 'source_fwhm_pc': source_size_pc}
    quasar = Quasar(kwargs_quasar)

    cosmo = pyhalo_instance._cosmology
    z_source = pyhalo_instance.zsource

    lens_system = QuadLensSystem(macromodel_class, z_source, quasar, realization, cosmo)
    lens_system.initialize(data_class)

    if optimizer_class is not None:
        optimizer = optimizer_class(lens_system, **kwargs_optimizer_init)
        kwargs_lens_final, lens_model_full, return_kwargs = optimizer.optimize(data_class, verbose=verbose)
    else:
        kwargs_lens_final, lens_model_full, return_kwargs = lens_system.fit(data_class, HierarchicalOptimization,
                                                                        verbose=verbose, **kwargs_optimizer)

    magnifications_fit = lens_system.quasar_magnification(data_class.x, data_class.y,
                                      lens_model_full, kwargs_lens_final, normed=True)
    flux_ratios_fit = magnifications_fit[1:]/magnifications_fit[0]

    summary_statistic = np.sqrt(np.sum((1 - flux_ratios_fit/flux_ratios_observed)**2))

    lens_model_names, lens_redshift_list, kwargs_lens_full, _, _ = lens_system.get_lenstronomy_args(True)
    lens_model_names_macro, lens_redshift_list_macro, kwargs_lens_macro, _, _ = lens_system.get_lenstronomy_args(False)

    macromodel_params = None

    for i, kwarg_set in enumerate(kwargs_lens_macro):

        if lens_model_names[i] == 'SPEMD':
            kwarg_set = kwargs_e1e2_to_polar(kwarg_set)
        elif lens_model_names[i] == 'SHEAR':
            kwarg_set = kwargs_gamma1gamma2_to_polar(kwarg_set)
            external_shear = kwarg_set['shear']

        if macromodel_params is None:
            macromodel_params = kwargs_to_array(kwarg_set)
        else:
            new = kwargs_to_array(kwarg_set)
            macromodel_params = np.append(macromodel_params, new)

    lens_system.update_realization(realization)

    out_kwargs = {'summary_stat': summary_statistic,
                  'macromodel_parameters': macromodel_params,
                  'flux_ratios_fit': np.round(flux_ratios_fit, 4),
                  'flux_ratios_observed': np.round(flux_ratios_observed, 4),
                  'magnifications_fit': np.round(magnifications_fit, 4),
                  'external_shear': external_shear,
                  'lens_system_optimized': lens_system}

    return out_kwargs
