class SamplerInit(object):

    def __init__(self, lens_system_class, lens_data_class,
                 time_delay_likelihood=False, D_dt_true=None, dt_measured=None, dt_sigma=None,
                 fix_D_dt=None):

        assert len(lens_data_class.x == 4)
        self.system = lens_system_class
        self.sourcelight_instance = lens_system_class.source_light_model
        self.lenslight_instance = lens_system_class.lens_light_model
        self.pointsource_instance = lens_data_class.point_source

        self._time_delay_likelihood = time_delay_likelihood
        if self._time_delay_likelihood:
            assert D_dt_true is not None
            assert dt_measured is not None
            assert dt_sigma is not None
            assert fix_D_dt is not None

        self.D_dt_true = D_dt_true
        self.dt_measured = dt_measured
        self.dt_sigma = dt_sigma
        self._fix_D_dt = fix_D_dt

        self.lens_data_class = lens_data_class

    def _fixed_lens_options(self, kwargs_lens_init_fixed):

        param_lower = []
        param_upper = []
        param_sigma = []

        for i in range(0, len(kwargs_lens_init_fixed)):
            dict_lower, dict_upper, dict_sigma = {}, {}, {}
            for kw in kwargs_lens_init_fixed[i].keys():
                dict_lower[kw] = kwargs_lens_init_fixed[i][kw] * 0.8
                dict_upper[kw] = kwargs_lens_init_fixed[i][kw] * 1.2
                dict_sigma[kw] = 0.01

            param_lower.append(dict_lower)
            param_upper.append(dict_upper)
            param_sigma.append(dict_sigma)

        return param_lower, param_upper, param_sigma

    def sampler_inputs(self):

        kwargs_model, fixed_models_lens, kwargs_lens_init_tovary, kwargs_lens_init_fixed = self.kwargs_model

        param_lower_fixed, param_upper_fixed, param_sigma_fixed = self._fixed_lens_options(kwargs_lens_init_fixed)

        kwargs_numerics = self.kwargs_numerics
        kwargs_constraints = self.kwargs_constraints
        kwargs_likelihood = self.kwargs_likelihood
        image_band = [self.lens_data_class.kwargs_data, self.lens_data_class.kwargs_psf, kwargs_numerics]
        multi_band_list = [image_band]

        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        if self._time_delay_likelihood:
            kwargs_data_joint['time_delays_measured'] = self.dt_measured
            kwargs_data_joint['time_delays_uncertainties'] = self.dt_sigma

        kwargs_source_init = self.sourcelight_instance.param_init
        kwargs_lens_light_init = self.lenslight_instance.param_init
        kwargs_ps_init = self.pointsource_instance.param_init

        # initial spread in parameter estimation #
        kwargs_lens_sigma = self.system.macromodel.param_sigma + param_sigma_fixed
        kwargs_source_sigma = self.sourcelight_instance.param_sigma
        kwargs_lens_light_sigma = self.lenslight_instance.param_sigma
        kwargs_ps_sigma = self.pointsource_instance.param_sigma

        # hard bound lower limit in parameter space #
        kwargs_lower_lens = self.system.macromodel.param_lower + param_lower_fixed
        kwargs_lower_source = self.sourcelight_instance.param_lower
        kwargs_lower_lens_light = self.lenslight_instance.param_lower
        kwargs_lower_ps = self.pointsource_instance.param_lower

        # hard bound upper limit in parameter space #
        kwargs_upper_lens = self.system.macromodel.param_upper + param_upper_fixed
        kwargs_upper_source = self.sourcelight_instance.param_upper
        kwargs_upper_lens_light = self.lenslight_instance.param_upper
        kwargs_upper_ps = self.pointsource_instance.param_upper

        fixed_models_source_light = self.sourcelight_instance.fixed_models
        fixed_models_lens_light = self.lenslight_instance.fixed_models
        fixed_models_ps = self.pointsource_instance.fixed_models

        kwargs_lens_init = kwargs_lens_init_tovary + kwargs_lens_init_fixed
        lens_params = [kwargs_lens_init, kwargs_lens_sigma, fixed_models_lens, kwargs_lower_lens, kwargs_upper_lens]
        source_params = [kwargs_source_init, kwargs_source_sigma, fixed_models_source_light, kwargs_lower_source,
                         kwargs_upper_source]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_models_lens_light,
                             kwargs_lower_lens_light, kwargs_upper_lens_light]
        ps_params = [kwargs_ps_init, kwargs_ps_sigma, fixed_models_ps, kwargs_lower_ps, kwargs_upper_ps]

        kwargs_params = {'lens_model': lens_params,
                         'source_model': source_params,
                         'lens_light_model': lens_light_params,
                         'point_source_model': ps_params}

        if self._time_delay_likelihood:
            cosmo_params = self.cosmo_params()
            kwargs_params['special'] = cosmo_params
        return kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, multi_band_list

    def cosmo_params(self):

        fixed_cosmo = {}
        if self._fix_D_dt:
            minmax = 1e-9
        else:
            minmax = 0.9
        kwargs_cosmo_init = {'D_dt': self.D_dt_true}
        kwargs_cosmo_sigma = {'D_dt': self.D_dt_true * minmax}
        kwargs_lower_cosmo = {'D_dt': self.D_dt_true * (1-minmax)}
        kwargs_upper_cosmo = {'D_dt': self.D_dt_true * (1+minmax)}
        cosmo_params = [kwargs_cosmo_init, kwargs_cosmo_sigma, fixed_cosmo, kwargs_lower_cosmo, kwargs_upper_cosmo]
        return cosmo_params

    @property
    def fixed_lens_models(self):

        fixed_models_lens = self.system.macromodel.fixed_models
        fixed_models_source_light = self.sourcelight_instance.fixed_models
        fixed_models_lens_light = self.lenslight_instance.fixed_models
        fixed_models_ps = self.pointsource_instance.fixed_models
        return fixed_models_lens, fixed_models_source_light, fixed_models_lens_light, fixed_models_ps

    @property
    def ps_sampler_kwargs(self):

        instance = self.pointsource_instance
        return instance.param_init, instance.param_sigma, instance.param_lower, instance.param_upper

    @property
    def source_light_sampler_kwargs(self):

        instance = self.sourcelight_instance
        return instance.param_init, instance.param_sigma, instance.param_lower, instance.param_upper

    @property
    def lens_light_sampler_kwargs(self):

        instance = self.lenslight_instance
        return instance.param_init, instance.param_sigma, instance.param_lower, instance.param_upper

    @property
    def lens_sampler_kwargs(self):

        instance = self.system.macromodel
        return instance.param_init, instance.param_sigma, instance.param_lower, instance.param_upper

    @property
    def kwargs_numerics(self):
        return {'supersampling_factor': 1, 'supersampling_convolution': False}

    @property
    def num_source_model(self):
        return len(self.sourcelight_instance.light_model_list)

    @property
    def lens_priors(self):
        instance = self.system.macromodel
        return instance.priors

    @property
    def light_priors(self):
        instance = self.system.lens_light_model
        return instance.priors

    @property
    def linked_parameters_lensmodel_lightmodel(self):
        pass

    @property
    def linked_parameters_lensmodel_lensmodel(self):
        pass

    @property
    def linked_parameters_source_source(self):
        return [[0, 0]]

    @property
    def kwargs_model(self):

        lens_model_list, lens_redshift_list, kwargs_lens_init, _, _ = self.system.get_lenstronomy_args()

        fixed_params_macro = self.system.macromodel.fixed_models
        fixed_params_other = []

        n_total = len(lens_model_list)
        n_fixed_macro = len(fixed_params_macro)
        n = n_total - n_fixed_macro

        if n > 0:

            kwargs_lens_init_tovary = kwargs_lens_init[0:n_fixed_macro]
            kwargs_lens_init_fixed = kwargs_lens_init[n_fixed_macro:]

            i_start = n_fixed_macro
            for i in range(0, n):
                new_dict = {}
                keys = kwargs_lens_init[i_start+i].keys()
                for key in keys:
                    new_dict[key] = kwargs_lens_init[i_start+i][key]
                if 'kappa_ext' in keys:
                    new_dict['ra_0'] = 0.
                    new_dict['dec_0'] = 0.
                fixed_params_other.append(new_dict)

        else:

            kwargs_lens_init_tovary = kwargs_lens_init
            kwargs_lens_init_fixed = []

        fixed_params_lens = fixed_params_macro + fixed_params_other

        source_light_model_list = self.sourcelight_instance.light_model_list
        point_source_model_list = self.pointsource_instance.point_source_list
        lens_light_model_list = self.lenslight_instance.light_model_list

        additional_images_list = [False]
        fixed_magnification_list = [False]

        kwargs = {'lens_model_list': lens_model_list,
                  'source_light_model_list': source_light_model_list,
                  'lens_light_model_list': lens_light_model_list,
                  'point_source_model_list': point_source_model_list,
                  'additional_images_list': additional_images_list,
                  'fixed_magnification_list': fixed_magnification_list,
                  'z_lens': self.system.zlens, 'z_source': self.system.zsource,
                  'multi_plane': True, 'lens_redshift_list': lens_redshift_list}

        return kwargs, fixed_params_lens, kwargs_lens_init_tovary, kwargs_lens_init_fixed

    @property
    def kwargs_likelihood(self):

        check_bounds = True
        force_no_add_image = False
        source_marg = False
        image_position_uncertainty = self.lens_data_class.image_sigma
        check_matched_source_position = True
        source_position_tolerance = 0.001
        source_position_sigma = 0.001

        kwargs = {'check_bounds': check_bounds,
                  'force_no_add_image': force_no_add_image,
                  'source_marg': source_marg,
                  #'image_position_uncertainty': image_position_uncertainty,
                  'check_matched_source_position': check_matched_source_position,
                  'source_position_tolerance': source_position_tolerance,
                  'source_position_sigma': source_position_sigma,
                  'prior_lens': self.lens_priors,
                  'prior_lens_light': self.light_priors,
                  'time_delay_likelihood': self._time_delay_likelihood}

        return kwargs

    @property
    def kwargs_constraints(self):

        joint_source_with_point_source = self.linked_parameters_source_source

        if len(self.lens_data_class.x) == 4:
            solver_type = 'PROFILE_SHEAR'
            nimg = 4
        elif len(self.lens_data_class.x) == 2:
            raise Exception('two image lenses not yet implemented')
        else:
            raise Exception('only four image lenses currently implemented')

        if self._time_delay_likelihood:
            Ddt_sampling = True
        else:
            Ddt_sampling = False

        kwargs = {'joint_source_with_point_source': joint_source_with_point_source,
                  'num_point_source_list': [nimg], 'solver_type': solver_type,
                  'Ddt_sampling': Ddt_sampling}

        return kwargs

    def init_source_model(self):

        pass

    def init_lens_light(self):

        pass
