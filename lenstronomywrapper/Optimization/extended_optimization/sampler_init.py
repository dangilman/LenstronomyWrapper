from lenstronomywrapper.Optimization.extended_optimization.custom_priors import ExecuteList

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

    def sampler_inputs(self, reoptimize):

        self.system.source_light_model.set_reoptimize(reoptimize)
        self.system.lens_light_model.set_reoptimize(reoptimize)
        self.system.macromodel.set_reoptimize(reoptimize)

        kwargs_model = self.kwargs_model
        kwargs_numerics = self.kwargs_numerics
        kwargs_constraints = self.kwargs_constraints
        kwargs_likelihood = self.kwargs_likelihood
        image_band = [self.lens_data_class.kwargs_data, self.lens_data_class.kwargs_psf, kwargs_numerics]
        multi_band_list = [image_band]

        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        if self._time_delay_likelihood:
            kwargs_data_joint['time_delays_measured'] = self.dt_measured
            kwargs_data_joint['time_delays_uncertainties'] = self.dt_sigma

        kwargs_lens_init = self.system.macromodel.kwargs
        kwargs_source_init = self.sourcelight_instance.param_init
        kwargs_lens_light_init = self.lenslight_instance.param_init
        kwargs_ps_init = self.pointsource_instance.param_init

        # initial spread in parameter estimation #
        kwargs_lens_sigma = self.system.macromodel.param_sigma
        kwargs_source_sigma = self.sourcelight_instance.param_sigma
        kwargs_lens_light_sigma = self.lenslight_instance.param_sigma
        kwargs_ps_sigma = self.pointsource_instance.param_sigma

        # hard bound lower limit in parameter space #
        kwargs_lower_lens = self.system.macromodel.param_lower
        kwargs_lower_source = self.sourcelight_instance.param_lower
        kwargs_lower_lens_light = self.lenslight_instance.param_lower
        kwargs_lower_ps = self.pointsource_instance.param_lower

        # hard bound upper limit in parameter space #
        kwargs_upper_lens = self.system.macromodel.param_upper
        kwargs_upper_source = self.sourcelight_instance.param_upper
        kwargs_upper_lens_light = self.lenslight_instance.param_upper
        kwargs_upper_ps = self.pointsource_instance.param_upper

        fixed_models_lens = self.system.macromodel.fixed_models
        fixed_models_source_light = self.sourcelight_instance.fixed_models
        fixed_models_lens_light = self.lenslight_instance.fixed_models
        fixed_models_ps = self.pointsource_instance.fixed_models

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

        joint_lens_with_light = []

        k_lens = 0

        for k, component in enumerate(self.system.macromodel.components):

            if component.concentric_with_lens_light is False or component.concentric_with_lens_light is None:
                pass

            else:
                i_light = component.concentric_with_lens_light
                joint_lens_with_light.append([i_light, k_lens, ['center_x', 'center_y']])

            k_lens += component.n_models

        return joint_lens_with_light

    @property
    def linked_parameters_lensmodel_lensmodel(self):

        joint_lens_with_lens = []

        k_lens = 0

        for k, component in enumerate(self.system.macromodel.components):

            if component.concentric_with_lens_model is False or component.concentric_with_lens_model is None:
                pass

            else:
                idx = component.concentric_with_lens_model
                joint_lens_with_lens.append([idx, k_lens, ['center_x', 'center_y']])

            k_lens += component.n_models

        return joint_lens_with_lens

    @property
    def linked_parameters_source_source(self):

        joint_source_with_source, joint_source_with_point_source = [], []

        for k, source_model in enumerate(self.system.source_light_model.components):

            if source_model.concentric_with_source is False or source_model.concentric_with_source is None:
                continue
            else:

                idx = source_model.concentric_with_source

                if idx == 0:
                    joint_source_with_point_source.append([0, k])
                else:
                    joint_source_with_source.append([idx, k, ['center_x', 'center_y']])

        return joint_source_with_point_source, joint_source_with_source

    @property
    def linked_parameters_lens_light_lens_light(self):

        joint_lens_light_with_lens_light = []

        k_light = 0

        for k, component in enumerate(self.system.lens_light_model.components):

            if component.concentric_with_lens_light is False or component.concentric_with_lens_light is None:
                pass

            else:
                i_light = component.concentric_with_lens_light
                joint_lens_light_with_lens_light.append([k_light, i_light, ['center_x', 'center_y']])

            k_light += component.n_models

        return joint_lens_light_with_lens_light

    @property
    def custom_LogLike(self):

        custom_functions = []

        custom_functions += self._custom_LogLikeMass
        custom_functions += self._custom_LogLikeLight

        if len(custom_functions) > 0:
            custom_loglike = ExecuteList(custom_functions)
        else:
            custom_loglike = None

        return custom_loglike

    @property
    def _custom_LogLikeLight(self):

        custom_functions = []

        k_light = 0

        for k, component in enumerate(self.system.lens_light_model.components):

            if component.custom_prior is False or component.custom_prior is None:
                pass

            else:

                if component.custom_prior.linked_with_lens_light_model:
                    raise Exception('Custom priors linked with other light models not '
                                    'implemented for light model classes')

                elif component.custom_prior.linked_with_lens_model:

                    raise Exception('Custom priors linked with lens mass models not '
                                    'implemented for light model classes, try adding the custom prior to the '
                                    'lens mass model class instead.')

                else:
                    component.custom_prior.set_eval(eval_kwargs_lens_light=k_light)

                custom_functions.append(component.custom_prior)

            k_light += component.n_models

        return custom_functions

    @property
    def _custom_LogLikeMass(self):

        custom_functions = []

        k_lens = 0

        for k, component in enumerate(self.system.macromodel.components):

            if component.custom_prior is False or component.custom_prior is None:
                pass

            else:

                if component.custom_prior.linked_with_lens_light_model:
                    light_component_index = component.custom_prior.lens_light_component_index
                    i_light = self.kwargs_index_from_component_index(self.system.lens_light_model,
                                                                     light_component_index)
                    component.custom_prior.set_eval(eval_kwargs_lens=k_lens,
                                                    eval_kwargs_lens_light=i_light)

                elif component.custom_prior.linked_with_lens_model:

                    lens_component_index = component.custom_prior.lens_component_index
                    i_lens = self.kwargs_index_from_component_index(self.system.macromodel,
                                                                    lens_component_index)
                    component.custom_prior.set_eval(eval_kwargs_lens=i_lens)

                else:
                    component.custom_prior.set_eval(eval_kwargs_lens=k_lens)

                custom_functions.append(component.custom_prior)

            k_lens += component.n_models

        return custom_functions

    @staticmethod
    def kwargs_index_from_component_index(model_class, model_idx):

        i = 0
        for idx, component in enumerate(model_class.components):
            if idx == model_idx:
                break
            i += component.n_models
        return i

    @property
    def kwargs_model(self):

        lens_model_list = self.system.macromodel.lens_model_list
        lens_redshift_list = self.system.macromodel.redshift_list
        source_light_model_list = self.sourcelight_instance.light_model_list
        point_source_model_list = self.pointsource_instance.point_source_list
        lens_light_model_list = self.lenslight_instance.light_model_list
        source_redshift_list = self.sourcelight_instance.redshift_list

        additional_images_list = [False]
        fixed_magnification_list = [False]

        if source_redshift_list is not None:
            assert len(source_redshift_list) == len(source_light_model_list)

        kwargs = {'lens_model_list': lens_model_list,
                  'source_light_model_list': source_light_model_list,
                  'lens_light_model_list': lens_light_model_list,
                  'point_source_model_list': point_source_model_list,
                  'additional_images_list': additional_images_list,
                  'fixed_magnification_list': fixed_magnification_list,
                  'z_lens': self.system.zlens, 'z_source': self.system.zsource,
                  'multi_plane': True, 'lens_redshift_list': lens_redshift_list,
                  'source_redshift_list': source_redshift_list}

        return kwargs

    @property
    def kwargs_likelihood(self):

        check_bounds = True
        force_no_add_image = False
        source_marg = False
        image_position_uncertainty = self.lens_data_class.image_sigma
        check_matched_source_position = True
        source_position_tolerance = 0.001
        source_position_sigma = 0.001

        custom_loglike = self.custom_LogLike

        kwargs = {'check_bounds': check_bounds,
                  'force_no_add_image': force_no_add_image,
                  'source_marg': source_marg,
                  'image_position_likelihood': False,
                  'image_position_uncertainty': image_position_uncertainty,
                  'check_matched_source_position': check_matched_source_position,
                  'source_position_tolerance': source_position_tolerance,
                  'source_position_sigma': source_position_sigma,
                  'prior_lens': self.lens_priors,
                  'prior_lens_light': self.light_priors,
                  'time_delay_likelihood': self._time_delay_likelihood,
                  'image_likelihood_mask_list': [self.lens_data_class.likelihood_mask],
                  'custom_logL_addition': custom_loglike}

        return kwargs

    @property
    def kwargs_constraints(self):

        joint_source_with_point_source, joint_source_with_source = self.linked_parameters_source_source

        joint_lens_with_lens = self.linked_parameters_lensmodel_lensmodel

        joint_lens_with_light = self.linked_parameters_lensmodel_lightmodel

        joint_lens_light_with_lens_light = self.linked_parameters_lens_light_lens_light

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

        kwargs = {'num_point_source_list': [nimg], 'solver_type': solver_type,
                  'Ddt_sampling': Ddt_sampling}

        if len(joint_source_with_point_source) > 0:
            kwargs['joint_source_with_point_source'] = joint_source_with_point_source
        if len(joint_source_with_source) > 0:
            kwargs['joint_source_with_source'] = joint_source_with_source
        if len(joint_lens_with_lens) > 0:
            kwargs['joint_lens_with_lens'] = joint_lens_with_lens
        if len(joint_lens_with_light) > 0:
            kwargs['joint_lens_with_light'] = joint_lens_with_light
        if len(joint_lens_light_with_lens_light):
            kwargs['joint_lens_light_with_lens_light'] = joint_lens_light_with_lens_light

        return kwargs

    def init_source_model(self):

        pass

    def init_lens_light(self):

        pass
