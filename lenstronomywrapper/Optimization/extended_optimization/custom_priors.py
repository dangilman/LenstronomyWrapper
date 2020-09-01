from lenstronomy.Util.param_util import ellipticity2phi_q
import numpy as np

class ExecuteList(object):

    def __init__(self, custom_priors_list):

        self._list = custom_priors_list

    def __call__(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction):

        logL = 0
        for func in self._list:
            logL += func(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction)

        return logL

class MassFollowsLight(object):

    def __init__(self, lens_light_component_index, sigma_angle, sigma_q):

        self.lens_light_component_index = lens_light_component_index

        self.sigma_angle, self.sigma_q = sigma_angle, sigma_q

        self.linked_with_lens_model = False

        self.linked_with_lens_light_model = True

    def __call__(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction):

        angle_lens, angle_light, q_lens, q_light = self.evaluate_component(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction)

        delta_theta = angle_lens - angle_light
        delta_q = q_lens - q_light

        penalty_theta = delta_theta ** 2 / self.sigma_angle ** 2
        penalty_q = delta_q ** 2 / self.sigma_q ** 2

        return -0.5 * (penalty_theta + penalty_q)

    def evaluate_component(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction):

        assert self._eval_kwargs_lens is True

        e1_lens, e2_lens = kwargs_lens[self._idx_kwargs_lens]['e1'], kwargs_lens[self._idx_kwargs_lens]['e2']
        kwargs_light = kwargs_lens_light[self.lens_light_component_index]
        e1_light, e2_light = kwargs_light['e1'], kwargs_light['e2']

        phi_lens, q_lens = ellipticity2phi_q(e1_lens, e2_lens)
        phi_light, q_light = ellipticity2phi_q(e1_light, e2_light)

        angle_lens, angle_light = phi_lens * 180/np.pi, phi_light * 180/np.pi

        return angle_lens, angle_light, q_lens, q_light

    def set_eval(self, eval_kwargs_lens=None, eval_kwargs_source=None,
                 eval_kwargs_lens_light=None):

        self._idx_kwargs_lens = None
        self._eval_kwargs_lens = False
        self._idx_kwargs_source_light = None
        self._eval_kwargs_source_light = False
        self._idx_kwargs_lens_light = None
        self._eval_kwargs_lens_light = False

        self.eval_set = True

        if eval_kwargs_lens is not None:
            self._idx_kwargs_lens = eval_kwargs_lens
            self._eval_kwargs_lens = True

        if eval_kwargs_source is not None:
            self._idx_kwargs_source_light = eval_kwargs_source
            self._eval_kwargs_source_light = True

        if eval_kwargs_lens_light is not None:
            self._idx_kwargs_lens_light = eval_kwargs_lens_light
            self._eval_kwargs_lens_light = True

class AxisRatio(object):

    def __init__(self, target_q, target_q_sigma):

        self.target_q, self.target_q_sigma = target_q, target_q_sigma
        self.eval_set = False

        self.linked_with_lens_model = False
        self.linked_with_lens_light_model = False

    def set_eval(self, eval_kwargs_lens=None, eval_kwargs_source=None,
                 eval_kwargs_lens_light=None):

        self._idx_kwargs_lens = None
        self._eval_kwargs_lens = False
        self._idx_kwargs_source_light = None
        self._eval_kwargs_source_light = False
        self._idx_kwargs_lens_light = None
        self._eval_kwargs_lens_light = False

        self.eval_set = True

        if eval_kwargs_lens is not None:
            self._idx_kwargs_lens = eval_kwargs_lens
            self._eval_kwargs_lens = True

        if eval_kwargs_source is not None:
            self._idx_kwargs_source_light = eval_kwargs_source
            self._eval_kwargs_source_light = True

        if eval_kwargs_lens_light is not None:
            self._idx_kwargs_lens_light = eval_kwargs_lens_light
            self._eval_kwargs_lens_light = True

    def __call__(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction):

        q = self.evaluate_component(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction)

        dq = self.target_q - q

        return -0.5 * dq ** 2 / self.target_q_sigma ** 2

    def evaluate_component(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction):

        assert self.eval_set is True

        if self._eval_kwargs_lens:
            e1, e2 = kwargs_lens[self._idx_kwargs_lens]['e1'], kwargs_lens[self._idx_kwargs_lens]['e2']

        elif self._eval_kwargs_lens_light:
            e1, e2 = kwargs_source[self._idx_kwargs_lens_light]['e1'], kwargs_source[self._idx_kwargs_lens_light]['e2']

        elif self._eval_kwargs_source_light:
            e1, e2 = kwargs_lens_light[self._idx_kwargs_source_light]['e1'], kwargs_lens_light[self._idx_kwargs_source_light]['e2']

        else:
            raise Exception('MUST INITIALIZE KWARGS TO BE EVALUATED')

        _, q = ellipticity2phi_q(e1, e2)

        return q

class PositionAnglePolar(object):

    def __init__(self, target_angle, target_angle_sigma):

        self.target_angle, self.target_angle_sigma = target_angle, target_angle_sigma
        self.eval_set = False

        self.linked_with_lens_model = False
        self.linked_with_lens_light_model = False

    def set_eval(self, eval_kwargs_lens=None, eval_kwargs_source=None,
                 eval_kwargs_lens_light=None):

        self._idx_kwargs_lens = None
        self._eval_kwargs_lens = False
        self._idx_kwargs_source_light = None
        self._eval_kwargs_source_light = False
        self._idx_kwargs_lens_light = None
        self._eval_kwargs_lens_light = False

        self.eval_set = True

        if eval_kwargs_lens is not None:
            self._idx_kwargs_lens = eval_kwargs_lens
            self._eval_kwargs_lens = True

        if eval_kwargs_source is not None:
            self._idx_kwargs_source_light = eval_kwargs_source
            self._eval_kwargs_source_light = True

        if eval_kwargs_lens_light is not None:
            self._idx_kwargs_lens_light = eval_kwargs_lens_light
            self._eval_kwargs_lens_light = True

    def __call__(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction):

        angle_model = self.evaluate_component(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction)

        d_theta = self.target_angle - angle_model

        return -0.5 * d_theta ** 2 / self.target_angle_sigma

    def evaluate_component(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_special, kwargs_extinction):

        assert self.eval_set is True

        if self._eval_kwargs_lens:
            e1, e2 = kwargs_lens[self._idx_kwargs_lens]['e1'], kwargs_lens[self._idx_kwargs_lens]['e2']

        elif self._eval_kwargs_lens_light:
            e1, e2 = kwargs_source[self._idx_kwargs_lens_light]['e1'], kwargs_source[self._idx_kwargs_lens_light]['e2']

        elif self._eval_kwargs_source_light:
            e1, e2 = kwargs_lens_light[self._idx_kwargs_source_light]['e1'], kwargs_lens_light[self._idx_kwargs_source_light]['e2']

        else:
            raise Exception('MUST INITIALIZE KWARGS TO BE EVALUATED')

        phi, q = ellipticity2phi_q(e1, e2)

        angle_degrees = phi * 180 / np.pi

        return angle_degrees

