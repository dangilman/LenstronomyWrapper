import numpy as np
from lenstronomy.Util.param_util import ellipticity2phi_q, phi_q2_ellipticity
from lenstronomy.LensModel.QuadOptimizer.param_manager import PowerLawFixedShear, PowerLawFreeShear

class PowerLawFreeShearAxisRatioPen(PowerLawFreeShear):

    """
    This class implements a fit of EPL + external shear with every parameter except the power law slope allowed to vary
    """

    def param_chi_square_penalty(self, args):

        e1, e2 = args[3], args[4]
        q = 1 - np.sqrt(e1**2 + e2**2)
        if q < 0.6:
            return np.inf
        else:
            return 0.

class PowerLawFixedShearAxisRatioPen(PowerLawFixedShear):

    """
    This class implements a fit of EPL + external shear with every parameter except the power law slope allowed to vary
    """

    def param_chi_square_penalty(self, args):

        e1, e2 = args[3], args[4]
        q = 1 - np.sqrt(e1**2 + e2**2)
        if q < 0.6:
            return np.inf
        else:
            return 0.

class FixedNFWShearBulgeDisk(object):

    def __init__(self, kwargs_lens_init, n_sersic, alphaRs, Rs, q_bulge, q_nfw_min=0.8):

        self.kwargs_lens = kwargs_lens_init
        self._n_sersic = n_sersic
        self._host_alphaRs = alphaRs
        self._host_Rs = Rs
        self.q_bulge = q_bulge
        self.q_nfw_min = q_nfw_min

    def param_chi_square_penalty(self, args):

        (center_x, center_y, e1_nfw, e2_nfw, g1, g2, k_eff_bulge, R_bulge, e1_bulge,
         e2_bulge, R_disk) = args

        _, q_nfw = ellipticity2phi_q(e1_nfw, e2_nfw)

        if k_eff_bulge < 0:
            return np.inf
        if q_nfw < self.q_nfw_min:
            return np.inf
        else:
            return 0.

    @property
    def to_vary_index(self):
        return 4

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (center_x, center_y, e1_nfw, e2_nfw, g1, g2, k_eff_bulge, R_bulge, e1_bulge,
         e2_bulge, R_disk) = args

        kwargs_nfw = {'alpha_Rs': self._host_alphaRs, 'Rs': self._host_Rs, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1_nfw, 'e2': e2_nfw}
        kwargs_shear = {'gamma1': g1, 'gamma2': g2}

        phi_bulge, _ = ellipticity2phi_q(e1_bulge, e2_bulge)
        e1_bulge, e2_bulge = phi_q2_ellipticity(phi_bulge, self.q_bulge)
        kwargs_bulge = {'k_eff': k_eff_bulge, 'R_sersic': R_bulge, 'n_sersic': self._n_sersic,
                        'e1': e1_bulge, 'e2': e2_bulge, 'center_x': center_x, 'center_y': center_y}

        kwargs_disk = self.kwargs_lens[3]
        kwargs_disk['center_x'] = center_x
        kwargs_disk['center_y'] = center_y

        kwargs_disk['R_sersic'] = R_disk

        self.kwargs_lens[0] = kwargs_nfw
        self.kwargs_lens[1] = kwargs_shear
        self.kwargs_lens[2] = kwargs_bulge
        self.kwargs_lens[3] = kwargs_disk

        return self.kwargs_lens

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        kwargs_nfw = kwargs[0]
        center_x = kwargs_nfw['center_x']
        center_y = kwargs_nfw['center_y']
        e1_nfw = kwargs_nfw['e1']
        e2_nfw = kwargs_nfw['e2']

        kwargs_shear = kwargs[1]
        g1, g2 = kwargs_shear['gamma1'], kwargs_shear['gamma2']

        kwargs_bulge = kwargs[2]
        k_eff_bulge = kwargs_bulge['k_eff']
        R_bulge = kwargs_bulge['R_sersic']
        e1_bulge, e2_bulge = kwargs_bulge['e1'], kwargs_bulge['e2']

        kwargs_disk = kwargs[3]
        R_disk = kwargs_disk['R_sersic']

        args = (center_x, center_y, e1_nfw, e2_nfw, g1, g2, k_eff_bulge, R_bulge, e1_bulge,
         e2_bulge, R_disk)

        return args

    def bounds(self, re_optimize, scale=1.):

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            center_shift = 0.01
            e_shift = 0.05
            g_shift = 0.025
            keff_shift = 0.5
            r_sersic_shift = 0.1

        else:
            center_shift = 0.2
            e_shift = 0.2
            g_shift = 0.05
            keff_shift = 0.1
            r_sersic_shift = 0.5

        shifts = np.array([center_shift, center_shift, e_shift, e_shift, g_shift, g_shift,
                           keff_shift, r_sersic_shift, e_shift, e_shift, r_sersic_shift])

        low = np.array(args) - shifts * scale
        high = np.array(args) + shifts * scale
        return low, high

class FixedNFWShearBulge(object):

    def __init__(self, kwargs_lens_init, n_sersic, alphaRs, Rs):

        self.kwargs_lens = kwargs_lens_init
        self._n_sersic = n_sersic
        self._host_alphaRs = alphaRs
        self._host_Rs = Rs

    def param_chi_square_penalty(self, args):

        if args[6] < 0:
            return np.inf
        else:
            return 0.

    @property
    def to_vary_index(self):
        return 3

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        center_x = args[0]
        center_y = args[1]

        kwargs_nfw = {'alpha_Rs': self._host_alphaRs, 'Rs': self._host_Rs, 'center_x': center_x, 'center_y': center_y,
                      'e1': args[2], 'e2': args[3]}
        kwargs_shear = {'gamma1': args[4], 'gamma2': args[5]}
        kwargs_bulge = {'k_eff': args[6], 'R_sersic': args[7], 'n_sersic': self._n_sersic,
                        'e1': self.kwargs_lens[2]['e1'], 'e2': self.kwargs_lens[2]['e2'], 'center_x': center_x, 'center_y': center_y}

        self.kwargs_lens[0] = kwargs_nfw
        self.kwargs_lens[1] = kwargs_shear
        self.kwargs_lens[2] = kwargs_bulge

        return self.kwargs_lens

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        kwargs_nfw = kwargs[0]
        center_x = kwargs_nfw['center_x']
        center_y = kwargs_nfw['center_y']
        e1_nfw = kwargs_nfw['e1']
        e2_nfw = kwargs_nfw['e2']

        kwargs_shear = kwargs[1]
        g1, g2 = kwargs_shear['gamma1'], kwargs_shear['gamma2']

        kwargs_bulge = kwargs[2]
        keff = kwargs_bulge['k_eff']
        r_sersic = kwargs_bulge['R_sersic']

        args = (center_x, center_y, e1_nfw, e2_nfw, g1, g2, keff, r_sersic)
        return args

    def bounds(self, re_optimize, scale=1.):

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            center_shift = 0.01
            e_shift = 0.05
            g_shift = 0.025
            keff_shift = 0.5
            r_sersic_shift = 0.1

        else:
            center_shift = 0.2
            e_shift = 0.2
            g_shift = 0.05
            keff_shift = 0.1
            r_sersic_shift = 0.5

        shifts = np.array([center_shift, center_shift, e_shift, e_shift, g_shift, g_shift,
                           keff_shift, r_sersic_shift])

        low = np.array(args) - shifts * scale
        high = np.array(args) + shifts * scale
        return low, high

class NFWShearBulge(object):

    def __init__(self, kwargs_lens_init, n_sersic, Rs):

        self.kwargs_lens = kwargs_lens_init
        self._n_sersic = n_sersic
        self.host_rs = Rs

    def param_chi_square_penalty(self, args):

        if args[0] < 0:
            return np.inf
        elif args[7] < 0:
            return np.inf
        else:
            return 0.

    @property
    def to_vary_index(self):
        return 3

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        center_x = args[1]
        center_y = args[2]

        kwargs_nfw = {'alpha_Rs': args[0], 'Rs': self.host_rs, 'center_x': center_x, 'center_y': center_y,
                      'e1': args[3], 'e2': args[4]}
        kwargs_shear = {'gamma1': args[5], 'gamma2': args[6]}
        kwargs_bulge = {'k_eff': args[7], 'R_sersic': args[8], 'n_sersic': self._n_sersic,
                        'e1': 0., 'e2': 0., 'center_x': center_x, 'center_y': center_y}
        self.kwargs_lens[0] = kwargs_nfw
        self.kwargs_lens[1] = kwargs_shear
        self.kwargs_lens[2] = kwargs_bulge

        return self.kwargs_lens

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        kwargs_nfw = kwargs[0]
        alpha_Rs = kwargs_nfw['alpha_Rs']
        center_x = kwargs_nfw['center_x']
        center_y = kwargs_nfw['center_y']
        e1_nfw = kwargs_nfw['e1']
        e2_nfw = kwargs_nfw['e2']

        kwargs_shear = kwargs[1]
        g1, g2 = kwargs_shear['gamma1'], kwargs_shear['gamma2']

        kwargs_bulge = kwargs[2]
        keff = kwargs_bulge['k_eff']
        r_sersic = kwargs_bulge['R_sersic']

        args = (alpha_Rs, center_x, center_y, e1_nfw, e2_nfw, g1, g2, keff, r_sersic)
        return args

    def bounds(self, re_optimize, scale=1.):

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            alpha_Rs_shift = 0.05
            center_shift = 0.01
            e_shift = 0.05
            g_shift = 0.025
            keff_shift = 0.5
            r_sersic_shift = 0.1

        else:
            alpha_Rs_shift = 0.25
            center_shift = 0.2
            e_shift = 0.2
            g_shift = 0.05
            keff_shift = 0.1
            r_sersic_shift = 0.5

        shifts = np.array([alpha_Rs_shift, center_shift, center_shift, e_shift, e_shift, g_shift, g_shift,
                           keff_shift, r_sersic_shift])

        low = np.array(args) - shifts * scale
        high = np.array(args) + shifts * scale
        return low, high

