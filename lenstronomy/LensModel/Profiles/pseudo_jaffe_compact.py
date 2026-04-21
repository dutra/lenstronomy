import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["PseudoJaffeCompact"]


class PseudoJaffeCompact(LensProfileBase):
    """Compact-support pseudo-Jaffe profile.

    The profile matches the standard pseudo-Jaffe solution in the inner region and
    replaces the outer shell with a quintic continuation so that the radial
    deflection angle and its first two radial derivatives vanish at ``Rs``.
    """

    param_names = ["sigma0", "Ra", "Rs", "center_x", "center_y"]
    lower_limit_default = {
        "sigma0": 0,
        "Ra": 0,
        "Rs": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "sigma0": 10,
        "Ra": 100,
        "Rs": 100,
        "center_x": 100,
        "center_y": 100,
    }

    _s = 0.0001
    _match_frac = 0.8

    def __init__(self):
        LensProfileBase.__init__(self)

    @classmethod
    def _sort_ra_rs(cls, Ra, Rs):
        if Ra >= Rs:
            Ra, Rs = Rs, Ra
        if Ra < 0.00000001:
            Ra = 0.00000001
        if Rs < Ra + 0.00000001:
            Rs += 0.00000002
        return Ra, Rs

    @staticmethod
    def _g(u):
        return u / (1 + np.sqrt(1 + u**2))

    @staticmethod
    def _g_prime(u):
        sqrt_term = np.sqrt(1 + u**2)
        return 1.0 / (sqrt_term * (1.0 + sqrt_term))

    @staticmethod
    def _g_second(u):
        sqrt_term = np.sqrt(1 + u**2)
        return -u * (1.0 + 2.0 * sqrt_term) / (sqrt_term**3 * (1.0 + sqrt_term) ** 2)

    @classmethod
    def alpha_r_standard(cls, r, sigma0, Ra, Rs):
        Ra, Rs = cls._sort_ra_rs(Ra, Rs)
        return 2 * sigma0 * Ra * Rs / (Rs - Ra) * (cls._g(r / Ra) - cls._g(r / Rs))

    @classmethod
    def d_alpha_dr_standard(cls, r, sigma0, Ra, Rs):
        Ra, Rs = cls._sort_ra_rs(Ra, Rs)
        prefactor = 2 * sigma0 * Ra * Rs / (Rs - Ra)
        return prefactor * (cls._g_prime(r / Ra) / Ra - cls._g_prime(r / Rs) / Rs)

    @classmethod
    def d2_alpha_dr2_standard(cls, r, sigma0, Ra, Rs):
        Ra, Rs = cls._sort_ra_rs(Ra, Rs)
        prefactor = 2 * sigma0 * Ra * Rs / (Rs - Ra)
        return prefactor * (cls._g_second(r / Ra) / Ra**2 - cls._g_second(r / Rs) / Rs**2)

    @classmethod
    def potential_standard(cls, r, sigma0, Ra, Rs):
        Ra, Rs = cls._sort_ra_rs(Ra, Rs)
        return (
            -2
            * sigma0
            * Ra
            * Rs
            / (Rs - Ra)
            * (
                np.sqrt(Rs**2 + r**2)
                - np.sqrt(Ra**2 + r**2)
                + Ra * np.log(Ra + np.sqrt(Ra**2 + r**2))
                - Rs * np.log(Rs + np.sqrt(Rs**2 + r**2))
            )
        )

    @classmethod
    def _quintic_coefficients(cls, alpha_match, d_alpha_match, d2_alpha_match, delta):
        m0 = delta * d_alpha_match
        n0 = delta**2 * d2_alpha_match
        c0 = alpha_match
        c1 = m0
        c2 = 0.5 * n0
        c3 = -10.0 * alpha_match - 6.0 * m0 - 1.5 * n0
        c4 = 15.0 * alpha_match + 8.0 * m0 + 1.5 * n0
        c5 = -6.0 * alpha_match - 3.0 * m0 - 0.5 * n0
        return c0, c1, c2, c3, c4, c5

    @classmethod
    def _shell_values(cls, t, coeffs, delta):
        c0, c1, c2, c3, c4, c5 = coeffs
        alpha = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
        d_alpha_dt = c1 + 2.0 * c2 * t + 3.0 * c3 * t**2 + 4.0 * c4 * t**3 + 5.0 * c5 * t**4
        d2_alpha_dt2 = 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t**2 + 20.0 * c5 * t**3
        potential_offset = delta * (
            c0 * t
            + c1 * t**2 / 2.0
            + c2 * t**3 / 3.0
            + c3 * t**4 / 4.0
            + c4 * t**5 / 5.0
            + c5 * t**6 / 6.0
        )
        return alpha, d_alpha_dt / delta, d2_alpha_dt2 / delta**2, potential_offset

    @classmethod
    def _radial_quantities(cls, r, sigma0, Ra, Rs):
        Ra, Rs = cls._sort_ra_rs(Ra, Rs)
        r_match = cls._match_frac * Rs
        delta = Rs - r_match
        alpha_standard = cls.alpha_r_standard(r, sigma0, Ra, Rs)
        d_alpha_standard = cls.d_alpha_dr_standard(r, sigma0, Ra, Rs)
        d2_alpha_standard = cls.d2_alpha_dr2_standard(r, sigma0, Ra, Rs)
        phi_standard = cls.potential_standard(r, sigma0, Ra, Rs)

        alpha_match = cls.alpha_r_standard(r_match, sigma0, Ra, Rs)
        d_alpha_match = cls.d_alpha_dr_standard(r_match, sigma0, Ra, Rs)
        d2_alpha_match = cls.d2_alpha_dr2_standard(r_match, sigma0, Ra, Rs)
        phi_match = cls.potential_standard(r_match, sigma0, Ra, Rs)
        coeffs = cls._quintic_coefficients(alpha_match, d_alpha_match, d2_alpha_match, delta)

        t = (r - r_match) / delta
        alpha_shell, d_alpha_shell, d2_alpha_shell, phi_offset = cls._shell_values(t, coeffs, delta)
        phi_shell = phi_match + phi_offset
        _, _, _, phi_cut_offset = cls._shell_values(np.ones_like(r), coeffs, delta)
        phi_cut = phi_match + phi_cut_offset

        shell_mask = (r > r_match) & (r < Rs)
        outer_mask = r >= Rs
        alpha = np.where(shell_mask, alpha_shell, alpha_standard)
        alpha = np.where(outer_mask, 0.0, alpha)
        d_alpha = np.where(shell_mask, d_alpha_shell, d_alpha_standard)
        d_alpha = np.where(outer_mask, 0.0, d_alpha)
        d2_alpha = np.where(shell_mask, d2_alpha_shell, d2_alpha_standard)
        d2_alpha = np.where(outer_mask, 0.0, d2_alpha)
        phi = np.where(shell_mask, phi_shell, phi_standard)
        phi = np.where(outer_mask, phi_cut, phi)
        return alpha, d_alpha, d2_alpha, phi

    @classmethod
    def alpha_r(cls, r, sigma0, Ra, Rs):
        alpha, _, _, _ = cls._radial_quantities(r, sigma0, Ra, Rs)
        return alpha

    @classmethod
    def d_alpha_dr(cls, r, sigma0, Ra, Rs):
        _, d_alpha, _, _ = cls._radial_quantities(r, sigma0, Ra, Rs)
        return d_alpha

    @classmethod
    def d2_alpha_dr2(cls, r, sigma0, Ra, Rs):
        _, _, d2_alpha, _ = cls._radial_quantities(r, sigma0, Ra, Rs)
        return d2_alpha

    def function(self, x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        _, _, _, phi = self._radial_quantities(r, sigma0, Ra, Rs)
        return phi

    def derivatives(self, x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        r_safe = np.maximum(r, self._s)
        alpha_r = self.alpha_r(r_safe, sigma0, Ra, Rs)
        f_x = alpha_r * x_ / r_safe
        f_y = alpha_r * y_ / r_safe
        zeros = np.zeros_like(f_x)
        return np.where(r == 0, zeros, f_x), np.where(r == 0, zeros, f_y)

    def hessian(self, x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        r_safe = np.maximum(r, self._s)
        alpha = self.alpha_r(r_safe, sigma0, Ra, Rs)
        d_alpha = self.d_alpha_dr(r_safe, sigma0, Ra, Rs)
        f_xx = d_alpha * x_**2 / r_safe**2 + alpha * y_**2 / r_safe**3
        f_yy = d_alpha * y_**2 / r_safe**2 + alpha * x_**2 / r_safe**3
        f_xy = x_ * y_ * (d_alpha / r_safe**2 - alpha / r_safe**3)
        zeros = np.zeros_like(f_xx)
        return (
            np.where(r == 0, zeros, f_xx),
            np.where(r == 0, zeros, f_xy),
            np.where(r == 0, zeros, f_xy),
            np.where(r == 0, zeros, f_yy),
        )

    def mass_3d_lens(self, r, sigma0, Ra, Rs):
        rho0 = self.sigma2rho(sigma0, Ra, Rs)
        return (
            4
            * np.pi
            * rho0
            * Ra**2
            * Rs**2
            / (Rs**2 - Ra**2)
            * (Rs * np.arctan(r / Rs) - Ra * np.arctan(r / Ra))
        )

    @staticmethod
    def rho2sigma(rho0, Ra, Rs):
        return np.pi * rho0 * Ra * Rs / (Rs + Ra)

    @staticmethod
    def sigma2rho(sigma0, Ra, Rs):
        return (Rs + Ra) / Ra / Rs / np.pi * sigma0
