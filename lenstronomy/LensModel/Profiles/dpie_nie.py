import numpy as np

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.nie import NIE

__all__ = ["DPIENIE"]


class DPIENIE(LensProfileBase):
    """dPIE profile represented as the difference of two NIE profiles."""

    param_names = ["sigma0", "Ra", "Rs", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "sigma0": 0,
        "Ra": 0,
        "Rs": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "sigma0": 10,
        "Ra": 100,
        "Rs": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self._nie_inner = NIE()
        self._nie_outer = NIE()
        super(DPIENIE, self).__init__()

    def function(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        theta_E, Ra, Rs = self._param_convert(sigma0, Ra, Rs)
        f_inner = self._nie_inner.function(x, y, theta_E, e1, e2, Ra, center_x, center_y)
        f_outer = self._nie_outer.function(x, y, theta_E, e1, e2, Rs, center_x, center_y)
        return f_inner - f_outer

    def derivatives(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        theta_E, Ra, Rs = self._param_convert(sigma0, Ra, Rs)
        f_x_inner, f_y_inner = self._nie_inner.derivatives(
            x, y, theta_E, e1, e2, Ra, center_x, center_y
        )
        f_x_outer, f_y_outer = self._nie_outer.derivatives(
            x, y, theta_E, e1, e2, Rs, center_x, center_y
        )
        return f_x_inner - f_x_outer, f_y_inner - f_y_outer

    def hessian(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        theta_E, Ra, Rs = self._param_convert(sigma0, Ra, Rs)
        f_xx_inner, f_xy_inner, f_yx_inner, f_yy_inner = self._nie_inner.hessian(
            x, y, theta_E, e1, e2, Ra, center_x, center_y
        )
        f_xx_outer, f_xy_outer, f_yx_outer, f_yy_outer = self._nie_outer.hessian(
            x, y, theta_E, e1, e2, Rs, center_x, center_y
        )
        return (
            f_xx_inner - f_xx_outer,
            f_xy_inner - f_xy_outer,
            f_yx_inner - f_yx_outer,
            f_yy_inner - f_yy_outer,
        )

    @staticmethod
    def _param_convert(sigma0, Ra, Rs):
        Ra, Rs = DPIENIE._sort_ra_rs(Ra, Rs)
        theta_E = 2.0 * sigma0 * Ra * Rs / (Rs - Ra)
        return theta_E, Ra, Rs

    @staticmethod
    def _sort_ra_rs(Ra, Rs):
        if Ra >= Rs:
            Ra, Rs = Rs, Ra
        if Ra < 1.0e-8:
            Ra = 1.0e-8
        if Rs < Ra + 1.0e-8:
            Rs += 2.0e-8
        return Ra, Rs
