import numpy as np
import numpy.testing as npt

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.pseudo_jaffe_ellipse_potential_compact import (
    PseudoJaffeEllipsePotentialCompact,
)
from jaxtronomy.LensModel.Profiles.pseudo_jaffe_ellipse_potential_compact import (
    PseudoJaffeEllipsePotentialCompact as JaxPseudoJaffeEllipsePotentialCompact,
)


class TestPseudoJaffeEllipsePotentialCompact:
    def setup_method(self):
        self.profile = PseudoJaffeEllipsePotentialCompact()

    def test_deflection_vanishes_outside_cut_radius_in_spherical_limit(self):
        sigma0 = 1.0
        Ra, Rs = 0.2, 1.0
        e1, e2 = 0.0, 0.0
        x = np.array([1.2])
        y = np.array([0.0])

        alpha_x, alpha_y = self.profile.derivatives(x, y, sigma0, Ra, Rs, e1, e2)
        npt.assert_allclose(alpha_x, 0.0, atol=1.0e-12)
        npt.assert_allclose(alpha_y, 0.0, atol=1.0e-12)

    def test_jax_parity(self):
        sigma0 = 1.0
        Ra, Rs = 0.2, 1.0
        q, phi_g = 0.8, 0.3
        e1, e2 = param_util.phi_q2_ellipticity(phi_g, q)
        x = np.array([0.2, 0.8, 1.2])
        y = np.array([0.1, 0.4, 0.0])

        npt.assert_allclose(
            np.asarray(self.profile.function(x, y, sigma0, Ra, Rs, e1, e2)),
            np.asarray(JaxPseudoJaffeEllipsePotentialCompact.function(x, y, sigma0, Ra, Rs, e1, e2)),
            atol=1.0e-10,
        )
        np_values = self.profile.derivatives(x, y, sigma0, Ra, Rs, e1, e2)
        jax_values = JaxPseudoJaffeEllipsePotentialCompact.derivatives(x, y, sigma0, Ra, Rs, e1, e2)
        npt.assert_allclose(np.asarray(np_values[0]), np.asarray(jax_values[0]), atol=1.0e-10)
        npt.assert_allclose(np.asarray(np_values[1]), np.asarray(jax_values[1]), atol=1.0e-10)
