import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.dpie_nie import DPIENIE
from lenstronomy.LensModel.Profiles.pseudo_jaffe import PseudoJaffe


class TestDPIENIE(object):
    def setup_method(self):
        self.profile = DPIENIE()

    def test_lens_model_constructible(self):
        model = LensModel(["DPIE_NIE"])
        x = np.array([0.2, 1.1])
        y = np.array([0.4, -0.3])
        kwargs = [
            {
                "sigma0": 1.2,
                "Ra": 0.15,
                "Rs": 3.0,
                "e1": 0.05,
                "e2": -0.02,
                "center_x": 0.1,
                "center_y": -0.1,
            }
        ]

        beta_x, beta_y = model.ray_shooting(x, y, kwargs)

        assert np.isfinite(beta_x).all()
        assert np.isfinite(beta_y).all()

    def test_spherical_derivatives_match_pseudo_jaffe(self):
        x = np.array([0.2, 1.0, 3.0, 4.0])
        y = np.array([0.4, 2.0, 1.0, -1.0])
        sigma0 = 1.0
        Ra = 0.5
        Rs = 0.8
        f_x, f_y = self.profile.derivatives(x, y, sigma0, Ra, Rs, 0.0, 0.0)
        f_x_ref, f_y_ref = PseudoJaffe().derivatives(x, y, sigma0, Ra, Rs)

        npt.assert_allclose(f_x, f_x_ref, rtol=1.0e-8, atol=1.0e-8)
        npt.assert_allclose(f_y, f_y_ref, rtol=1.0e-8, atol=1.0e-8)

    def test_elliptical_outputs_are_finite(self):
        x = np.array([0.2, 1.0, 3.0])
        y = np.array([0.4, 2.0, -1.0])
        kwargs = {
            "sigma0": 1.2,
            "Ra": 0.15,
            "Rs": 3.0,
            "e1": 0.05,
            "e2": -0.02,
            "center_x": 0.1,
            "center_y": -0.1,
        }

        values = self.profile.function(x, y, **kwargs)
        hessian = self.profile.hessian(x, y, **kwargs)

        assert np.isfinite(values).all()
        for item in hessian:
            assert np.isfinite(item).all()


if __name__ == "__main__":
    pytest.main()
