import numpy as np
import numpy.testing as npt

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.pseudo_jaffe import PseudoJaffe
from lenstronomy.LensModel.Profiles.pseudo_jaffe_compact import PseudoJaffeCompact
from jaxtronomy.LensModel.lens_model import LensModel as JaxLensModel
from jaxtronomy.LensModel.lens_model_bulk import LensModelBulk as JaxLensModelBulk
from jaxtronomy.LensModel.Profiles.pseudo_jaffe_compact import (
    PseudoJaffeCompact as JaxPseudoJaffeCompact,
)


class TestPseudoJaffeCompact:
    def setup_method(self):
        self.profile = PseudoJaffeCompact()
        self.reference = PseudoJaffe()

    def test_inner_region_matches_original_profile(self):
        sigma0 = 1.0
        Ra, Rs = 0.2, 1.0
        x = np.array([0.25])
        y = np.array([0.15])

        npt.assert_allclose(
            self.profile.function(x, y, sigma0, Ra, Rs),
            self.reference.function(x, y, sigma0, Ra, Rs),
            rtol=0,
            atol=1.0e-10,
        )
        npt.assert_allclose(
            self.profile.derivatives(x, y, sigma0, Ra, Rs),
            self.reference.derivatives(x, y, sigma0, Ra, Rs),
            rtol=0,
            atol=1.0e-10,
        )

    def test_deflection_and_hessian_vanish_outside_cut_radius(self):
        sigma0 = 1.0
        Ra, Rs = 0.2, 1.0
        x = np.array([1.2, 2.0])
        y = np.array([0.0, 0.0])

        alpha_x, alpha_y = self.profile.derivatives(x, y, sigma0, Ra, Rs)
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, sigma0, Ra, Rs)
        phi_cut = self.profile.function(np.array([Rs]), np.array([0.0]), sigma0, Ra, Rs)
        phi_outer = self.profile.function(x, y, sigma0, Ra, Rs)

        npt.assert_allclose(alpha_x, 0.0, atol=1.0e-12)
        npt.assert_allclose(alpha_y, 0.0, atol=1.0e-12)
        npt.assert_allclose(f_xx, 0.0, atol=1.0e-12)
        npt.assert_allclose(f_xy, 0.0, atol=1.0e-12)
        npt.assert_allclose(f_yx, 0.0, atol=1.0e-12)
        npt.assert_allclose(f_yy, 0.0, atol=1.0e-12)
        npt.assert_allclose(phi_outer, phi_cut[0], atol=1.0e-12)

    def test_cut_radius_constraints_hold(self):
        sigma0 = 1.0
        Ra, Rs = 0.2, 1.0
        alpha_cut = self.profile.alpha_r(np.array([Rs]), sigma0, Ra, Rs)
        d_alpha_cut = self.profile.d_alpha_dr(np.array([Rs]), sigma0, Ra, Rs)
        d2_alpha_cut = self.profile.d2_alpha_dr2(np.array([Rs]), sigma0, Ra, Rs)

        npt.assert_allclose(alpha_cut, 0.0, atol=1.0e-12)
        npt.assert_allclose(d_alpha_cut, 0.0, atol=1.0e-10)
        npt.assert_allclose(d2_alpha_cut, 0.0, atol=1.0e-8)

    def test_jax_parity(self):
        sigma0 = 1.0
        Ra, Rs = 0.2, 1.0
        x = np.array([0.25, 0.9, 1.2])
        y = np.array([0.15, 0.1, 0.0])

        npt.assert_allclose(
            np.asarray(self.profile.function(x, y, sigma0, Ra, Rs)),
            np.asarray(JaxPseudoJaffeCompact.function(x, y, sigma0, Ra, Rs)),
            atol=1.0e-10,
        )
        np_values = self.profile.derivatives(x, y, sigma0, Ra, Rs)
        jax_values = JaxPseudoJaffeCompact.derivatives(x, y, sigma0, Ra, Rs)
        npt.assert_allclose(np.asarray(np_values[0]), np.asarray(jax_values[0]), atol=1.0e-10)
        npt.assert_allclose(np.asarray(np_values[1]), np.asarray(jax_values[1]), atol=1.0e-10)

        np_hessian = self.profile.hessian(x, y, sigma0, Ra, Rs)
        jax_hessian = JaxPseudoJaffeCompact.hessian(x, y, sigma0, Ra, Rs)
        for np_item, jax_item in zip(np_hessian, jax_hessian):
            npt.assert_allclose(np.asarray(np_item), np.asarray(jax_item), atol=1.0e-9)

    def test_profile_registration(self):
        np_model = LensModel(["PJAFFE_COMPACT"])
        jax_model = JaxLensModel(["PJAFFE_COMPACT"])

        alpha_np = np_model.alpha(
            np.array([0.25]),
            np.array([0.15]),
            [{"sigma0": 1.0, "Ra": 0.2, "Rs": 1.0}],
        )
        alpha_jax = jax_model.alpha(
            np.array([0.25]),
            np.array([0.15]),
            [{"sigma0": 1.0, "Ra": 0.2, "Rs": 1.0}],
        )

        npt.assert_allclose(np.asarray(alpha_np[0]), np.asarray(alpha_jax[0]), atol=1.0e-10)
        npt.assert_allclose(np.asarray(alpha_np[1]), np.asarray(alpha_jax[1]), atol=1.0e-10)

    def test_jax_compact_skip_masks_points_outside_support(self):
        kwargs = [
            {
                "sigma0": 1.0,
                "Ra": 0.2,
                "Rs": 1.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]
        x = np.array([0.25, 1.2])
        y = np.array([0.15, 0.0])

        compact_model = JaxLensModel(
            ["PJAFFE_ELLIPSE_POTENTIAL_COMPACT"],
            profile_kwargs_list=[{"compact_skip_factor": 1.0}],
        )
        baseline_model = JaxLensModel(["PJAFFE_ELLIPSE_POTENTIAL_COMPACT"])

        alpha_compact = compact_model.alpha(x, y, kwargs)
        alpha_baseline = baseline_model.alpha(x, y, kwargs)

        npt.assert_allclose(np.asarray(alpha_compact[0]), np.asarray(alpha_baseline[0]), atol=1.0e-10)
        npt.assert_allclose(np.asarray(alpha_compact[1]), np.asarray(alpha_baseline[1]), atol=1.0e-10)
        npt.assert_allclose(np.asarray(alpha_compact[0][1]), 0.0, atol=1.0e-12)
        npt.assert_allclose(np.asarray(alpha_compact[1][1]), 0.0, atol=1.0e-12)

    def test_jax_bulk_compact_skip_masks_far_field_components(self):
        lens_model_list = ["PJAFFE_ELLIPSE_POTENTIAL_COMPACT"]
        kwargs_lens = [
            {
                "sigma0": 1.0,
                "Ra": 0.2,
                "Rs": 1.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]
        x = np.array([0.25, 1.2])
        y = np.array([0.15, 0.0])

        bulk_model = JaxLensModelBulk(
            unique_lens_model_list=lens_model_list,
            multi_plane=False,
            profile_kwargs_list=[{"compact_skip_factor": 1.0}],
        )
        packed_kwargs = bulk_model.prepare_ray_shooting_kwargs(lens_model_list, kwargs_lens)
        beta_x, beta_y = bulk_model.ray_shooting(x, y, packed_kwargs)

        baseline_model = JaxLensModel(["PJAFFE_ELLIPSE_POTENTIAL_COMPACT"])
        alpha_x, alpha_y = baseline_model.alpha(x, y, kwargs_lens)
        npt.assert_allclose(np.asarray(beta_x), x - np.asarray(alpha_x), atol=1.0e-10)
        npt.assert_allclose(np.asarray(beta_y), y - np.asarray(alpha_y), atol=1.0e-10)
        npt.assert_allclose(np.asarray(beta_x[1]), x[1], atol=1.0e-12)
        npt.assert_allclose(np.asarray(beta_y[1]), y[1], atol=1.0e-12)

    def test_original_jax_profile_is_unaffected_by_compact_skip_metadata(self):
        x = np.array([1.2])
        y = np.array([0.0])
        kwargs = [
            {
                "sigma0": 1.0,
                "Ra": 0.2,
                "Rs": 1.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]

        original_model = JaxLensModel(["PJAFFE_ELLIPSE_POTENTIAL"])
        compact_model = JaxLensModel(
            ["PJAFFE_ELLIPSE_POTENTIAL_COMPACT"],
            profile_kwargs_list=[{"compact_skip_factor": 1.0}],
        )

        alpha_original = original_model.alpha(x, y, kwargs)
        alpha_compact = compact_model.alpha(x, y, kwargs)

        assert not np.allclose(np.asarray(alpha_original[0]), 0.0)
        npt.assert_allclose(np.asarray(alpha_compact[0]), 0.0, atol=1.0e-12)
