import argparse
import sys

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from playground import cluster_solver


def _parameter_specs():
    return [
        cluster_solver.ParameterSpec(
            name="u",
            sample_name="u",
            potential_id="p0",
            profile_type=81,
            field="theta_E",
            prior_kind="uniform",
            lower=-2.0,
            upper=3.0,
            step=0.1,
        ),
        cluster_solver.ParameterSpec(
            name="n",
            sample_name="n",
            potential_id="p1",
            profile_type=81,
            field="sigma",
            prior_kind="normal",
            lower=-1.5,
            upper=1.5,
            step=0.1,
            mean=0.2,
            std=0.3,
        ),
    ]


def _transformed_parameter_specs():
    return [
        cluster_solver.ParameterSpec(
            name="potfile1.sigma",
            sample_name="potfile1_sigma",
            potential_id="potfile1",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=np.log(30.0),
            std=0.3,
            component_family="scaling",
            transform_kind="log_positive",
            physical_lower=1.0,
            physical_upper=200.0,
            physical_mean=30.0,
            physical_std=10.0,
        ),
        cluster_solver.ParameterSpec(
            name="potfile1.cutkpc",
            sample_name="potfile1_cutkpc",
            potential_id="potfile1",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="cutkpc",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=np.log(19.9),
            std=0.3,
            component_family="scaling",
            transform_kind="log_offset_positive",
            physical_lower=0.1,
            physical_upper=200.0,
            physical_mean=20.0,
            physical_std=8.0,
            transform_offset=0.1,
        ),
    ]


def _roundtrip_seed_from_init_params(parameter_specs, chain_seed):
    model_for_init = cluster_solver._sample_site_model(parameter_specs)
    init_params = cluster_solver._seed_values_to_init_params(
        parameter_specs,
        [chain_seed],
        model_for_init=model_for_init,
    )
    constrained = cluster_solver.constrain_fn(
        model_for_init,
        model_args=(),
        model_kwargs={},
        params={spec.sample_name: np.asarray(init_params[spec.sample_name])[0] for spec in parameter_specs},
    )
    return np.asarray(
        [float(np.asarray(constrained[spec.sample_name])) for spec in parameter_specs],
        dtype=float,
    )


def _potfiles():
    return [
        {"id": "potfile0", "catalog_df": [object()] * 5},
        {"id": "potfile1", "catalog_df": [object()] * 3},
        {"id": "potfile2", "catalog_df": [object()] * 7},
    ]


def _args():
    return argparse.Namespace(
        continuation_sigma_scale=2.4,
        continuation_validation_top_k=2,
        validate_top_k_families=6,
        validation_approx="exact",
        sampling_engine="full",
        map_broad_seeds=5,
        map_local_refine_seeds=4,
        map_local_jitter_scale=0.05,
        seed=13,
        map_maxiter=10,
        match_tolerance_arcsec=1.5,
        active_scaling_galaxies=None,
        refresh_every=250,
        refresh_param_drift_frac=0.25,
        chains=4,
        nuts_init_top_k=0,
        nuts_init_boundary_frac=0.02,
        nuts_init_jitter_frac=0.02,
        nuts_init_dedup_distance=0.35,
        nuts_init_strategy="ranked_map",
        svi_steps=25,
        svi_learning_rate=0.01,
        warmup=5,
        samples=4,
        thin=1,
        sampler="numpyro_nuts",
        target_accept=0.9,
        max_tree_depth=6,
        smc_particles=32,
        smc_ess_threshold=0.5,
        smc_move_steps=2,
        smc_move_scale=0.03,
        smc_seed_mode="ranked_map",
        refine_with_nuts=False,
        likelihood_mode="source",
        run_name="test_run",
        par_path="fake.par",
        plot_caustics=False,
        skip_validation=False,
        skip_plots=False,
        quiet=True,
        refine_from_run_dir=None,
        profile_variant=cluster_solver.PROFILE_VARIANT_ORIGINAL,
        compact_skip_factor=1.0,
    )


def test_build_broad_starts_cover_support_and_keep_carry_forward():
    specs = _parameter_specs()
    carry_forward = np.array([1.75, -0.4], dtype=float)

    starts = cluster_solver._build_broad_starts(specs, num_broad_seeds=4, seed=7, carry_forward=carry_forward)

    assert starts[0].label == "midpoint"
    assert starts[1].label == "carry_forward"
    assert np.allclose(starts[1].values, carry_forward)
    assert len(starts) == 6
    for start in starts:
        assert start.values.shape == (2,)
        assert specs[0].lower <= start.values[0] <= specs[0].upper
        assert specs[1].lower <= start.values[1] <= specs[1].upper


def test_normalize_active_scaling_counts_defaults_to_constant():
    counts = cluster_solver._normalize_active_scaling_counts(None, _potfiles())

    assert counts == [cluster_solver.DEFAULT_ACTIVE_SCALING_GALAXIES] * 3


def test_normalize_active_scaling_counts_preserves_explicit_values():
    counts = cluster_solver._normalize_active_scaling_counts([16, 8, 4], _potfiles())

    assert counts == [16, 8, 4]


def test_normalize_active_scaling_counts_negative_means_all_per_potfile():
    counts = cluster_solver._normalize_active_scaling_counts([16, -1, 8], _potfiles())

    assert counts == [16, 3, 8]


def test_normalize_active_scaling_counts_all_negative_uses_all():
    counts = cluster_solver._normalize_active_scaling_counts([-1, -1, -1], _potfiles())

    assert counts == [5, 3, 7]


def test_normalize_active_scaling_counts_requires_one_value_per_potfile():
    try:
        cluster_solver._normalize_active_scaling_counts([16, -1], _potfiles())
    except ValueError as exc:
        assert "expects exactly one value per potfile" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected mismatched active-scaling-galaxies length to fail.")


def test_local_refinement_starts_include_best_broad_and_are_jittered():
    specs = _parameter_specs()
    ranked = [
        cluster_solver.MAPRunResult(theta=np.array([0.5, 0.2]), logprob=4.0, iterations=8, label="broad_1"),
        cluster_solver.MAPRunResult(theta=np.array([-1.0, -0.1]), logprob=3.0, iterations=7, label="broad_2"),
    ]

    starts = cluster_solver._build_local_refinement_starts(
        specs,
        ranked_results=ranked,
        num_local_seeds=3,
        jitter_scale=0.03,
        seed=11,
    )

    assert starts[0].label == "best_broad"
    assert np.allclose(starts[0].values, ranked[0].theta)
    assert len(starts) == 4
    unique_points = {tuple(np.round(start.values, 6)) for start in starts[1:]}
    assert len(unique_points) >= 2


def test_build_continuation_passes_orders_relaxed_to_final():
    args = _args()

    passes = cluster_solver._build_continuation_passes(args)

    assert [item.name for item in passes] == ["pass1_relaxed", "pass2_tightened", "pass3_final"]
    assert passes[0].sigma_scale == args.continuation_sigma_scale
    assert passes[-1].sigma_scale == 1.0
    assert passes[-1].validate_top_k_families == args.validate_top_k_families
    assert passes[-1].validation_approx == args.validation_approx


def test_build_continuation_passes_preserves_zero_validation():
    args = _args()
    args.validate_top_k_families = 0

    passes = cluster_solver._build_continuation_passes(args)

    assert [item.validate_top_k_families for item in passes] == [0, 0, 0]


def test_run_continuation_map_carries_forward_previous_best(monkeypatch):
    args = _args()
    state = object()
    calls = []

    def fake_run_map_search_pass(args, state, continuation_pass, carry_forward):
        calls.append((continuation_pass.name, None if carry_forward is None else carry_forward.copy()))
        value = np.array([float(len(calls)), -float(len(calls))])
        evaluator = type("Evaluator", (), {"timing_totals": {"map_runtime": 0.0}})()
        history = [{"stage": continuation_pass.name, "best_loglike": float(len(calls)), "iterations": 1, "restart": 0}]
        ranked = [cluster_solver.MAPRunResult(theta=value, logprob=float(len(calls)), iterations=1, label=continuation_pass.name)]
        return value, history, evaluator, ranked

    monkeypatch.setattr(cluster_solver, "_run_map_search_pass", fake_run_map_search_pass)

    best_fit, history, evaluator, ranked = cluster_solver._run_continuation_map(args, state)

    assert calls[0] == ("pass1_relaxed", None)
    assert np.allclose(calls[1][1], np.array([1.0, -1.0]))
    assert np.allclose(calls[2][1], np.array([2.0, -2.0]))
    assert np.allclose(best_fit, np.array([3.0, -3.0]))
    assert len(history) == 3
    assert ranked[0].label == "pass3_final"
    assert evaluator.timing_totals["map_runtime"] >= 0.0


def test_deduplicate_ranked_candidates_rejects_boundary_and_duplicates():
    specs = _parameter_specs()
    ranked = [
        cluster_solver.MAPRunResult(theta=np.array([2.95, 0.2]), logprob=10.0, iterations=4, label="boundary"),
        cluster_solver.MAPRunResult(theta=np.array([0.1, 0.2]), logprob=9.0, iterations=4, label="keep"),
        cluster_solver.MAPRunResult(theta=np.array([0.12, 0.205]), logprob=8.5, iterations=4, label="dup"),
        cluster_solver.MAPRunResult(theta=np.array([-1.2, -0.2]), logprob=8.0, iterations=4, label="keep_2"),
    ]

    selected, info = cluster_solver._deduplicate_ranked_candidates(
        specs,
        ranked,
        top_k=4,
        boundary_frac=0.02,
        dedup_distance=0.35,
    )

    assert [item.label for item in selected] == ["keep", "keep_2"]
    assert info["ranked_candidates_considered"] == 4
    assert info["near_boundary_rejected"] == 1
    assert info["duplicate_rejected"] == 1


def test_jitter_theta_in_support_stays_inside_bounds():
    specs = _parameter_specs()
    theta = np.array([2.9, 1.49], dtype=float)

    jittered = cluster_solver._jitter_theta_in_support(theta, specs, jitter_frac=0.05, rng=np.random.default_rng(3))

    assert jittered.shape == theta.shape
    assert specs[0].lower <= jittered[0] <= specs[0].upper
    assert specs[1].lower <= jittered[1] <= specs[1].upper


def test_seed_values_to_init_params_roundtrip_for_identity_specs():
    specs = _parameter_specs()
    chain_seeds = [
        cluster_solver.ChainSeed(values=np.array([0.0, 0.2]), source_label="a"),
        cluster_solver.ChainSeed(values=np.array([0.5, 0.1]), source_label="b"),
        cluster_solver.ChainSeed(values=np.array([-0.5, -0.1]), source_label="c"),
    ]

    model_for_init = cluster_solver._sample_site_model(specs)
    init_params = cluster_solver._seed_values_to_init_params(specs, chain_seeds, model_for_init=model_for_init)

    assert set(init_params) == {"u", "n"}
    assert tuple(init_params["u"].shape) == (3,)
    assert tuple(init_params["n"].shape) == (3,)
    for seed in chain_seeds:
        reconstructed = _roundtrip_seed_from_init_params(specs, seed)
        assert np.allclose(reconstructed, seed.values)


def test_seed_values_to_init_params_roundtrip_for_transformed_specs():
    specs = _transformed_parameter_specs()
    seed = cluster_solver.ChainSeed(values=np.array([np.log(35.0), np.log(24.9)], dtype=float), source_label="transformed")

    reconstructed = _roundtrip_seed_from_init_params(specs, seed)

    assert np.allclose(reconstructed, seed.values)


def test_run_inference_uses_init_params_and_falls_back_from_svi(monkeypatch, tmp_path):
    args = _args()
    args.nuts_init_strategy = "prior_center"
    args.output_dir = str(tmp_path)
    run_dir = tmp_path / "run"
    specs = _parameter_specs()
    best_fit = np.array([0.25, 0.15], dtype=float)

    class FakeFamily:
        n_images = 1

    class FakeEvaluator:
        timing_totals = {"nuts_runtime": 0.0, "validation_runtime": 0.0, "plot_runtime": 0.0}
        validation_family_ids = ["f1"]
        active_scaling_galaxies_by_potfile = []
        active_scaling_component_indices = []
        inactive_scaling_component_indices = []
        requested_active_scaling_by_potfile = {}
        actual_active_scaling_by_potfile = {}
        total_scaling_by_potfile = {}
        scaling_rank_df = cluster_solver.pd.DataFrame()

        def evaluate(self, best_fit, likelihood_mode="source"):
            return cluster_solver.EvaluationResult(loglike=-1.0, family_predictions={"f1": {}}, used_exact_validation=False)

    state = type(
        "State",
        (),
        {
            "run_name": "demo",
            "par_path": "fake.par",
            "fit_mode": "small-only",
            "parameter_specs": specs,
            "family_data": [FakeFamily()],
            "bin_data": [object()],
            "packed_lens_spec": type("PackedLensSpec", (), {"component_family": np.array([], dtype=int)})(),
        },
    )()
    evaluator = FakeEvaluator()
    captured = {}

    def fake_posterior_model(specs, evaluator):
        return lambda: None

    class FakeNUTS:
        def __init__(self, model, target_accept_prob, max_tree_depth):
            captured["nuts_ctor"] = {
                "target_accept_prob": target_accept_prob,
                "max_tree_depth": max_tree_depth,
            }

    class FakeMCMC:
        def __init__(self, nuts, num_warmup, num_samples, num_chains, chain_method, progress_bar, progress_rate):
            captured["mcmc_ctor"] = {
                "num_warmup": num_warmup,
                "num_samples": num_samples,
                "num_chains": num_chains,
                "chain_method": chain_method,
            }

        def run(self, rng_key, extra_fields=(), init_params=None):
            captured["run_extra_fields"] = extra_fields
            captured["init_params"] = init_params

        def get_samples(self, group_by_chain=True):
            return {
                "u": np.array([[0.1, 0.2, 0.3, 0.4]] * args.chains),
                "n": np.array([[0.0, 0.1, 0.2, 0.3]] * args.chains),
            }

        def get_extra_fields(self, group_by_chain=True):
            shape = (args.chains, args.samples)
            return {
                "accept_prob": np.full(shape, 0.95),
                "diverging": np.zeros(shape, dtype=bool),
                "num_steps": np.full(shape, 3.0),
                "potential_energy": np.full(shape, 1.0),
            }

    monkeypatch.setattr(
        cluster_solver,
        "_run_continuation_map",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("prior_center should skip MAP")),
    )
    monkeypatch.setattr(cluster_solver, "_prepare_direct_evaluator", lambda *a, **k: (evaluator, best_fit))
    monkeypatch.setattr(cluster_solver, "_posterior_model", fake_posterior_model)
    monkeypatch.setattr(cluster_solver, "NUTS", FakeNUTS)
    monkeypatch.setattr(cluster_solver, "MCMC", FakeMCMC)
    monkeypatch.setattr(cluster_solver, "_save_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(cluster_solver, "_generate_plots_and_tables", lambda *a, **k: None)

    cluster_solver._run_inference(args, state, run_dir)

    assert "init_params" in captured
    assert tuple(np.asarray(captured["init_params"]["u"]).shape) == (args.chains,)
    assert len(np.unique(np.asarray(captured["init_params"]["u"]))) > 1
    assert captured["run_extra_fields"] == ("accept_prob", "diverging", "num_steps", "potential_energy")


def test_parse_args_accepts_prior_center(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cluster_solver.py", "--par-path", "fake.par", "--nuts-init-strategy", "prior_center"])
    args = cluster_solver._parse_args()

    assert args.nuts_init_strategy == "prior_center"


def test_build_nuts_initialization_supports_prior_center():
    args = _args()
    args.nuts_init_strategy = "prior_center"
    init = cluster_solver._build_nuts_initialization(args, _parameter_specs(), ranked_results=[], sample_model=lambda: None)

    assert init.diagnostics["strategy_used"] == "prior_center"
    assert init.diagnostics["svi_used"] is False
    assert np.allclose(init.reference_theta, cluster_solver._default_theta(_parameter_specs()))
    assert len(init.chain_seeds) == args.chains
    assert all(seed.source_label.startswith("prior_center_chain_") for seed in init.chain_seeds)
    assert init.diagnostics["distinct_chain_seeds"] >= 1
    for seed in init.chain_seeds:
        reconstructed = _roundtrip_seed_from_init_params(_parameter_specs(), seed)
        assert np.allclose(reconstructed, seed.values)


def test_build_nuts_initialization_ranked_map_roundtrip():
    args = _args()
    ranked = [
        cluster_solver.MAPRunResult(theta=np.array([0.5, 0.2]), logprob=4.0, iterations=3, label="best"),
        cluster_solver.MAPRunResult(theta=np.array([-0.1, 0.0]), logprob=3.0, iterations=3, label="next"),
    ]

    init = cluster_solver._build_nuts_initialization(args, _parameter_specs(), ranked_results=ranked, sample_model=lambda: None)

    assert init.diagnostics["strategy_used"] == "ranked_map"
    assert len(init.chain_seeds) == args.chains
    for seed in init.chain_seeds:
        reconstructed = _roundtrip_seed_from_init_params(_parameter_specs(), seed)
        assert np.allclose(reconstructed, seed.values)


def test_nuts_init_params_smoke_test_for_bounded_model():
    specs = _parameter_specs()
    chain_seeds = [cluster_solver.ChainSeed(values=np.array([0.3, 0.1], dtype=float), source_label="seed")]
    init_params = cluster_solver._seed_values_to_init_params(
        specs,
        chain_seeds,
        model_for_init=cluster_solver._sample_site_model(specs),
    )

    def toy_model():
        u = numpyro.sample("u", dist.Uniform(-2.0, 3.0))
        n = numpyro.sample("n", dist.Normal(0.2, 0.3))
        numpyro.factor("toy_loglike", -0.5 * (u**2 + n**2))

    mcmc = MCMC(NUTS(toy_model, target_accept_prob=0.85, max_tree_depth=4), num_warmup=5, num_samples=5, num_chains=1, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(0),
        init_params={name: value[:1] for name, value in init_params.items()},
        extra_fields=("accept_prob", "potential_energy"),
    )
    extra = mcmc.get_extra_fields(group_by_chain=True)

    assert np.isfinite(np.asarray(extra["potential_energy"], dtype=float)).all()
    assert np.isfinite(np.asarray(extra["accept_prob"], dtype=float)).all()


def test_m0416_large_prior_center_init_roundtrip_and_seed_validity():
    args = _args()
    args.par_path = "playground/M0416_Bergamini22/Bergamini22_MACS0416.par"
    args.fit_mode = "large-only"
    args.run_name = "large"
    args.profile_variant = cluster_solver.PROFILE_VARIANT_ORIGINAL
    args.z_bin_tol = 0.25
    args.pos_sigma_arcsec = None
    args.nuts_init_strategy = "prior_center"
    args.validate_top_k_families = 0
    args.validation_approx = "adaptive"
    args.match_tolerance_arcsec = 0.3
    args.active_scaling_galaxies = [64]

    state = cluster_solver._build_state_from_inputs(args)
    evaluator, midpoint = cluster_solver._prepare_direct_evaluator(args, state)
    init = cluster_solver._build_nuts_initialization(args, state.parameter_specs, ranked_results=[], sample_model=lambda: None)

    assert np.isfinite(evaluator.source_loglike(midpoint))
    assert np.allclose(init.reference_theta, midpoint)
    for seed in init.chain_seeds:
        assert np.isfinite(evaluator.source_loglike(seed.values))
        reconstructed = _roundtrip_seed_from_init_params(state.parameter_specs, seed)
        assert np.allclose(reconstructed, seed.values)


def test_m0416_large_midpoint_gradient_is_finite():
    args = _args()
    args.par_path = "playground/M0416_Bergamini22/Bergamini22_MACS0416.par"
    args.fit_mode = "large-only"
    args.run_name = "large"
    args.profile_variant = cluster_solver.PROFILE_VARIANT_ORIGINAL
    args.z_bin_tol = 0.25
    args.pos_sigma_arcsec = None
    args.validate_top_k_families = 0
    args.validation_approx = "adaptive"
    args.match_tolerance_arcsec = 0.3
    args.active_scaling_galaxies = [64]

    state = cluster_solver._build_state_from_inputs(args)
    evaluator, midpoint = cluster_solver._prepare_direct_evaluator(args, state)
    midpoint_jax = cluster_solver.jnp.asarray(midpoint, dtype=cluster_solver.jnp.float64)
    grad = jax.grad(evaluator._source_loglike_fn)(midpoint_jax)

    assert np.isfinite(float(evaluator._source_loglike_fn(midpoint_jax)))
    assert np.isfinite(np.asarray(grad, dtype=float)).all()


def test_m0416_large_differentiable_packed_state_builder_has_finite_gradient():
    args = _args()
    args.par_path = "playground/M0416_Bergamini22/Bergamini22_MACS0416.par"
    args.fit_mode = "large-only"
    args.run_name = "large"
    args.profile_variant = cluster_solver.PROFILE_VARIANT_ORIGINAL
    args.z_bin_tol = 0.25
    args.pos_sigma_arcsec = None
    args.validate_top_k_families = 0
    args.validation_approx = "adaptive"
    args.match_tolerance_arcsec = 0.3
    args.active_scaling_galaxies = [64]

    state = cluster_solver._build_state_from_inputs(args)
    evaluator, midpoint = cluster_solver._prepare_direct_evaluator(args, state)
    midpoint_jax = cluster_solver.jnp.asarray(midpoint, dtype=cluster_solver.jnp.float64)
    sw_idx = next(i for i, component in enumerate(state.base_components) if component["id"] == "2 # SW dark matter halo")

    def sigma0_component(theta):
        packed_state = evaluator._build_packed_lens_state(theta, state.bin_data[0].effective_z_source)
        return packed_state["sigma0"][sw_idx]

    grad = jax.grad(sigma0_component)(midpoint_jax)

    assert np.isfinite(float(sigma0_component(midpoint_jax)))
    assert np.isfinite(np.asarray(grad, dtype=float)).all()


def test_m0416_large_short_nuts_smoke_test_avoids_all_invalid_rejections():
    args = _args()
    args.par_path = "playground/M0416_Bergamini22/Bergamini22_MACS0416.par"
    args.fit_mode = "large-only"
    args.run_name = "large"
    args.profile_variant = cluster_solver.PROFILE_VARIANT_ORIGINAL
    args.z_bin_tol = 0.25
    args.pos_sigma_arcsec = None
    args.nuts_init_strategy = "prior_center"
    args.validate_top_k_families = 0
    args.validation_approx = "adaptive"
    args.match_tolerance_arcsec = 0.3
    args.active_scaling_galaxies = [64]
    args.chains = 1
    args.warmup = 5
    args.samples = 5
    args.max_tree_depth = 6
    args.target_accept = 0.85

    state = cluster_solver._build_state_from_inputs(args)
    evaluator, _midpoint = cluster_solver._prepare_direct_evaluator(args, state)
    sample_model = cluster_solver._posterior_model(state.parameter_specs, evaluator)
    nuts_init = cluster_solver._build_nuts_initialization(args, state.parameter_specs, ranked_results=[], sample_model=sample_model)
    posterior = cluster_solver._run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init, map_history=[])

    max_all_invalid = args.chains * (args.warmup + args.samples) * len(state.bin_data)
    assert int(evaluator.invalid_state_rejection_count) < max_all_invalid
    assert int(evaluator.invalid_state_rejection_count) == 0
    assert np.isfinite(np.asarray(posterior.log_prob, dtype=float)).all()
    assert np.isfinite(np.asarray(posterior.samples, dtype=float)).all()


def _m0416_small_surrogate_args():
    args = _args()
    args.par_path = "playground/M0416_Bergamini22/Bergamini22_MACS0416.par"
    args.stage1_run_dir = "plots/m0416_original/large"
    args.fit_mode = "small-only"
    args.run_name = "exp_ranked"
    args.profile_variant = cluster_solver.PROFILE_VARIANT_ORIGINAL
    args.z_bin_tol = 0.25
    args.pos_sigma_arcsec = None
    args.nuts_init_strategy = "prior_center"
    args.validate_top_k_families = 0
    args.validation_approx = "adaptive"
    args.match_tolerance_arcsec = 0.3
    args.active_scaling_galaxies = [32]
    args.sampling_engine = "refreshing_surrogate"
    args.refresh_param_drift_frac = 0.08
    args.target_accept = 0.9
    args.max_tree_depth = 8
    return args


def test_m0416_small_surrogate_midpoint_and_gradient_are_finite():
    args = _m0416_small_surrogate_args()
    state = cluster_solver._build_state_from_inputs(args)
    evaluator, midpoint = cluster_solver._prepare_direct_evaluator(args, state)
    midpoint_jax = cluster_solver.jnp.asarray(midpoint, dtype=cluster_solver.jnp.float64)
    grad = jax.grad(evaluator._source_loglike_fn)(midpoint_jax)

    assert evaluator.surrogate_enabled
    assert np.isfinite(float(evaluator._source_loglike_fn(midpoint_jax)))
    assert np.isfinite(np.asarray(grad, dtype=float)).all()


def test_m0416_small_surrogate_cache_is_finite_and_matches_full_beta():
    surrogate_args = _m0416_small_surrogate_args()
    surrogate_state = cluster_solver._build_state_from_inputs(surrogate_args)
    surrogate_evaluator, midpoint = cluster_solver._prepare_direct_evaluator(surrogate_args, surrogate_state)
    assert surrogate_evaluator.surrogate_enabled
    assert surrogate_evaluator.surrogate_cache_by_z

    for cache in surrogate_evaluator.surrogate_cache_by_z.values():
        assert np.isfinite(np.asarray(cache.inactive_alpha_x, dtype=float)).all()
        assert np.isfinite(np.asarray(cache.inactive_alpha_y, dtype=float)).all()
        assert np.isfinite(np.asarray(cache.inactive_alpha_dx_dparams, dtype=float)).all()
        assert np.isfinite(np.asarray(cache.inactive_alpha_dy_dparams, dtype=float)).all()

    full_args = _m0416_small_surrogate_args()
    full_args.sampling_engine = "full"
    full_state = cluster_solver._build_state_from_inputs(full_args)
    full_evaluator, _ = cluster_solver._prepare_direct_evaluator(full_args, full_state)

    params = cluster_solver.jnp.asarray(midpoint, dtype=cluster_solver.jnp.float64)
    surrogate_bin = surrogate_state.bin_data[0]
    full_bin = full_state.bin_data[0]
    surrogate_beta_x, surrogate_beta_y, invalid = surrogate_evaluator._surrogate_beta(params, surrogate_bin)
    full_packed_state = full_evaluator._build_packed_lens_state(params, full_bin.effective_z_source)
    full_beta_x, full_beta_y = full_evaluator._ray_shooting_for_components(
        full_bin.effective_z_source,
        cluster_solver.jnp.asarray(full_bin.x_obs, dtype=cluster_solver.jnp.float64),
        cluster_solver.jnp.asarray(full_bin.y_obs, dtype=cluster_solver.jnp.float64),
        full_packed_state,
    )

    assert not bool(np.asarray(invalid))
    assert np.isfinite(np.asarray(surrogate_beta_x, dtype=float)).all()
    assert np.isfinite(np.asarray(surrogate_beta_y, dtype=float)).all()
    assert np.allclose(np.asarray(surrogate_beta_x), np.asarray(full_beta_x), atol=1.0e-8, rtol=1.0e-8)
    assert np.allclose(np.asarray(surrogate_beta_y), np.asarray(full_beta_y), atol=1.0e-8, rtol=1.0e-8)


def test_m0416_small_surrogate_short_nuts_smoke_test_avoids_invalid_plateau():
    args = _m0416_small_surrogate_args()
    args.chains = 1
    args.warmup = 5
    args.samples = 5
    args.max_tree_depth = 6
    args.target_accept = 0.85

    state = cluster_solver._build_state_from_inputs(args)
    evaluator, _midpoint = cluster_solver._prepare_direct_evaluator(args, state)
    sample_model = cluster_solver._posterior_model(state.parameter_specs, evaluator)
    nuts_init = cluster_solver._build_nuts_initialization(args, state.parameter_specs, ranked_results=[], sample_model=sample_model)
    posterior = cluster_solver._run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init, map_history=[])

    assert evaluator.surrogate_cache_by_z
    assert np.isfinite(np.asarray(posterior.log_prob, dtype=float)).all()
    assert np.isfinite(np.asarray(posterior.samples, dtype=float)).all()


def test_summary_table_uses_sample_weights():
    specs = _parameter_specs()
    samples = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.1],
            [2.0, 0.5],
        ],
        dtype=float,
    )
    best_fit = np.array([2.0, 0.5], dtype=float)
    weights = np.array([0.05, 0.15, 0.8], dtype=float)

    summary_df = cluster_solver._summary_table(specs, samples, best_fit, sample_weights=weights)

    row = summary_df.iloc[0]
    expected_mean = np.average(samples[:, 0], weights=weights)
    assert np.isclose(float(row["mean"]), expected_mean)
    assert float(row["median"]) > 0.5


def test_run_summary_prefers_cli_run_name_over_loaded_state_name():
    args = _args()
    args.run_name = "refine_run"
    state = type(
        "State",
        (),
        {
            "run_name": "loaded_source_run",
            "par_path": "fake.par",
            "fit_mode": "small-only",
            "profile_variant": cluster_solver.PROFILE_VARIANT_ORIGINAL,
            "compact_skip_factor": 1.3,
            "parameter_specs": _parameter_specs(),
            "family_data": [],
            "packed_lens_spec": type("PackedLensSpec", (), {"component_family": np.array([], dtype=int)})(),
            "potfiles": [],
        },
    )()
    results = cluster_solver.PosteriorResults(
        samples=np.array([[0.1, 0.2]], dtype=float),
        log_prob=np.array([1.0], dtype=float),
        accept_prob=np.array([0.9], dtype=float),
        diverging=np.array([False]),
        num_steps=np.array([4.0], dtype=float),
        map_history=[],
        warmup_steps=1,
        sample_steps=1,
        num_chains=1,
        init_diagnostics={},
        sampler="numpyro_nuts",
    )
    evaluator = type(
        "Evaluator",
        (),
        {
            "active_scaling_galaxies_by_potfile": [],
            "active_scaling_component_indices": [],
            "inactive_scaling_component_indices": [],
            "requested_active_scaling_by_potfile": {},
            "actual_active_scaling_by_potfile": {},
            "total_scaling_by_potfile": {},
            "invalid_state_rejection_count": 0,
            "invalid_state_reason_counts": {name: 0 for name in cluster_solver.INVALID_STATE_REASON_NAMES},
            "surrogate_enabled": False,
            "approximate_eval_count": 0,
            "full_refresh_count": 0,
            "validation_fallback_count": 0,
            "timing_totals": {},
            "eval_wall_times": [],
        },
    )()

    summary = cluster_solver._run_summary(args, state, 1.0, results, -1.0, evaluator)

    assert summary["run_name"] == "refine_run"
    assert summary["compact_skip_factor"] == 1.3


def test_build_scaling_parameter_specs_uses_validity_preserving_latent_parameters():
    potfiles = [
        {
            "id": "potfile1",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": [object()] * 2,
            "corekpc": 0.1,
            "sigma": [3, 55.0, 20.0],
            "cutkpc": [3, 25.0, 12.0],
            "vdslope": [0, 4.0, 0.0],
            "slope": [0, 4.0, 0.0],
        }
    ]

    specs, param_index_by_potfile, lens_models = cluster_solver._build_scaling_parameter_specs(potfiles)

    assert lens_models == [cluster_solver.ORIGINAL_DPIE_PROFILE_NAME, cluster_solver.ORIGINAL_DPIE_PROFILE_NAME]
    sigma_spec = next(spec for spec in specs if spec.field == "sigma")
    cut_spec = next(spec for spec in specs if spec.field == "cutkpc")
    assert sigma_spec.transform_kind == "log_positive"
    assert cut_spec.transform_kind == "log_offset_positive"
    assert np.isclose(sigma_spec.physical_mean, 55.0)
    assert np.isclose(cut_spec.physical_mean, 25.0)
    assert cut_spec.transform_offset == 0.1
    assert sigma_spec.mean < np.log(55.0)
    assert cut_spec.lower < cut_spec.upper
    assert param_index_by_potfile[0]["sigma"] == 0
    assert param_index_by_potfile[0]["cutkpc"] == 1


def test_build_parameter_specs_supports_compact_profile_variant_for_large_dpie():
    potentials_with_priors = [
        {"id": "halo0", "profil": cluster_solver.DP_IE_PROFILE, "priors": {}},
        {"id": "shear0", "profil": cluster_solver.SHEAR_PROFILE, "priors": {}},
    ]

    _specs, _assignments, lens_models = cluster_solver._build_parameter_specs(
        potentials_with_priors,
        profile_variant=cluster_solver.PROFILE_VARIANT_COMPACT,
    )

    assert lens_models == [cluster_solver.COMPACT_DPIE_PROFILE_NAME, "SHEAR"]


def test_build_scaling_parameter_specs_supports_compact_profile_variant():
    potfiles = [
        {
            "id": "potfile1",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": [object()] * 2,
            "corekpc": 0.1,
            "sigma": [3, 55.0, 20.0],
            "cutkpc": [3, 25.0, 12.0],
            "vdslope": [0, 4.0, 0.0],
            "slope": [0, 4.0, 0.0],
        }
    ]

    _specs, _param_index_by_potfile, lens_models = cluster_solver._build_scaling_parameter_specs(
        potfiles,
        profile_variant=cluster_solver.PROFILE_VARIANT_COMPACT,
    )

    assert lens_models == [cluster_solver.COMPACT_DPIE_PROFILE_NAME, cluster_solver.COMPACT_DPIE_PROFILE_NAME]


def test_build_scaling_parameter_specs_converts_arcsec_cut_and_core_to_kpc():
    potfiles = [
        {
            "id": "potfile1",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": [object()] * 2,
            "core": [1, 1.0, 2.0, 0.1],
            "cut": [3, 25.0, 12.0],
            "sigma": [3, 55.0, 20.0],
            "vdslope": [0, 4.0, 0.0],
            "slope": [0, 4.0, 0.0],
            "corekpc": None,
            "core_arcsec": 1.0,
            "cutkpc_nominal": None,
            "cut_arcsec_nominal": 25.0,
        }
    ]

    specs, param_index_by_potfile, lens_models = cluster_solver._build_scaling_parameter_specs(
        potfiles,
        kpc_per_arcsec=2.0,
    )

    assert lens_models == [cluster_solver.ORIGINAL_DPIE_PROFILE_NAME, cluster_solver.ORIGINAL_DPIE_PROFILE_NAME]
    core_spec = next(spec for spec in specs if spec.field == "corekpc")
    cut_spec = next(spec for spec in specs if spec.field == "cutkpc")
    assert np.isclose(core_spec.physical_lower, 2.0)
    assert np.isclose(cut_spec.physical_mean, 50.0)
    assert np.isclose(cut_spec.transform_offset, 2.0)
    assert param_index_by_potfile[0]["corekpc"] >= 0
    assert param_index_by_potfile[0]["cutkpc"] >= 0


def test_build_scaling_components_converts_arcsec_cut_and_core_to_kpc():
    potfiles = [
        {
            "id": "potfile1",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": pd.DataFrame(
                {
                    "id": ["g1"],
                    "ra": [64.0],
                    "dec": [-24.0],
                    "catalog_a": [1.0],
                    "catalog_b": [1.0],
                    "catalog_theta": [0.0],
                    "catalog_mag": [20.0],
                }
            ),
            "mag0": 20.0,
            "sigma_nominal": 100.0,
            "zlens": 0.4,
            "corekpc": None,
            "core_arcsec": 1.5,
            "cutkpc_nominal": None,
            "cut_arcsec_nominal": 30.0,
            "vdslope_nominal": 4.0,
            "slope_nominal": 4.0,
        }
    ]

    components, _assignments, scaling_component_assignments, _records = cluster_solver._build_scaling_components(
        potfiles,
        reference=(3, 64.0, -24.0),
        scaling_param_indices=[{}],
        start_component_index=0,
        kpc_per_arcsec=2.0,
    )

    assert np.isclose(components[0]["core_radius_kpc"], 3.0)
    assert np.isclose(components[0]["cut_radius_kpc"], 60.0)
    assert np.isclose(scaling_component_assignments[0]["core_ref_base"], 3.0)
    assert np.isclose(scaling_component_assignments[0]["cut_ref_base"], 60.0)


def test_parse_args_defaults_profile_variant_to_original(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cluster_solver.py", "--par-path", "fake.par"])
    args = cluster_solver._parse_args()

    assert args.profile_variant == cluster_solver.PROFILE_VARIANT_ORIGINAL
    assert args.compact_skip_factor == 1.0


def test_parse_args_accepts_compact_skip_factor(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver.py",
            "--par-path",
            "fake.par",
            "--compact-skip-factor",
            "1.2",
        ],
    )
    args = cluster_solver._parse_args()

    assert np.isclose(args.compact_skip_factor, 1.2)


def test_extract_samples_converts_transformed_scaling_parameters_to_physical_units():
    specs = [
        cluster_solver.ParameterSpec(
            name="potfile1.sigma",
            sample_name="potfile1_sigma",
            potential_id="potfile1",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma",
            prior_kind="normal",
            lower=np.log(1.0),
            upper=np.log(200.0),
            step=0.1,
            mean=np.log(55.0),
            std=0.2,
            component_family="scaling",
            transform_kind="log_positive",
            physical_lower=1.0,
            physical_upper=200.0,
            physical_mean=55.0,
            physical_std=20.0,
        ),
        cluster_solver.ParameterSpec(
            name="potfile1.cutkpc",
            sample_name="potfile1_cutkpc",
            potential_id="potfile1",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="cutkpc",
            prior_kind="normal",
            lower=np.log(0.1),
            upper=np.log(199.9),
            step=0.1,
            mean=np.log(24.9),
            std=0.2,
            component_family="scaling",
            transform_kind="log_offset_positive",
            physical_lower=0.2,
            physical_upper=200.0,
            physical_mean=25.0,
            physical_std=12.0,
            transform_offset=0.1,
        ),
    ]
    samples_dict = {
        "potfile1_sigma": np.array([[np.log(30.0), np.log(40.0)]]),
        "potfile1_cutkpc": np.array([[np.log(19.9), np.log(29.9)]]),
    }

    extracted = cluster_solver._extract_samples(samples_dict, specs, thin=1)
    grouped = cluster_solver._extract_grouped_samples(samples_dict, specs, thin=1)
    extracted_physical = cluster_solver._convert_sample_matrix_to_physical(extracted, specs)
    grouped_physical = cluster_solver._convert_sample_matrix_to_physical(grouped, specs)
    summary_df = cluster_solver._summary_table(specs, extracted_physical, extracted_physical[0])

    assert np.allclose(extracted_physical[:, 0], np.array([30.0, 40.0]))
    assert np.allclose(extracted_physical[:, 1], np.array([20.0, 30.0]))
    assert np.allclose(grouped_physical[0, :, 0], np.array([30.0, 40.0]))
    assert np.allclose(grouped_physical[0, :, 1], np.array([20.0, 30.0]))
    assert float(summary_df.loc[summary_df["parameter"] == "sigma", "upper"].iloc[0]) == 200.0
    assert float(summary_df.loc[summary_df["parameter"] == "cutkpc", "lower"].iloc[0]) == 0.2


def test_transformed_scaling_parameters_avoid_invalid_sigma_and_rs_flags():
    sigma_spec = cluster_solver.ParameterSpec(
        name="potfile1.sigma",
        sample_name="potfile1_sigma",
        potential_id="potfile1",
        profile_type=cluster_solver.DP_IE_PROFILE,
        field="sigma",
        prior_kind="normal",
        lower=np.log(1.0e-6),
        upper=np.log(200.0),
        step=0.1,
        mean=np.log(30.0),
        std=0.3,
        component_family="scaling",
        transform_kind="log_positive",
        physical_lower=1.0e-6,
        physical_upper=200.0,
    )
    cut_spec = cluster_solver.ParameterSpec(
        name="potfile1.cutkpc",
        sample_name="potfile1_cutkpc",
        potential_id="potfile1",
        profile_type=cluster_solver.DP_IE_PROFILE,
        field="cutkpc",
        prior_kind="normal",
        lower=np.log(1.0e-6),
        upper=np.log(200.0),
        step=0.1,
        mean=np.log(20.0),
        std=0.3,
        component_family="scaling",
        transform_kind="log_offset_positive",
        physical_lower=0.100001,
        physical_upper=200.1,
        transform_offset=0.1,
    )
    packed = cluster_solver.PackedLensSpec(
        profile_type=np.array([cluster_solver.DP_IE_PROFILE], dtype=np.int32),
        component_family=np.array([1], dtype=np.int32),
        x_center_base=np.array([0.0]),
        y_center_base=np.array([0.0]),
        ellipticite_base=np.array([0.0]),
        angle_pos_base=np.array([0.0]),
        core_radius_kpc_base=np.array([0.1]),
        cut_radius_kpc_base=np.array([20.0]),
        v_disp_base=np.array([100.0]),
        gamma_base=np.array([0.0]),
        x_center_param_index=np.array([-1], dtype=np.int32),
        y_center_param_index=np.array([-1], dtype=np.int32),
        ellipticite_param_index=np.array([-1], dtype=np.int32),
        angle_pos_param_index=np.array([-1], dtype=np.int32),
        core_radius_param_index=np.array([-1], dtype=np.int32),
        cut_radius_param_index=np.array([-1], dtype=np.int32),
        v_disp_param_index=np.array([-1], dtype=np.int32),
        gamma_param_index=np.array([-1], dtype=np.int32),
        luminosity_ratio=np.array([1.0]),
        sigma_ref_base=np.array([30.0]),
        cut_ref_base=np.array([20.0]),
        core_ref_base=np.array([0.1]),
        vdslope_base=np.array([4.0]),
        slope_base=np.array([4.0]),
        sigma_ref_param_index=np.array([0], dtype=np.int32),
        cut_ref_param_index=np.array([1], dtype=np.int32),
        core_ref_param_index=np.array([-1], dtype=np.int32),
        vdslope_param_index=np.array([-1], dtype=np.int32),
        slope_param_index=np.array([-1], dtype=np.int32),
    )
    state = type("State", (), {"packed_lens_spec": packed, "parameter_specs": [sigma_spec, cut_spec]})()
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.state = state
    evaluator.kpc_per_arcsec = 1.0
    evaluator.dpie_sigma0_factors = {1.0: 1.0}
    evaluator.timing_totals = {"packed_parameter_update": 0.0}
    params = cluster_solver.jnp.asarray([np.log(1.0e-9), np.log(1.0e-9)], dtype=cluster_solver.jnp.float64)

    packed_state, details = cluster_solver.ClusterJAXEvaluator._build_packed_lens_state_details(evaluator, params, 1.0)
    validity = cluster_solver.ClusterJAXEvaluator._packed_lens_validity(evaluator, details)

    assert bool(validity["is_valid"])
    reason_flags = np.asarray(validity["reason_flags"], dtype=bool)
    assert not bool(reason_flags[cluster_solver.INVALID_STATE_REASON_NAMES.index("rs_not_greater_than_ra")])
    assert not bool(reason_flags[cluster_solver.INVALID_STATE_REASON_NAMES.index("nonpositive_vdisp")])
    assert float(np.asarray(packed_state["Ra"])[0]) > 0.0
    assert float(np.asarray(packed_state["Rs"])[0]) > float(np.asarray(packed_state["Ra"])[0])


def test_run_inference_dispatches_blackjax_smc(monkeypatch, tmp_path):
    args = _args()
    args.sampler = "blackjax_smc"
    args.smc_seed_mode = "prior"
    args.output_dir = str(tmp_path)
    run_dir = tmp_path / "run"
    specs = _parameter_specs()
    smc_best_fit = np.array([0.9, 0.4], dtype=float)

    class FakeFamily:
        n_images = 1

    class FakeEvaluator:
        timing_totals = {"smc_runtime": 0.0, "validation_runtime": 0.0, "plot_runtime": 0.0}
        validation_family_ids = ["f1"]
        active_scaling_galaxies_by_potfile = []
        active_scaling_component_indices = []
        inactive_scaling_component_indices = []
        requested_active_scaling_by_potfile = {}
        actual_active_scaling_by_potfile = {}
        total_scaling_by_potfile = {}
        scaling_rank_df = cluster_solver.pd.DataFrame()
        surrogate_enabled = False
        invalid_state_rejection_count = 0
        invalid_state_reason_counts = {name: 0 for name in cluster_solver.INVALID_STATE_REASON_NAMES}
        eval_wall_times = []

        def evaluate(self, best_fit, likelihood_mode="source"):
            assert np.allclose(best_fit, smc_best_fit)
            return cluster_solver.EvaluationResult(loglike=-0.5, family_predictions={"f1": {}}, used_exact_validation=False)

    state = type(
        "State",
        (),
        {
            "run_name": "demo",
            "par_path": "fake.par",
            "fit_mode": "small-only",
            "parameter_specs": specs,
            "family_data": [FakeFamily()],
            "bin_data": [object()],
            "packed_lens_spec": type("PackedLensSpec", (), {"component_family": np.array([], dtype=int)})(),
        },
    )()
    evaluator = FakeEvaluator()
    captured = {}

    def fake_posterior_model(specs, evaluator):
        return lambda: None

    def fake_run_smc_sampler(args, state, evaluator, ranked_results, map_history):
        captured["ranked_results"] = ranked_results
        captured["map_history"] = map_history
        assert ranked_results == []
        assert map_history == []
        captured["ranked_results"] = ranked_results
        posterior = cluster_solver.PosteriorResults(
            samples=np.array([[0.8, 0.35], [0.9, 0.4]], dtype=float),
            log_prob=np.array([1.0, 2.0], dtype=float),
            accept_prob=np.array([0.7, 0.8], dtype=float),
            diverging=np.zeros(2, dtype=bool),
            num_steps=np.array([20.0, 18.0], dtype=float),
            map_history=map_history,
            warmup_steps=0,
            sample_steps=2,
            num_chains=1,
            init_diagnostics={"strategy_used": "blackjax_smc"},
            sampler="blackjax_smc",
            sample_weights=np.array([0.25, 0.75], dtype=float),
            temperature_schedule=np.array([0.0, 0.5, 1.0], dtype=float),
            ess_history=np.array([32.0, 20.0, 15.0], dtype=float),
            move_acceptance_history=np.array([0.8, 0.7], dtype=float),
        )
        return smc_best_fit, posterior

    def fake_prepare_direct_evaluator(*_args, **_kwargs):
        captured["direct_evaluator"] = True
        return evaluator, np.array([0.0, 0.0], dtype=float)

    monkeypatch.setattr(cluster_solver, "_run_continuation_map", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SMC should skip continuation MAP")))
    monkeypatch.setattr(cluster_solver, "_prepare_direct_evaluator", fake_prepare_direct_evaluator)
    monkeypatch.setattr(cluster_solver, "_posterior_model", fake_posterior_model)
    monkeypatch.setattr(cluster_solver, "_run_smc_sampler", fake_run_smc_sampler)
    monkeypatch.setattr(cluster_solver, "_save_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(cluster_solver, "_generate_plots_and_tables", lambda *a, **k: captured.setdefault("generated", True))

    cluster_solver._run_inference(args, state, run_dir)

    assert captured["generated"] is True
    assert captured["direct_evaluator"] is True
    assert captured["ranked_results"] == []
    assert captured["map_history"] == []


def test_blackjax_smc_ranked_map_requires_map_candidates():
    args = _args()
    args.sampler = "blackjax_smc"
    args.smc_seed_mode = "ranked_map"

    try:
        cluster_solver._smc_initial_particles(args, _parameter_specs(), [])
    except ValueError as exc:
        assert "Use '--smc-seed-mode prior'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ranked_map SMC seeding without MAP candidates to fail.")


def test_blackjax_smc_refine_with_nuts_uses_smc_particles(monkeypatch, tmp_path):
    args = _args()
    args.sampler = "blackjax_smc"
    args.smc_seed_mode = "prior"
    args.refine_with_nuts = True
    args.output_dir = str(tmp_path)
    run_dir = tmp_path / "run"
    specs = _parameter_specs()
    smc_best_fit = np.array([0.9, 0.4], dtype=float)

    class FakeFamily:
        n_images = 1

    class FakeEvaluator:
        timing_totals = {"smc_runtime": 0.0, "nuts_runtime": 0.0, "validation_runtime": 0.0, "plot_runtime": 0.0}
        validation_family_ids = ["f1"]
        active_scaling_galaxies_by_potfile = []
        active_scaling_component_indices = []
        inactive_scaling_component_indices = []
        requested_active_scaling_by_potfile = {}
        actual_active_scaling_by_potfile = {}
        total_scaling_by_potfile = {}
        scaling_rank_df = cluster_solver.pd.DataFrame()
        surrogate_enabled = False
        invalid_state_rejection_count = 0
        invalid_state_reason_counts = {name: 0 for name in cluster_solver.INVALID_STATE_REASON_NAMES}
        eval_wall_times = []

        def evaluate(self, best_fit, likelihood_mode="source"):
            return cluster_solver.EvaluationResult(loglike=-0.5, family_predictions={"f1": {}}, used_exact_validation=False)

    state = type(
        "State",
        (),
        {
            "run_name": "demo",
            "par_path": "fake.par",
            "fit_mode": "small-only",
            "parameter_specs": specs,
            "family_data": [FakeFamily()],
            "bin_data": [object()],
            "packed_lens_spec": type("PackedLensSpec", (), {"component_family": np.array([], dtype=int)})(),
        },
    )()
    evaluator = FakeEvaluator()
    captured = {}

    def fake_run_smc_sampler(args, state, evaluator, ranked_results, map_history):
        posterior = cluster_solver.PosteriorResults(
            samples=np.array([[0.8, 0.35], [0.9, 0.4]], dtype=float),
            log_prob=np.array([1.0, 2.0], dtype=float),
            accept_prob=np.array([0.7, 0.8], dtype=float),
            diverging=np.zeros(2, dtype=bool),
            num_steps=np.array([20.0, 18.0], dtype=float),
            map_history=map_history,
            warmup_steps=0,
            sample_steps=2,
            num_chains=1,
            init_diagnostics={"strategy_used": "blackjax_smc"},
            sampler="blackjax_smc",
            sample_weights=np.array([0.25, 0.75], dtype=float),
            temperature_schedule=np.array([0.0, 0.5, 1.0], dtype=float),
            ess_history=np.array([32.0, 20.0, 15.0], dtype=float),
            move_acceptance_history=np.array([0.8, 0.7], dtype=float),
        )
        return smc_best_fit, posterior

    def fake_build_smc_refine_initialization(args, parameter_specs, particles, log_prob, sample_weights):
        captured["refine_particles"] = np.asarray(particles, dtype=float)
        captured["refine_log_prob"] = np.asarray(log_prob, dtype=float)
        captured["refine_weights"] = np.asarray(sample_weights, dtype=float)
        return cluster_solver.NUTSInitialization(
            init_params={"u": cluster_solver.jnp.asarray([0.8, 0.9]), "n": cluster_solver.jnp.asarray([0.35, 0.4])},
            chain_seeds=[],
            diagnostics={"strategy_used": "smc_refine_nuts", "distinct_chain_seeds": 2},
            reference_theta=np.array([0.9, 0.4], dtype=float),
        )

    def fake_run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init, map_history):
        captured["nuts_refine_called"] = True
        return cluster_solver.PosteriorResults(
            samples=np.array([[0.88, 0.39], [0.91, 0.41]], dtype=float),
            log_prob=np.array([2.5, 2.7], dtype=float),
            accept_prob=np.array([0.9, 0.92], dtype=float),
            diverging=np.zeros(2, dtype=bool),
            num_steps=np.array([5.0, 6.0], dtype=float),
            map_history=map_history,
            warmup_steps=5,
            sample_steps=4,
            num_chains=2,
            init_diagnostics={"strategy_used": "smc_refine_nuts", "distinct_chain_seeds": 2},
            sampler="numpyro_nuts",
        )

    def fake_prepare_direct_evaluator(*_args, **_kwargs):
        captured["direct_evaluator"] = True
        return evaluator, np.array([0.0, 0.0], dtype=float)

    monkeypatch.setattr(cluster_solver, "_run_continuation_map", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SMC should skip continuation MAP")))
    monkeypatch.setattr(cluster_solver, "_prepare_direct_evaluator", fake_prepare_direct_evaluator)
    monkeypatch.setattr(cluster_solver, "_posterior_model", lambda *a, **k: (lambda: None))
    monkeypatch.setattr(cluster_solver, "_run_smc_sampler", fake_run_smc_sampler)
    monkeypatch.setattr(cluster_solver, "_build_smc_refine_initialization", fake_build_smc_refine_initialization)
    monkeypatch.setattr(cluster_solver, "_run_numpyro_nuts_sampler", fake_run_numpyro_nuts_sampler)
    monkeypatch.setattr(cluster_solver, "_save_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(cluster_solver, "_generate_plots_and_tables", lambda *a, **k: None)

    cluster_solver._run_inference(args, state, run_dir)

    assert captured["direct_evaluator"] is True
    assert captured["nuts_refine_called"] is True
    assert captured["refine_particles"].shape == (2, 2)


def test_run_refine_from_saved_smc_uses_loaded_artifacts(monkeypatch, tmp_path):
    args = _args()
    args.refine_from_run_dir = str(tmp_path / "smc_run")
    args.output_dir = str(tmp_path / "out")
    args.run_name = "cpu_refine"
    specs = [
        cluster_solver.ParameterSpec(
            name="potfile1.sigma",
            sample_name="potfile1_sigma",
            potential_id="potfile1",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=np.log(30.0),
            std=0.3,
            component_family="scaling",
            transform_kind="log_positive",
            physical_lower=1.0,
            physical_upper=200.0,
            physical_mean=30.0,
            physical_std=10.0,
        ),
        cluster_solver.ParameterSpec(
            name="potfile1.cutkpc",
            sample_name="potfile1_cutkpc",
            potential_id="potfile1",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="cutkpc",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=np.log(19.9),
            std=0.3,
            component_family="scaling",
            transform_kind="log_offset_positive",
            physical_lower=0.1,
            physical_upper=200.0,
            physical_mean=20.0,
            physical_std=8.0,
            transform_offset=0.1,
        ),
    ]
    state = type(
        "State",
        (),
        {
            "run_name": "loaded",
            "par_path": "fake.par",
            "fit_mode": "small-only",
            "parameter_specs": specs,
            "family_data": [],
            "bin_data": [],
            "packed_lens_spec": type("PackedLensSpec", (), {"component_family": np.array([], dtype=int)})(),
            "potfiles": [],
            "lens_model_list": [],
            "cosmo_config": {},
            "parsed": {},
            "z_lens": 0.3,
            "sigma_arcsec": 0.5,
            "reference": (1, 0.0, 0.0),
            "base_components": [],
            "scaling_component_records": [],
        },
    )()
    arrays = {
        "samples": np.array([[30.0, 20.0], [40.0, 30.0]], dtype=float),
        "log_prob": np.array([1.0, 2.0], dtype=float),
        "sample_weights": np.array([0.25, 0.75], dtype=float),
    }
    saved_args = {"sampler": "blackjax_smc"}
    captured = {}

    class FakeEvaluator:
        def __init__(self, *args, **kwargs):
            self.validation_family_ids = set()
            self.timing_totals = {"nuts_runtime": 0.0, "validation_runtime": 0.0, "plot_runtime": 0.0}
            self.invalid_state_rejection_count = 0
            self.invalid_state_reason_counts = {name: 0 for name in cluster_solver.INVALID_STATE_REASON_NAMES}

        def evaluate(self, best_fit, likelihood_mode="source"):
            captured["validated_best_fit"] = np.asarray(best_fit, dtype=float)
            return cluster_solver.EvaluationResult(loglike=-0.5, family_predictions={}, used_exact_validation=False)

    def fake_build_smc_refine_initialization(args, parameter_specs, particles, log_prob, sample_weights):
        captured["latent_particles"] = np.asarray(particles, dtype=float)
        captured["refine_log_prob"] = np.asarray(log_prob, dtype=float)
        captured["refine_weights"] = np.asarray(sample_weights, dtype=float)
        return cluster_solver.NUTSInitialization(
            init_params={"potfile1_sigma": cluster_solver.jnp.asarray([np.log(30.0)]), "potfile1_cutkpc": cluster_solver.jnp.asarray([np.log(19.9)])},
            chain_seeds=[],
            diagnostics={"strategy_used": "smc_refine_nuts"},
            reference_theta=np.array([np.log(30.0), np.log(19.9)], dtype=float),
        )

    def fake_run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init, map_history):
        captured["nuts_diagnostics"] = dict(nuts_init.diagnostics)
        return cluster_solver.PosteriorResults(
            samples=np.array([[31.0, 21.0], [39.0, 29.0]], dtype=float),
            log_prob=np.array([2.5, 2.7], dtype=float),
            accept_prob=np.array([0.9, 0.92], dtype=float),
            diverging=np.zeros(2, dtype=bool),
            num_steps=np.array([5.0, 6.0], dtype=float),
            map_history=[],
            warmup_steps=5,
            sample_steps=4,
            num_chains=2,
            init_diagnostics={"strategy_used": "saved_smc_refine_nuts"},
            sampler="numpyro_nuts",
        )

    monkeypatch.setattr(cluster_solver, "_has_plot_artifacts", lambda path: True)
    monkeypatch.setattr(cluster_solver, "_load_artifacts", lambda path: (state, saved_args, arrays, {}))
    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_posterior_model", lambda *a, **k: (lambda: None))
    monkeypatch.setattr(cluster_solver, "_build_smc_refine_initialization", fake_build_smc_refine_initialization)
    monkeypatch.setattr(cluster_solver, "_run_numpyro_nuts_sampler", fake_run_numpyro_nuts_sampler)
    def fake_save_artifacts(artifacts_dir, state, args, best_fit, results):
        captured["saved"] = True
        captured["saved_state_run_name"] = state.run_name
        captured["saved_best_fit"] = np.asarray(best_fit, dtype=float)
        captured["saved_results_samples"] = np.asarray(results.samples, dtype=float)

    def fake_generate_plots_and_tables(*, run_dir, state, evaluator, best_fit, best_eval, results, runtime_sec, args):
        captured["plotted"] = True
        captured["plot_state_run_name"] = state.run_name
        captured["plot_results_samples"] = np.asarray(results.samples, dtype=float)

    monkeypatch.setattr(cluster_solver, "_save_artifacts", fake_save_artifacts)
    monkeypatch.setattr(cluster_solver, "_generate_plots_and_tables", fake_generate_plots_and_tables)

    cluster_solver._run_refine_from_saved_smc(args)

    assert captured["saved"] is True
    assert captured["plotted"] is True
    assert captured["saved_state_run_name"] == "cpu_refine"
    assert captured["plot_state_run_name"] == "cpu_refine"
    assert np.allclose(captured["refine_log_prob"], arrays["log_prob"])
    assert np.allclose(captured["refine_weights"], arrays["sample_weights"])
    assert np.allclose(captured["latent_particles"][:, 0], np.log(np.array([30.0, 40.0])))
    assert np.allclose(captured["latent_particles"][:, 1], np.log(np.array([19.9, 29.9])))
    assert captured["nuts_diagnostics"]["source_sampler"] == "blackjax_smc"
    assert np.allclose(captured["saved_results_samples"], np.array([[np.exp(31.0), 0.1 + np.exp(21.0)], [np.exp(39.0), 0.1 + np.exp(29.0)]], dtype=float))
    assert np.allclose(captured["plot_results_samples"], np.array([[np.exp(31.0), 0.1 + np.exp(21.0)], [np.exp(39.0), 0.1 + np.exp(29.0)]], dtype=float))


def test_run_refine_from_saved_smc_rejects_non_smc_source(monkeypatch, tmp_path):
    args = _args()
    args.refine_from_run_dir = str(tmp_path / "nuts_run")

    monkeypatch.setattr(cluster_solver, "_has_plot_artifacts", lambda path: True)
    monkeypatch.setattr(
        cluster_solver,
        "_load_artifacts",
        lambda path: (
            type("State", (), {"parameter_specs": []})(),
            {"sampler": "numpyro_nuts"},
            {"samples": np.empty((0, 0)), "log_prob": np.empty((0,)), "sample_weights": np.empty((0,))},
            {},
        ),
    )

    try:
        cluster_solver._run_refine_from_saved_smc(args)
    except ValueError as exc:
        assert "requires a blackjax_smc source run" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected non-SMC refinement source to fail.")


def test_maybe_convert_loaded_posterior_arrays_to_physical_for_legacy_refine():
    specs = [
        cluster_solver.ParameterSpec(
            name="potfile0.sigma",
            sample_name="potfile0_sigma",
            potential_id="potfile0",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=np.log(30.0),
            std=0.3,
            component_family="scaling",
            transform_kind="log_positive",
            physical_lower=1.0,
            physical_upper=200.0,
            physical_mean=30.0,
            physical_std=10.0,
        ),
        cluster_solver.ParameterSpec(
            name="potfile0.cutkpc",
            sample_name="potfile0_cutkpc",
            potential_id="potfile0",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="cutkpc",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=np.log(19.9),
            std=0.3,
            component_family="scaling",
            transform_kind="log_offset_positive",
            physical_lower=0.1,
            physical_upper=200.0,
            physical_mean=20.0,
            physical_std=8.0,
            transform_offset=0.1,
        ),
    ]
    latent_samples = np.array([[np.log(30.0), np.log(19.9)], [np.log(40.0), np.log(29.9)]], dtype=float)
    arrays = {
        "samples": latent_samples,
        "grouped_samples": latent_samples.reshape(1, 2, 2),
        "best_fit": np.array([30.0, 20.0], dtype=float),
    }

    converted, did_convert = cluster_solver._maybe_convert_loaded_posterior_arrays_to_physical(
        arrays,
        specs,
        {"saved_smc_refine": True},
    )

    assert did_convert is True
    assert np.allclose(converted["samples"][:, 0], np.array([30.0, 40.0], dtype=float))
    assert np.allclose(converted["samples"][:, 1], np.array([20.0, 30.0], dtype=float))
    assert np.allclose(converted["grouped_samples"][0, :, 0], np.array([30.0, 40.0], dtype=float))
    assert np.allclose(converted["grouped_samples"][0, :, 1], np.array([20.0, 30.0], dtype=float))


def test_evaluator_zero_validation_skips_source_exact_validation(monkeypatch):
    specs = _parameter_specs()

    class FakeFamily:
        def __init__(self, family_id):
            self.family_id = family_id
            self.z_source = 1.0
            self.effective_z_source = 1.0
            self.sigma_arcsec = 0.5
            self.n_images = 1

    class FakeCosmo:
        def kpc_proper_per_arcmin(self, _z):
            class FakeQuantity:
                value = 1.0

                def to(self, _unit):
                    return self

            return FakeQuantity()

    state = type(
        "State",
        (),
        {
            "cosmo_config": {},
            "parsed": {},
            "potfiles": [],
            "lens_model_list": ["PJAFFE_ELLIPSE_POTENTIAL"],
            "bin_data": [type("Bin", (), {"effective_z_source": 1.0})()],
            "z_lens": 0.375,
            "family_data": [FakeFamily("f1"), FakeFamily("f2")],
            "packed_lens_spec": type("PackedLensSpec", (), {"component_family": np.array([], dtype=np.int32)})(),
            "parameter_specs": specs,
        },
    )()

    monkeypatch.setattr(cluster_solver, "_build_cosmology", lambda _parsed: FakeCosmo())
    monkeypatch.setattr(cluster_solver, "_dpie_sigma0_factor", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(cluster_solver, "LensModel", lambda *a, **k: object())
    monkeypatch.setattr(cluster_solver.ClusterJAXEvaluator, "_build_scaling_rank_diagnostics", lambda self: cluster_solver.pd.DataFrame())
    monkeypatch.setattr(cluster_solver.ClusterJAXEvaluator, "_select_validation_families", lambda self, k: self.state.family_data[:k])
    monkeypatch.setattr(cluster_solver.ClusterJAXEvaluator, "source_loglike", lambda self, params: 1.0)
    monkeypatch.setattr(
        cluster_solver.ClusterJAXEvaluator,
        "_family_source_summary",
        lambda self, params: {
            family.family_id: {"source_plane_rms": 1.0, "source_x": 0.0, "source_y": 0.0}
            for family in self.state.family_data
        },
    )

    exact_calls = []

    def fake_exact_family_prediction(self, params, family):
        exact_calls.append(family.family_id)
        return np.array([0.0]), np.array([0.0]), 0.1

    monkeypatch.setattr(cluster_solver.ClusterJAXEvaluator, "_exact_family_prediction", fake_exact_family_prediction)

    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=1.5,
        validate_top_k_families=0,
        sampling_engine="full",
        validation_approx="adaptive",
    )

    result = evaluator.evaluate(np.array([0.0, 0.2]), likelihood_mode="source")

    assert evaluator.validation_family_ids == set()
    assert exact_calls == []
    assert result.used_exact_validation is False


def test_evaluator_zero_validation_keeps_image_mode_exact_validation(monkeypatch):
    specs = _parameter_specs()

    class FakeFamily:
        def __init__(self, family_id):
            self.family_id = family_id
            self.z_source = 1.0
            self.effective_z_source = 1.0
            self.sigma_arcsec = 0.5
            self.n_images = 1

    class FakeCosmo:
        def kpc_proper_per_arcmin(self, _z):
            class FakeQuantity:
                value = 1.0

                def to(self, _unit):
                    return self

            return FakeQuantity()

    state = type(
        "State",
        (),
        {
            "cosmo_config": {},
            "parsed": {},
            "potfiles": [],
            "lens_model_list": ["PJAFFE_ELLIPSE_POTENTIAL"],
            "bin_data": [type("Bin", (), {"effective_z_source": 1.0})()],
            "z_lens": 0.375,
            "family_data": [FakeFamily("f1"), FakeFamily("f2")],
            "packed_lens_spec": type("PackedLensSpec", (), {"component_family": np.array([], dtype=np.int32)})(),
            "parameter_specs": specs,
        },
    )()

    monkeypatch.setattr(cluster_solver, "_build_cosmology", lambda _parsed: FakeCosmo())
    monkeypatch.setattr(cluster_solver, "_dpie_sigma0_factor", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(cluster_solver, "LensModel", lambda *a, **k: object())
    monkeypatch.setattr(cluster_solver.ClusterJAXEvaluator, "_build_scaling_rank_diagnostics", lambda self: cluster_solver.pd.DataFrame())
    monkeypatch.setattr(cluster_solver.ClusterJAXEvaluator, "_select_validation_families", lambda self, k: self.state.family_data[:k])
    monkeypatch.setattr(cluster_solver.ClusterJAXEvaluator, "source_loglike", lambda self, params: 1.0)
    monkeypatch.setattr(
        cluster_solver.ClusterJAXEvaluator,
        "_family_source_summary",
        lambda self, params: {
            family.family_id: {"source_plane_rms": 1.0, "source_x": 0.0, "source_y": 0.0}
            for family in self.state.family_data
        },
    )

    exact_calls = []

    def fake_exact_family_prediction(self, params, family):
        exact_calls.append(family.family_id)
        return np.array([0.0]), np.array([0.0]), 0.1

    monkeypatch.setattr(cluster_solver.ClusterJAXEvaluator, "_exact_family_prediction", fake_exact_family_prediction)

    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=1.5,
        validate_top_k_families=0,
        sampling_engine="full",
        validation_approx="adaptive",
    )

    result = evaluator.evaluate(np.array([0.0, 0.2]), likelihood_mode="image")

    assert evaluator.validation_family_ids == set()
    assert exact_calls == ["f1", "f2"]
    assert result.used_exact_validation is True
