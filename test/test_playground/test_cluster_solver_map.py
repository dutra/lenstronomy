import numpy as np

from playground.cluster_solver import BoundedMAPOptimizer, MAPRunResult, ParameterSpec, SeededStart


class SequentialBoundedMAPOptimizer(BoundedMAPOptimizer):
    def __init__(self, evaluation_map):
        parameter_specs = [
            ParameterSpec(
                name="p0",
                sample_name="p0",
                potential_id="test",
                profile_type=81,
                field="value",
                prior_kind="uniform",
                lower=-10.0,
                upper=10.0,
                step=1.0,
            )
        ]
        super().__init__(
            parameter_specs=parameter_specs,
            logprob_fn=lambda theta: float(theta[0]),
            maxiter=5,
        )
        self.evaluation_map = {
            label: {
                "input_theta": np.asarray(input_theta, dtype=float),
                "final_theta": np.asarray(final_theta, dtype=float),
                "count": int(count),
            }
            for label, input_theta, final_theta, count in evaluation_map
        }
        self.current_label = None
        self.execution_labels = []

    def to_unconstrained(self, theta):
        theta_array = np.asarray(theta, dtype=float)
        matched_label = None
        for label, record in self.evaluation_map.items():
            if np.allclose(theta_array, record["input_theta"]):
                matched_label = label
                break
        if matched_label is not None:
            self.current_label = matched_label
        return np.asarray(theta, dtype=float)

    def run_single(self, init_theta_unconstrained, tol):
        label = self.current_label
        if label is None:
            raise AssertionError("run_single called before a restart label was established.")
        self.execution_labels.append(label)
        record = self.evaluation_map[label]
        return np.asarray(record["final_theta"], dtype=float), np.asarray(record["count"], dtype=int)

    def to_constrained(self, theta_unconstrained):
        return np.asarray(theta_unconstrained, dtype=float)

    def prior_log_prob(self, theta):
        return 0.0


def test_bounded_map_optimizer_runs_restarts_sequentially():
    starts = [
        SeededStart(values=np.array([0.0]), label="midpoint"),
        SeededStart(values=np.array([1.0]), label="broad_1"),
        SeededStart(values=np.array([2.0]), label="broad_2"),
    ]
    optimizer = SequentialBoundedMAPOptimizer(
        [
            ("midpoint", np.array([0.0]), np.array([1.0]), 4),
            ("broad_1", np.array([1.0]), np.array([3.0]), 2),
            ("broad_2", np.array([2.0]), np.array([2.0]), 6),
        ]
    )
    log_messages = []

    best_theta, history, results = optimizer.run(
        starts=starts,
        stage_label="pass1_relaxed:broad",
        log_fn=log_messages.append,
    )

    assert optimizer.execution_labels == ["midpoint", "broad_1", "broad_2"]
    assert np.allclose(best_theta, np.array([3.0]))
    assert len(history) == len(starts)
    assert len(results) == len(starts)
    assert [item["start_label"] for item in history] == ["midpoint", "broad_1", "broad_2"]
    assert history[0]["is_best_so_far"] is True
    assert history[1]["is_best_so_far"] is True
    assert history[2]["is_best_so_far"] is False
    assert all(isinstance(item, MAPRunResult) for item in results)
    assert [item.label for item in results] == ["broad_1", "broad_2", "midpoint"]
    assert [item.logprob for item in results] == [3.0, 2.0, 1.0]
    assert any("restart 1/3 seed=midpoint starting" in message for message in log_messages)
    assert any("restart 1/3 complete" in message for message in log_messages)
    assert any("restarts finished best_restart=2/3 best_loglike=3.000" in message for message in log_messages)
