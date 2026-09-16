import numpy as np
import pytest

from euclid_agn.fit.linear import LinearProblem, delta_chi2, solve


def straight_line_problem(n=60, slope=2.0, intercept=1.0, noise=0.1, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    design = np.column_stack([np.ones_like(x), x])
    truth = intercept + slope * x
    data = truth + rng.normal(0.0, noise, n)
    variance = np.full(n, noise**2)
    return LinearProblem(
        design=design, data=data, variance=variance, names=("intercept", "slope"), **kwargs
    )


def test_unbounded_solve_recovers_the_truth():
    solution = solve(straight_line_problem(noise=1e-6))
    assert solution.value("intercept") == pytest.approx(1.0, abs=1e-4)
    assert solution.value("slope") == pytest.approx(2.0, abs=1e-4)
    assert solution.effective_dof == pytest.approx(2.0, abs=1e-6)


def test_chi2_is_about_the_number_of_data_points_for_a_correct_model():
    solution = solve(straight_line_problem(n=500, noise=0.1, seed=3))
    assert solution.chi2 == pytest.approx(500 - 2, rel=0.15)
    assert solution.chi2_reduced == pytest.approx(1.0, rel=0.15)


def test_errors_match_the_analytic_least_squares_errors():
    n, noise = 200, 0.2
    problem = straight_line_problem(n=n, noise=noise, seed=5)
    solution = solve(problem)
    a_w, _ = problem.weighted()
    expected = np.sqrt(np.diag(np.linalg.inv(a_w.T @ a_w)))
    assert np.allclose(solution.errors(), expected, rtol=1e-8)


def test_non_negativity_is_enforced():
    n = 40
    x = np.linspace(0.0, 1.0, n)
    design = np.column_stack([np.ones_like(x), x])
    data = 1.0 - 0.5 * x  # genuinely decreasing: the slope wants to go negative
    problem = LinearProblem(
        design=design,
        data=data,
        variance=np.full(n, 0.01),
        lower=np.array([-np.inf, 0.0]),
        names=("intercept", "slope"),
    )
    solution = solve(problem)
    assert solution.value("slope") >= 0.0
    assert solution.at_bound[1]
    assert np.isnan(solution.error("slope"))


def test_pinned_parameters_report_nan_errors_but_free_ones_do_not():
    n = 40
    x = np.linspace(0.0, 1.0, n)
    design = np.column_stack([np.ones_like(x), x])
    problem = LinearProblem(
        design=design,
        data=1.0 - 0.5 * x,
        variance=np.full(n, 0.01),
        lower=np.array([-np.inf, 0.0]),
        names=("intercept", "slope"),
    )
    solution = solve(problem)
    assert np.isfinite(solution.error("intercept"))
    assert np.isnan(solution.error("slope"))


def test_regularisation_shrinks_the_effective_degrees_of_freedom():
    n = 80
    x = np.linspace(0.0, 1.0, n)
    design = np.column_stack([x**k for k in range(8)])
    data = np.sin(3 * x)
    variance = np.full(n, 1e-4)
    plain = solve(LinearProblem(design=design, data=data, variance=variance))
    penalised = solve(
        LinearProblem(
            design=design,
            data=data,
            variance=variance,
            regularisation=1e3 * np.eye(8),
        )
    )
    assert plain.effective_dof == pytest.approx(8.0, abs=1e-6)
    assert penalised.effective_dof < plain.effective_dof
    assert penalised.penalty > 0.0
    assert penalised.chi2 > plain.chi2


def test_delta_chi2_is_positive_when_the_extra_component_helps():
    n = 100
    x = np.linspace(-1.0, 1.0, n)
    rng = np.random.default_rng(1)
    signal = np.exp(-0.5 * (x / 0.1) ** 2)
    data = 1.0 + 0.5 * signal + rng.normal(0.0, 0.02, n)
    variance = np.full(n, 0.02**2)
    null = solve(LinearProblem(design=np.ones((n, 1)), data=data, variance=variance))
    alternative = solve(
        LinearProblem(
            design=np.column_stack([np.ones(n), signal]),
            data=data,
            variance=variance,
            lower=np.array([-np.inf, 0.0]),
        )
    )
    assert delta_chi2(null, alternative) > 100.0
    assert alternative.log_likelihood > null.log_likelihood


def test_delta_chi2_is_zero_when_the_extra_component_is_pinned_out():
    n = 100
    rng = np.random.default_rng(2)
    data = 1.0 + rng.normal(0.0, 0.02, n)
    variance = np.full(n, 0.02**2)
    x = np.linspace(-1.0, 1.0, n)
    # A strictly negative-going feature that a non-negative amplitude cannot use.
    feature = -np.exp(-0.5 * (x / 0.1) ** 2)
    null = solve(LinearProblem(design=np.ones((n, 1)), data=data, variance=variance))
    alternative = solve(
        LinearProblem(
            design=np.column_stack([np.ones(n), feature]),
            data=data,
            variance=variance,
            lower=np.array([-np.inf, 0.0]),
        )
    )
    assert alternative.coefficients[1] == pytest.approx(0.0, abs=1e-10)
    assert delta_chi2(null, alternative) == pytest.approx(0.0, abs=1e-6)


def test_invalid_inputs_are_rejected():
    n = 10
    design = np.ones((n, 2))
    data = np.ones(n)
    with pytest.raises(ValueError):
        LinearProblem(design=design, data=np.ones(n + 1), variance=np.ones(n + 1))
    with pytest.raises(ValueError):
        LinearProblem(design=design, data=data, variance=np.zeros(n))
    with pytest.raises(ValueError):
        LinearProblem(design=design, data=data, variance=-np.ones(n))
    with pytest.raises(ValueError):
        LinearProblem(design=design, data=data, variance=np.ones(n), names=("a",))
    bad = design.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        LinearProblem(design=bad, data=data, variance=np.ones(n))
