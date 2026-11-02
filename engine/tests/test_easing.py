import numpy as np
import pytest
from typer.testing import CliRunner

from motionly_engine.cli import app
from motionly_engine.fit.easing import (
    EXACT,
    GSAP_ALIASES,
    PRESETS,
    _poly,
    as_samples,
    bezier,
    bezier_solve_s,
    curve_distance,
    grid,
    nearest_preset,
    spring,
    spring_curve,
    spring_settle_time,
)

T = np.linspace(0, 1, 1001)


# --- bezier -----------------------------------------------------------------

def test_linear_midpoint():
    assert bezier(0.5, PRESETS["linear"]) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("name", list(PRESETS))
def test_bezier_endpoints(name):
    assert bezier(0.0, PRESETS[name]) == pytest.approx(0.0, abs=1e-12)
    assert bezier(1.0, PRESETS[name]) == pytest.approx(1.0, abs=1e-12)


def test_css_ease_known_value():
    assert bezier(0.5, PRESETS["ease"]) == pytest.approx(0.8024, abs=1e-3)


def test_ease_in_out_symmetric():
    p = PRESETS["ease-in-out"]
    assert bezier(0.5, p) == pytest.approx(0.5, abs=1e-12)
    np.testing.assert_allclose(bezier(T, p) + bezier(1 - T, p), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    "params", list(PRESETS.values()) + [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0)]
)
def test_solver_recovers_t(params):
    s = bezier_solve_s(T, params)
    np.testing.assert_allclose(_poly(s, params[0], params[2]), T, atol=1e-9)


def test_back_overshoot():
    assert np.max(bezier(T, PRESETS["easeOutBack"])) > 1.0
    assert np.min(bezier(T, PRESETS["easeInBack"])) < 0.0


def test_invalid_x_raises():
    with pytest.raises(ValueError):
        bezier(0.5, (1.2, 0, 0.5, 1))
    with pytest.raises(ValueError):
        bezier(0.5, (0.2, 0, -0.1, 1))


def test_shapes():
    assert isinstance(bezier(0.3, PRESETS["ease"]), float)
    assert bezier(np.zeros((3, 4)), PRESETS["ease"]).shape == (3, 4)
    assert isinstance(spring(0.3, 100, 10), float)


# --- spring -----------------------------------------------------------------

@pytest.mark.parametrize("k,c", [(170, 26), (100, 5), (200, 2 * np.sqrt(200)), (100, 40)])
def test_spring_starts_and_settles(k, c):
    assert spring(0.0, k, c) == pytest.approx(0.0, abs=1e-12)
    ts = spring_settle_time(k, c)
    tail = np.linspace(ts, ts * 5 + 1, 2000)
    assert np.all(np.abs(spring(tail, k, c) - 1.0) < 1e-3)
    assert spring(ts * 20, k, c) == pytest.approx(1.0, abs=1e-6)


def test_spring_overshoot_regimes():
    t = np.linspace(0, 5, 5000)
    assert np.max(spring(t, 100, 5)) > 1.0  # underdamped
    assert np.max(spring(t, 100, 20)) <= 1.0 + 1e-12  # critical
    assert np.max(spring(t, 100, 40)) <= 1.0 + 1e-12  # overdamped


def test_spring_continuous_at_critical():
    t = np.linspace(0, 2, 500)
    k = 100.0
    crit = spring(t, k, 20.0)
    for c in (20.0 * (1 - 1e-4), 20.0 * (1 + 1e-4)):
        np.testing.assert_allclose(spring(t, k, c), crit, atol=1e-4)


def test_spring_curve_normalized():
    f = spring_curve(170, 26)
    assert f(0.0) == pytest.approx(0.0, abs=1e-12)
    assert abs(f(1.0) - 1.0) < 1e-3


def test_spring_rejects_bad_params():
    with pytest.raises(ValueError):
        spring(0.1, 0, 10)
    with pytest.raises(ValueError):
        spring(0.1, 100, -1)


# --- exact functions --------------------------------------------------------

@pytest.mark.parametrize("name", list(EXACT))
def test_exact_endpoints_and_close_to_bezier(name):
    f = EXACT[name]
    assert f(0.0) == 0.0 and f(1.0) == 1.0
    if name in PRESETS:
        assert curve_distance(f, PRESETS[name]) < 0.05


def test_exact_back_overshoot():
    assert np.max(EXACT["easeOutBack"](T)) > 1.0


def test_gsap_aliases_resolve():
    for target in GSAP_ALIASES.values():
        assert target in EXACT or target in PRESETS
    assert curve_distance("power2.out", "easeOutCubic") == 0.0


# --- distance and naming ----------------------------------------------------

def test_curve_distance_basics():
    a, b = PRESETS["linear"], PRESETS["ease-in-out"]
    assert curve_distance(a, a) == 0.0
    assert curve_distance(a, b) == curve_distance(b, a) > 0


@pytest.mark.parametrize("name", list(PRESETS))
def test_nearest_names_own_bezier(name):
    m = nearest_preset(PRESETS[name])
    assert m.name == name and m.dist < 0.03


def test_nearest_exact_expo_samples():
    samples = EXACT["easeOutExpo"](grid(201))
    m = nearest_preset(samples)
    assert m.name == "easeOutExpo" and m.dist < 0.005
    assert m.kind == "exact"


def test_nearest_perturbed_expo():
    x1, y1, x2, y2 = PRESETS["easeOutExpo"]
    m = nearest_preset((x1 + 0.01, y1 - 0.01, x2 - 0.01, y2 + 0.01))
    assert m.name == "easeOutExpo" and m.dist < 0.03


def test_as_samples_rejects_wrong_length():
    with pytest.raises(ValueError):
        as_samples(np.zeros(10))


# --- CLI --------------------------------------------------------------------

def test_cli_ease():
    r = CliRunner()
    out = r.invoke(app, ["ease", "easeOutBack"])
    assert out.exit_code == 0, out.output
    out = r.invoke(app, ["ease", ".16,1,.3,1"])
    assert out.exit_code == 0 and "easeOutExpo" in out.output
