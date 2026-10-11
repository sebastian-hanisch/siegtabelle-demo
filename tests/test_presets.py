"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Experimente über die festen Instanzen."""

import pytest

import stb_constants as C
import stb_evaluation as ev
import stb_presets as P
from stb_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX, name
        assert C.DELTA_MIN <= p["delta"] <= C.DELTA_MAX, name
        assert C.K_MIN <= p["k"] <= C.K_MAX, name
        assert C.X_MIN <= p["x"] <= C.X_MAX, name
        assert C.P_MIN <= p["p"] <= C.P_MAX, name
        assert C.CYCLE_MIN <= p["c"] <= C.CYCLE_MAX, name
        assert p["plan"] in C.PLAN_LABELS and p["stil"] in C.STIL_LABELS and p["wertung"] in C.WERTUNG_LABELS, name
        assert 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Ausgeglichen (Voreinstellung)"]) == Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_kept_keys_are_exactly_the_hideable_sliders():
    assert set(P.KEPT_KEYS) == {"x_slider", "p_slider", "stil_select"}


def test_each_preset_tells_its_story():
    """Jedes Preset zeigt das, was seine Beschreibung verspricht (über die festen Instanzen gemessen, nicht an einem Seed)."""
    seeds = C.SWEEP_SEEDS[:60]

    def mean(preset, attr):
        vals = [getattr(ev.analyse(Settings(**{**C.PRESETS[preset], "seed": s})), attr) for s in seeds]
        return sum(float(v) for v in vals) / len(vals)

    assert mean("Ausgeglichen (Voreinstellung)", "top_is_best_s") < 0.5              # Platz 1 stimmt in weniger als der Hälfte
    assert mean("Zwei Staffeln", "tau_s") < 0.55
    assert mean("Lückenhafter Plan", "components") > 1.0                              # oft zerfallender Plan
    assert mean("Konter-Stil", "best_place") > 2.0                                    # der Stärkste rutscht
    assert mean("Konter-Stil", "tau_e") > mean("Konter-Stil", "tau_s") + 0.1          # trifft e besser als s


def test_experiments_use_the_fixed_instances_and_consistent_values():
    rows = ev.sweep_best_case(seeds=C.SWEEP_SEEDS[:20])
    assert [r["k"] for r in rows] == list(C.K_SWEEP)
    for r in rows:
        assert -1 <= r["tau"] <= 1 and 0 <= r["top"] <= 1 and 0 <= r["ties"] <= 1
    assert rows[-1]["tau"] > rows[0]["tau"]                                           # mehr Partien, bessere Ordnung


def test_sweep_seeds_are_fixed_and_numerous():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200


@pytest.mark.parametrize("plan", ["vollstaendig", "staffeln", "luecken"])
def test_every_plan_runs_at_the_slider_extremes(plan):
    for n in (C.N_MIN, C.N_MAX):
        for k in (C.K_MIN, C.K_MAX):
            a = ev.analyse(Settings(n=n, k=k, plan=plan, x=C.X_MAX, p=C.P_MIN, c=C.CYCLE_MAX, stil="konter", seed=3))
            assert 1 <= a.best_place <= n
            assert a.components >= 1
