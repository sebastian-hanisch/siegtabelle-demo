"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, nie ein einzelner Seed).
Schwellen mit Abstand zur gemessenen Zahl; SplitMix64 macht die Läufe plattformunabhängig, das Band schützt vor Änderungen am Kern."""

import math

import numpy as np
import pytest

import stb_constants as C
import stb_evaluation as ev
import turnier_generator as T
from stb_scenario import Settings, make_turnier


@pytest.fixture(scope="module")
def best_case():
    return {r["k"]: r for r in ev.sweep_best_case()}


@pytest.fixture(scope="module")
def staffeln():
    return {r["x"]: r for r in ev.sweep_staffeln()}


@pytest.fixture(scope="module")
def luecken():
    return {r["p"]: r for r in ev.sweep_luecken()}


@pytest.fixture(scope="module")
def zyklen():
    return {(r["stil"], r["c"]): r for r in ev.sweep_zyklen()}


def test_help_text_nearest_neighbour_win_probability_is_about_57_percent():
    assert 1 / (1 + math.exp(-0.3)) == pytest.approx(0.574, abs=0.001)


def test_help_text_first_beats_last_with_about_96_percent_at_twelve_participants():
    t = make_turnier(Settings(n=12, delta=0.3, k=1, c=0.0, seed=1))
    assert t.P[11, 0] == pytest.approx(0.964, abs=0.001)             # Erster gegen Letzten, ohne Störung
    assert t.P[11, 10] == pytest.approx(0.574, abs=0.001)            # Erster gegen den nächsten Nachbarn


# --- Experiment 1 ------------------------------------------------------------------------------------------------------------------------

def test_one_game_per_pair_the_strongest_leads_in_less_than_half_of_the_runs(best_case):
    assert 0.30 < best_case[1]["top"] < 0.50
    assert best_case[1]["tau"] == pytest.approx(0.53, abs=0.05)


def test_more_games_help_monotonically_and_forty_games_nearly_fix_it(best_case):
    taus = [best_case[k]["tau"] for k in C.K_SWEEP]
    tops = [best_case[k]["top"] for k in C.K_SWEEP]
    assert taus == sorted(taus) and tops == sorted(tops)
    assert best_case[40]["tau"] > 0.95 and best_case[40]["top"] > 0.85


def test_ties_are_common_with_one_game_and_rare_with_forty(best_case):
    assert best_case[1]["ties"] > 0.10 and best_case[40]["ties"] < 0.01


def _strongly_connected(W) -> bool:
    """Existenzkriterium des Bradley-Terry-Schätzers (Zermelo, Ford): Der Gewinngraph (i -> j, wenn i mindestens einmal gegen j gewann) ist stark zusammenhängend."""
    reach = (np.asarray(W) > 0) | np.eye(len(W), dtype=bool)
    for m in range(len(W)):
        reach = reach | (reach[:, [m]] & reach[[m], :])
    return bool(reach.all())


def _share_without_estimator(n: int) -> float:
    runs = [make_turnier(Settings(n=n, delta=0.3, k=1, plan="vollstaendig", seed=s)) for s in C.SWEEP_SEEDS]
    return sum(not _strongly_connected(t.wins) for t in runs) / len(runs)


def test_with_one_game_per_pair_the_bradley_terry_estimator_is_missing_in_a_third_of_the_runs():
    assert _share_without_estimator(8) == pytest.approx(0.335, abs=0.006)
    assert _share_without_estimator(12) == pytest.approx(0.37, abs=0.006)


def test_strong_connectivity_criterion_agrees_with_networkx():
    nx = pytest.importorskip("networkx")
    for s in C.SWEEP_SEEDS[:60]:
        t = make_turnier(Settings(n=8, delta=0.3, k=1, plan="vollstaendig", seed=s))
        g = nx.from_numpy_array((t.wins > 0).astype(int), create_using=nx.DiGraph)
        assert _strongly_connected(t.wins) == nx.is_strongly_connected(g), s


# --- Experiment 2 ------------------------------------------------------------------------------------------------------------------------

def test_without_crossing_games_the_two_leagues_are_ordered_like_a_coin_flip(staffeln):
    assert staffeln[0]["wrong"] == pytest.approx(0.5, abs=0.04)
    assert staffeln[0]["components"] == 2.0 and staffeln[0]["disconnected"] == 1.0


def test_error_falls_with_every_crossing_game_but_never_reaches_zero(staffeln):
    wrong = [staffeln[x]["wrong"] for x in C.X_SWEEP]
    assert wrong == sorted(wrong, reverse=True)
    assert 0.02 < staffeln[6]["wrong"] < 0.12
    assert all(staffeln[x]["components"] == 1.0 for x in C.X_SWEEP if x > 0)


# --- Experiment 3 ------------------------------------------------------------------------------------------------------------------------

def test_quote_beats_wins_when_games_are_missing(luecken):
    for p in (0.75, 0.5, 0.33, 0.2):
        assert luecken[p]["tau_quote"] > luecken[p]["tau_siege"], p
    assert luecken[1.0]["tau_quote"] == pytest.approx(luecken[1.0]["tau_siege"], abs=1e-9)   # ohne Lücken dieselbe Wertung


def test_sparse_plans_fall_apart_and_the_contradiction_hides_it(luecken):
    assert 0.05 < luecken[0.33]["disconnected"] < 0.35
    assert luecken[0.2]["disconnected"] > 0.5
    assert luecken[0.12]["contradiction"] < luecken[1.0]["contradiction"] / 2


# --- Experiment 4 ------------------------------------------------------------------------------------------------------------------------

def test_balanced_styles_leave_the_order_nearly_untouched_although_many_triples_cycle(zyklen):
    r = zyklen[("ausgewogen", 0.4)]
    assert r["tau_s"] > 0.95 and r["best_place"] == pytest.approx(1.0, abs=0.1)
    assert r["cyclic"] > 20 and r["contradiction"] > 0.2


def test_countered_strongest_slides_down_but_expected_rate_is_hit(zyklen):
    r = zyklen[("konter", 0.4)]
    assert r["best_place"] > 4 and r["tau_s"] < 0.7
    assert r["tau_e"] > 0.9 and r["top_e"] > 0.9 and r["top_s"] < 0.05
    assert r["contradiction"] < 0.1                                   # in sich stimmig
    assert zyklen[("konter", 0.2)]["best_place"] > 2.0


def test_prey_dominated_field_keeps_the_strongest_in_front(zyklen):
    assert zyklen[("beute", 0.2)]["best_place"] < 1.5


def test_table_hits_expected_rate_in_every_configuration(zyklen):
    assert all(r["tau_e"] > 0.9 for r in zyklen.values())


# --- Jede Zahl des README, mit der Rundung, die dort steht -------------------------------------------------------------------------------

def test_readme_table_numbers_experiment_1(best_case):
    for k, top in ((1, 0.405), (4, 0.645), (10, 0.745), (40, 0.93)):
        assert best_case[k]["top"] == pytest.approx(top, abs=0.0005), k
    for k, tau in ((1, 0.53), (4, 0.78), (10, 0.88), (40, 0.98)):
        assert best_case[k]["tau"] == pytest.approx(tau, abs=0.006), k
    assert best_case[1]["ties"] == pytest.approx(0.14, abs=0.006)
    assert best_case[40]["ties"] == pytest.approx(0.002, abs=0.0006)


def test_readme_table_numbers_experiment_2(staffeln):
    for x, wrong in ((0, 0.50), (1, 0.35), (3, 0.165), (6, 0.06)):
        assert staffeln[x]["wrong"] == pytest.approx(wrong, abs=0.006), x


def test_readme_table_numbers_experiment_3(luecken):
    assert luecken[0.5]["tau_quote"] == pytest.approx(0.58, abs=0.006)
    assert luecken[0.5]["tau_siege"] == pytest.approx(0.50, abs=0.006)
    assert luecken[0.33]["disconnected"] == pytest.approx(0.17, abs=0.006)
    assert luecken[0.2]["disconnected"] == pytest.approx(0.72, abs=0.006)
    assert luecken[1.0]["contradiction"] == pytest.approx(0.17, abs=0.006)
    assert luecken[0.12]["contradiction"] == pytest.approx(0.07, abs=0.006)


def test_readme_table_numbers_experiment_4(zyklen):
    a = zyklen[("ausgewogen", 0.4)]
    assert a["tau_s"] == pytest.approx(0.99, abs=0.006)
    assert a["cyclic"] == pytest.approx(47, abs=0.6) and a["triples"] == 220
    assert a["contradiction"] == pytest.approx(0.28, abs=0.006)
    k2, k4 = zyklen[("konter", 0.2)], zyklen[("konter", 0.4)]
    assert k2["best_place"] == pytest.approx(3.3, abs=0.05) and k2["tau_s"] == pytest.approx(0.74, abs=0.006)
    assert k4["best_place"] == pytest.approx(5.7, abs=0.05) and k4["tau_s"] == pytest.approx(0.525, abs=0.001)
    assert k2["contradiction"] < 0.05 and k4["contradiction"] < 0.05
    assert k4["tau_e"] == pytest.approx(0.97, abs=0.006)
    assert all(0.965 <= r["tau_e"] <= 1.0 for r in zyklen.values())
    assert zyklen[("beute", 0.2)]["best_place"] == pytest.approx(1.0, abs=0.05)


# --- Texte -------------------------------------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_values(best_case, staffeln, luecken, zyklen):
    _, text = ev.verdict_staffeln(list(staffeln.values()))
    assert ev.pct(staffeln[0]["wrong"]) in text and ev.pct(staffeln[6]["wrong"]) in text
    _, text = ev.verdict_best_case(list(best_case.values()))
    assert ev.pct(best_case[1]["top"]) in text and ev.de(best_case[40]["tau"]) in text
    _, text = ev.verdict_luecken(list(luecken.values()))
    assert ev.de(luecken[0.5]["tau_quote"]) in text and ev.pct(luecken[0.2]["disconnected"]) in text
    _, text = ev.verdict_zyklen(list(zyklen.values()))
    assert ev.de(zyklen[("konter", 0.4)]["best_place"], 1) in text


def test_core_verdict_levels():
    perfect = ev.analyse(ev.Settings(n=8, delta=1.0, k=10, seed=3))
    assert ev.verdict_core(perfect)[0] == "success"
    split = ev.analyse(ev.Settings(n=12, plan="staffeln", x=0, seed=3))
    level, text = ev.verdict_core(split)
    assert level == "warning" and "Teile" in text
    assert ev.verdict_core(ev.analyse(ev.Settings(n=12, k=10, c=0.4, stil="konter", seed=3)))[0] == "warning"


def test_decimal_comma_in_user_texts():
    assert ev.de(0.5) == "0,50" and ev.de(3.14159, 1) == "3,1" and ev.pct(0.405) == "40"


def test_unused_constants_stay_consistent():
    assert set(C.PLAN_LABELS) == set(T.PLANS) and set(C.STIL_LABELS) == set(T.STILE)
