"""Kennzahlen der Analyse (stb_evaluation.analyse) gegen unabhängige Zählungen; Lücken, die der Mutationstest gefunden hat."""

import itertools

import numpy as np
import pytest

import stb_algorithm as A
import stb_evaluation as ev
from stb_scenario import Settings

CASES = [Settings(n=n, k=k, plan=plan, x=x, p=p, c=c, stil=stil, seed=seed)
         for n, k, plan, x, p, c, stil, seed in itertools.product((5, 8, 12), (1, 3), ("vollstaendig", "staffeln", "luecken"), (1,), (0.5,),
                                                                  (0.0, 0.3), ("ausgewogen", "konter"), (11, 12))]


def independent_lot_places(W):
    """Zählt die Teilnehmer, die nach Wertung (Siege) UND Direktvergleich noch gleichstehen, also nur das Los trennt."""
    n = len(W)
    wins = [int(sum(W[i])) for i in range(n)]
    total = 0
    for w in set(wins):
        group = [i for i in range(n) if wins[i] == w]
        if len(group) < 2:
            continue
        mini = {i: int(sum(W[i][j] for j in group)) for i in group}
        for m in set(mini.values()):
            members = [i for i in group if mini[i] == m]
            if len(members) >= 2:
                total += len(members)
    return total


def test_lot_places_equal_an_independent_recount():
    seen_positive = 0
    for s in CASES:
        a = ev.analyse(s)
        if s.wertung == "siege":
            assert a.los_places == independent_lot_places(a.turnier.wins), s
            seen_positive += a.los_places > 0
    assert seen_positive > 10          # der Test sieht auch echte Lose


def test_games_total_counts_each_game_once():
    a = ev.analyse(Settings(n=6, k=3, plan="vollstaendig", seed=1))
    assert a.games_total == 45                     # C(6,2) Paare mal 3 Partien, von Hand
    for s in CASES:
        b = ev.analyse(s)
        assert b.games_total == int(b.turnier.wins.sum()), s      # jede Partie hat genau einen Sieger


def test_games_min_and_max_per_player():
    a = ev.analyse(Settings(n=6, k=3, plan="vollstaendig", seed=1))
    assert a.games_min == a.games_max == 15        # 5 Gegner mal 3 Partien


def test_crossing_errors_boundaries():
    assert A.crossing_errors((1, 0, 2), 1) == pytest.approx(0.5)      # eine schwache Person, zwei starke
    assert A.crossing_errors((2, 1, 0), 2) == 0.0                     # zwei schwache, ein starker (Stärkster vorn): kein nan
    assert np.isnan(A.crossing_errors((0, 1, 2), 0))                  # keine Schwachen
    assert np.isnan(A.crossing_errors((0, 1, 2), 3))                  # keine Starken


def _find(predicate, limit=400, **kw):
    for seed in range(limit):
        a = ev.analyse(Settings(seed=seed, **kw))
        if predicate(a):
            return a
    raise AssertionError("kein passender Fall gefunden")


def test_core_verdict_mentions_ties_only_when_there_are_some():
    with_ties = _find(lambda a: a.top_is_best_s and a.components == 1 and a.tie_share > 0, n=8, k=2)
    level, text = ev.verdict_core(with_ties)
    assert level == "success" and "punktgleich" in text and f"{with_ties.los_places} Plätze" in text
    without = _find(lambda a: a.top_is_best_s and a.components == 1 and a.tie_share == 0, n=8, k=10, delta=1.0)
    level, text = ev.verdict_core(without)
    assert level == "success" and "punktgleich" not in text


def test_core_verdict_adds_the_field_remark_only_when_expected_rate_is_much_better():
    gap = _find(lambda a: not a.top_is_best_s and a.components == 1 and a.tau_e - a.tau_s > 0.1, n=12, k=10, c=0.4, stil="konter")
    assert "Erfolg gegen **dieses** Feld" in ev.verdict_core(gap)[1]
    plain = _find(lambda a: not a.top_is_best_s and a.components == 1 and a.tau_e - a.tau_s <= 0.1, n=8, k=1)
    assert "dieses** Feld" not in ev.verdict_core(plain)[1]


def test_analysis_flags_top_is_best_consistently_with_best_place():
    for s in CASES:
        a = ev.analyse(s)
        assert a.top_is_best_s == (a.best_place == 1), s
