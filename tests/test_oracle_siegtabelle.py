"""Orakel-Tests siegtabelle-demo: Kern und angezeigte Kennzahlen gegen einen ANDEREN Rechenweg (DEMO-PLAYBOOK §12).

1. Zyklische Dreiecke: Spur-Formel der Demo und Zählformel gegen Aufzählung aller Dreiergruppen (Python-`random`, unabhängig vom Generator).
2. Kendall-τ gegen scipy.stats.kendalltau.
3. Komponenten gegen networkx.
4. Siege-Ordnung gleich Bradley-Terry-Ordnung bei vollständigem, ausgeglichenem Plan (nur wo der Schätzer existiert), gegen scipy.optimize.
5. Wahrscheinlichkeit „Platz 1 ist der wahre Beste“ für n = 3 exakt durch Aufzählung aller Ausgänge gegen die Simulation.
6. Direktvergleich bei Gleichstand gegen eine Aufzählung der Mini-Tabellen.
7. Erwartete Siegquote gegen die Aufzählung aus der Matrix P; im vollständigen Plan E[Siege]/(n-1) = e.
"""

import itertools
import math
import random

import numpy as np
import pytest

import stb_algorithm as A
import turnier_generator as T
from stb_evaluation import analyse
from stb_scenario import Settings


def random_tournament(n, rnd):
    """Vollständiges Turnier ohne Gleichstände: je Paar eine Partie."""
    W = np.zeros((n, n), dtype=int)
    for i, j in itertools.combinations(range(n), 2):
        if rnd.random() < 0.5:
            W[i, j] = 1
        else:
            W[j, i] = 1
    return W


def brute_cyclic_triples(W):
    """Aufzählung: zählt Dreiergruppen mit einem gerichteten Dreizyklus über die strenge Mehrheitskante."""
    n = len(W)
    beat = [[W[i][j] > W[j][i] for j in range(n)] for i in range(n)]
    count = 0
    for a, b, c in itertools.combinations(range(n), 3):
        if (beat[a][b] and beat[b][c] and beat[c][a]) or (beat[a][c] and beat[c][b] and beat[b][a]):
            count += 1
    return count


# --- 1. zyklische Dreiecke -------------------------------------------------------------------------------------------------------------

def test_cyclic_triads_trace_and_formula_match_enumeration_on_complete_tournaments():
    rnd = random.Random(2026)
    for _ in range(300):
        n = rnd.randint(3, 9)
        W = random_tournament(n, rnd)
        brute = brute_cyclic_triples(W)
        assert A.cyclic_triads(W) == brute
        assert A.cyclic_triads_formula(A.wins_vector(W)) == brute


def test_cyclic_triads_trace_matches_enumeration_with_ties_and_gaps():
    rnd = random.Random(7)
    for _ in range(150):
        n = rnd.randint(3, 8)
        W = np.zeros((n, n), dtype=int)
        for i, j in itertools.combinations(range(n), 2):
            games = rnd.choice([0, 0, 1, 2, 3])
            wins = sum(rnd.random() < 0.5 for _ in range(games))
            W[i, j], W[j, i] = wins, games - wins
        assert A.cyclic_triads(W) == brute_cyclic_triples(W)


# --- 2. Kendall-τ --------------------------------------------------------------------------------------------------------------------

def test_kendall_tau_matches_scipy_without_ties():
    stats = pytest.importorskip("scipy.stats")
    rnd = random.Random(11)
    for _ in range(60):
        n = rnd.randint(3, 12)
        rank = list(range(n))
        rnd.shuffle(rank)
        values = [rnd.random() for _ in range(n)]
        ours = A.kendall_tau(rank, values)
        theirs = stats.kendalltau([-r for r in rank], values).statistic     # größerer Wert = besserer (kleinerer) Rang
        assert ours == pytest.approx(theirs, abs=1e-12)


# --- 3. Komponenten ------------------------------------------------------------------------------------------------------------------

def test_components_match_networkx():
    nx = pytest.importorskip("networkx")
    rnd = random.Random(5)
    for _ in range(150):
        n = rnd.randint(2, 14)
        G = np.zeros((n, n), dtype=int)
        for i, j in itertools.combinations(range(n), 2):
            if rnd.random() < 0.2:
                G[i, j] = G[j, i] = rnd.randint(1, 3)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((i, j) for i in range(n) for j in range(i + 1, n) if G[i, j] > 0)
        assert T.components(G) == nx.number_connected_components(g)


# --- 4. Siege-Ordnung = Bradley-Terry-Ordnung -----------------------------------------------------------------------------------------

def bradley_terry_mle(W):
    """Maximum Likelihood der logarithmierten Stärken (Summe 0), per BFGS auf der negativen Log-Likelihood."""
    optimize = pytest.importorskip("scipy.optimize")
    n = len(W)

    def nll(theta):
        s = np.concatenate([theta, [-theta.sum()]])
        d = s[:, None] - s[None, :]
        return float(-(W * (-np.logaddexp(0.0, -d))).sum())

    res = optimize.minimize(nll, np.zeros(n - 1), method="BFGS", options={"gtol": 1e-9, "maxiter": 500})
    return np.concatenate([res.x, [-res.x.sum()]])


def test_wins_order_equals_bradley_terry_order_in_a_balanced_complete_tournament():
    checked = 0
    for k in (4, 6, 8):                      # genug Partien, damit Gleichstände und Null-Siege selten sind
        for seed in range(60):
            t = T.build_turnier(6, 0.3, k, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", 1000 + seed)
            wins = A.wins_vector(t.wins)
            losses = A.games_vector(t.games) - wins
            if len(set(wins)) < 6 or (wins == 0).any() or (losses == 0).any():
                continue                     # Schätzer existiert nicht oder Gleichstand: nicht auswertbar
            theta = bradley_terry_mle(t.wins)
            assert list(np.argsort(-wins, kind="stable")) == list(np.argsort(-theta, kind="stable")), (k, seed)
            checked += 1
    assert checked >= 40                      # der Test prüft wirklich etwas


# --- 5. exakte Wahrscheinlichkeit für n = 3 -------------------------------------------------------------------------------------------

def exact_top_probability(P, best):
    """Ein Partie je Paar, drei Teilnehmer: alle 8 Ausgänge. Dreiergleichstand (Zyklus) entscheidet das Los: 1/3 für jeden."""
    total = 0.0
    pairs = [(0, 1), (0, 2), (1, 2)]
    for outcome in itertools.product((0, 1), repeat=3):
        prob = 1.0
        wins = [0, 0, 0]
        for (i, j), o in zip(pairs, outcome):
            winner, loser = (i, j) if o == 0 else (j, i)
            prob *= P[winner, loser]
            wins[winner] += 1
        top = max(wins)
        leaders = [i for i in range(3) if wins[i] == top]
        total += prob * ((1.0 / len(leaders)) if best in leaders else 0.0)
    return total


def test_top_probability_for_three_players_matches_enumeration():
    delta = 0.5
    P = T.win_probabilities(T.make_strengths(3, delta))
    exact = exact_top_probability(P, best=2)
    runs = 3000
    hits = sum(analyse(Settings(n=3, delta=delta, k=1, seed=500000 + s)).top_is_best_s for s in range(runs))
    sigma = math.sqrt(exact * (1 - exact) / runs)
    assert abs(hits / runs - exact) < 4 * sigma, (hits / runs, exact)


# --- 6. Direktvergleich ----------------------------------------------------------------------------------------------------------------

def test_head_to_head_order_inside_tie_groups_matches_enumeration():
    rnd = random.Random(99)
    checked = 0
    for _ in range(200):
        n = rnd.randint(4, 8)
        W = random_tournament(n, rnd)
        wins = A.wins_vector(W)
        table = A.rank_table(wins.astype(float), W, T.SplitMix64(rnd.randint(0, 10**6)))
        # Reihenfolge nach Wertung: nie verletzt
        for a, b in itertools.combinations(range(n), 2):
            if wins[a] != wins[b]:
                assert (table.rank[a] < table.rank[b]) == (wins[a] > wins[b])
        # innerhalb jeder Gleichstandsgruppe: mehr Siege in der Mini-Tabelle heißt besserer Platz
        for w in set(wins):
            group = [i for i in range(n) if wins[i] == w]
            if len(group) < 2:
                continue
            mini = {i: sum(W[i][j] for j in group) for i in group}
            for a, b in itertools.combinations(group, 2):
                if mini[a] != mini[b]:
                    assert (table.rank[a] < table.rank[b]) == (mini[a] > mini[b])
                    checked += 1
    assert checked > 30


# --- 7. erwartete Siegquote --------------------------------------------------------------------------------------------------------------

def test_expected_rate_matches_enumeration_and_expected_wins():
    n = 6
    t0 = T.build_turnier(n, 0.3, 1, "vollstaendig", 1, 0.5, 0.3, "ausgewogen", 1)
    P = t0.P
    manual = [sum(P[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]
    assert np.allclose(t0.expected, manual)
    runs = 1500
    wins = np.zeros(n)
    for s in range(runs):
        t = T.build_turnier(n, 0.3, 1, "vollstaendig", 1, 0.5, 0.3, "ausgewogen", 7000 + s)
        # gleiche Stile und Stärken in jedem Lauf: nur die Partien ändern sich
        wins += A.wins_vector(t.wins)
    mean_rate = wins / runs / (n - 1)
    assert np.allclose(mean_rate, manual, atol=0.02)
