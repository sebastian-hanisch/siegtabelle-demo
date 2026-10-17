"""Literatur-Referenzwerte zur Zählung zyklischer Dreiecke (ORACLE-PLAYBOOK §5.1).

Quelle A (Zählformel, Beispiele, Verteilung), eingesehen 2026-10-17 als PDF-Volltext:
    Harris, I. R. (2024): Counting Tournaments with a Specified Number of Circular Triads. Journal of Integer Sequences 27,
    Article 24.8.7. https://cs.uwaterloo.ca/journals/JIS/VOL27/McShane/mcshane1.pdf
    - Theorem 6 (nach Moon, Topics on Tournaments, S. 9): Zahl der zyklischen Dreiecke = C(n,3) - Summe_i C(s_i,2), s_i Siegezahlen.
    - Abschnitt 2: Adjazenzmatrix [[0,1,0],[0,0,1],[1,0,0]] = ein zyklisches Dreieck mit Punktefolge (1,1,1); Punktefolgen
      (1,1,2,2) [Abb. 3, d = 2], (1,1,2,3,3) [d = 3], (1,2,2,2,3) [d = 4], (2,2,2,2,2) [d = 5], (1,1,1,4,4,4) [Abb. 4, d = 2],
      (1,1,2,3,4,4) [d = 4, Tabelle 1], (1,2,2,2,4,5,5), (1,1,2,4,4,4,5), (1,1,3,3,3,5,5) [alle d = 6, n = 7].
    - Beispiel 11/12 und Anhang C: Zahl g_d(n) beschrifteter Turniere mit n Spielern und genau d zyklischen Dreiecken. g_0(n) = n!,
      g_1(n) = n! (n-2)/3. Aus den erzeugenden Funktionen G_d(x) des Anhangs C (Koeffizient von x^n in x^a (1-x)^-(1+b) ist C(n-a+b, b),
      Satz 1) folgt für n = 3: {0: 6, 1: 2}; n = 4: {0: 24, 1: 16, 2: 24}; n = 5: {0: 120, 1: 120, 2: 240, 3: 240, 4: 280, 5: 24};
      n = 6: {0: 720, 1: 960, 2: 2240, 3: 2880, 4: 6240, 5: 3648, 6: 8640, 7: 4800, 8: 2640}. Die Summen sind 2^C(n,2) (die Quelle
      nennt die Summenprobe nicht; sie wurde beim Auswerten als Lesekontrolle gerechnet).
    Die Verteilung geht auf Kendall, Babington Smith (1940) zurück (Original nicht eingesehen, hier nur Harris 2024).

Quelle B (Extremalwert), eingesehen 2026-10-17 als PDF-Volltext (Zusammenfassung):
    Grzesik, A., Kráľ, D., Lovász, L. M., Volec, J. (2022): Cycles of a given length in tournaments. arXiv:2008.06577v3.
    Aussage: Ein Turnier mit n Spielern hat höchstens n(n^2-1)/24 zyklische Dreiecke bei ungeradem n und n(n^2-4)/24 bei geradem n,
    beide Schranken sind scharf (zurückgeführt auf Kendall, Babington Smith 1940 und Szele; Zusammenfassung, S. 1).
    Der Test mit dem zirkulanten Turnier (i schlägt die nächsten (n-1)/2 Teilnehmer modulo n) ist eine Handrechnung: alle Siegezahlen
    sind (n-1)/2, und C(n,3) - n C((n-1)/2, 2) = n(n^2-1)/24 nach der Formel aus Quelle A.

Die Demo rechnet Kennzahlen aus W (Siege je Paar); die Tests rufen `cyclic_triads` und `cyclic_triads_formula` auf, nicht ihr eigenes
Orakel.
"""

import functools
import itertools
import math

import numpy as np
import pytest

import stb_algorithm as A


def tournament_from_bits(n, bits):
    """Beschriftetes Turnier aus einer Bitmaske über die Paare i < j (Bit gesetzt: i schlägt j)."""
    W = np.zeros((n, n), dtype=int)
    for b, (i, j) in enumerate(itertools.combinations(range(n), 2)):
        if (bits >> b) & 1:
            W[i, j] = 1
        else:
            W[j, i] = 1
    return W


@functools.lru_cache(maxsize=None)
def _distribution(n):
    counts = {}
    for bits in range(1 << math.comb(n, 2)):
        d = A.cyclic_triads(tournament_from_bits(n, bits))
        counts[d] = counts.get(d, 0) + 1
    return tuple(sorted(counts.items()))


def distribution(n):
    return dict(_distribution(n))


# Harris (2024), Anhang C, ausgewertet nach Satz 1 (siehe Docstring)
HARRIS_DISTRIBUTION = {
    3: {0: 6, 1: 2},
    4: {0: 24, 1: 16, 2: 24},
    5: {0: 120, 1: 120, 2: 240, 3: 240, 4: 280, 5: 24},
    6: {0: 720, 1: 960, 2: 2240, 3: 2880, 4: 6240, 5: 3648, 6: 8640, 7: 4800, 8: 2640},
}


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_number_of_labelled_tournaments_by_cyclic_triads_matches_harris_2024(n):
    assert distribution(n) == HARRIS_DISTRIBUTION[n]


def test_transitive_and_single_triad_counts_follow_the_closed_forms_of_harris():
    """g_0(n) = n! (Beispiel 11) und g_1(n) = n! (n-2)/3 (Beispiel 12) für n = 3 bis 6, aus denselben Aufzählungen."""
    for n in range(3, 7):
        dist = HARRIS_DISTRIBUTION[n]          # Tabellenwerte der Quelle
        assert dist[0] == math.factorial(n)
        assert dist[1] == math.factorial(n) * (n - 2) // 3
        assert distribution(n)[0] == math.factorial(n)


# Punktefolge -> Zahl der zyklischen Dreiecke, wie in Harris (2024), Abschnitt 2 und 3, genannt
HARRIS_SCORE_SEQUENCES = [
    ((1, 1, 1), 1),
    ((1, 1, 2, 2), 2),                  # Abb. 3
    ((1, 1, 2, 3, 3), 3),
    ((1, 2, 2, 2, 3), 4),
    ((2, 2, 2, 2, 2), 5),
    ((1, 1, 1, 4, 4, 4), 2),            # Abb. 4
    ((1, 1, 2, 3, 4, 4), 4),            # Tabelle 1, N(6,4)
    ((1, 2, 2, 2, 4, 5, 5), 6),
    ((1, 1, 2, 4, 4, 4, 5), 6),
    ((1, 1, 3, 3, 3, 5, 5), 6),
]


@pytest.mark.parametrize("scores, d", HARRIS_SCORE_SEQUENCES)
def test_cyclic_triads_formula_reproduces_the_examples_of_harris_2024(scores, d):
    assert A.cyclic_triads_formula(scores) == d


def test_adjacency_matrix_of_figure_1_is_one_circular_triad_with_scores_1_1_1():
    W = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert A.cyclic_triads(W) == 1
    assert sorted(A.wins_vector(W)) == [1, 1, 1]


def test_compound_tournament_of_figure_4_has_scores_1_1_1_4_4_4_and_two_triads():
    """Zwei zyklische Dreiecke, das obere schlägt das untere: Punktefolge (1,1,1,4,4,4), Zählung 2."""
    W = np.zeros((6, 6), dtype=int)
    for a, b, c in ((0, 1, 2), (3, 4, 5)):
        W[a, b] = W[b, c] = W[c, a] = 1
    W[3:, :3] = 1
    assert sorted(A.wins_vector(W)) == [1, 1, 1, 4, 4, 4]
    assert A.cyclic_triads(W) == 2


@pytest.mark.parametrize("n, bound", [(3, 1), (4, 2), (5, 5), (6, 8)])
def test_maximum_over_all_tournaments_is_the_bound_of_grzesik_et_al(n, bound):
    formula = n * (n * n - 1) // 24 if n % 2 else n * (n * n - 4) // 24
    assert formula == bound
    assert max(distribution(n)) == bound


@pytest.mark.parametrize("n", [3, 5, 7, 9, 11])
def test_circulant_tournament_attains_the_bound_for_odd_n(n):
    W = np.zeros((n, n), dtype=int)
    for i in range(n):
        for step in range(1, (n - 1) // 2 + 1):
            W[i, (i + step) % n] = 1
    assert (A.wins_vector(W) == (n - 1) // 2).all()
    assert A.cyclic_triads(W) == n * (n * n - 1) // 24
