"""Einheitentests des Kerns gegen die von Hand gerechneten Mini-Instanzen (conftest.py)."""

import numpy as np
import pytest

import stb_algorithm as A
import turnier_generator as T


def rng(seed=1):
    return T.SplitMix64(seed)


# --- Wertung ---------------------------------------------------------------------------------------------------------------------------

def test_wins_and_games_of_the_cycle_instance(mini_cycle):
    W, G = mini_cycle
    assert list(A.wins_vector(W)) == [2, 2, 2, 0]
    assert list(A.games_vector(G)) == [3, 3, 3, 3]


def test_score_vector_siege_and_quote(mini_cycle):
    W, G = mini_cycle
    assert list(A.score_vector(W, G, "siege")) == [2, 2, 2, 0]
    assert A.score_vector(W, G, "quote").tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3, 0.0])


def test_quote_without_games_is_zero_not_nan():
    W = np.zeros((3, 3), dtype=int)
    W[0, 1] = 1
    G = W + W.T
    q = A.score_vector(W, G, "quote")
    assert q.tolist() == [1.0, 0.0, 0.0]
    assert not np.isnan(q).any()


def test_unknown_wertung_raises(mini_cycle):
    W, G = mini_cycle
    with pytest.raises(ValueError):
        A.score_vector(W, G, "elo")


def test_quote_differs_from_siege_when_game_counts_differ():
    """Branch-Test: ohne ungleiche Partienzahlen wäre die Siegquote nur eine Skalierung der Siege."""
    W = np.zeros((3, 3), dtype=int)
    W[0, 1] = 1          # 0 schlägt 1 (eine Partie)
    W[2, 0] = 0
    G = np.array([[0, 1, 2], [1, 0, 0], [2, 0, 0]])
    W[0, 2] = 1          # 0 und 2 spielen zweimal, je ein Sieg
    W[2, 0] = 1
    siege = A.score_vector(W, G, "siege")
    quote = A.score_vector(W, G, "quote")
    assert siege.tolist() == [2, 0, 1]
    assert quote.tolist() == pytest.approx([2 / 3, 0.0, 0.5])
    assert list(np.argsort(-siege)) == list(np.argsort(-quote))   # gleiche Ordnung hier, aber andere Zahlen
    assert siege[0] / siege[2] != pytest.approx(quote[0] / quote[2])


# --- Tabelle und Gleichstandsregel -------------------------------------------------------------------------------------------------------

def test_split_ties_groups_equal_values_descending():
    assert A.split_ties([2, 2, 2, 0]) == [[0, 1, 2], [3]]
    assert A.split_ties([1, 3, 3, 2]) == [[1, 2], [3], [0]]
    assert A.split_ties([]) == []


def test_cycle_instance_three_way_tie_is_decided_by_lot_and_last_place_by_score(mini_cycle):
    W, G = mini_cycle
    t = A.rank_table(A.score_vector(W, G, "siege"), W, rng(7))
    assert t.order[3] == 3
    assert sorted(t.order[:3]) == [0, 1, 2]
    assert t.decided_by == ("Los", "Los", "Los", "Wertung")
    assert [t.rank[i] for i in t.order] == [0, 1, 2, 3]


def test_head_to_head_decides_before_the_lot(mini_headtohead):
    W, G = mini_headtohead
    t = A.rank_table(A.score_vector(W, G, "siege"), W, rng(3))
    assert t.order == (0, 1, 2, 3)
    assert t.decided_by == ("Direktvergleich",) * 4


def test_the_lot_is_reproducible_and_actually_varies(mini_cycle):
    W, G = mini_cycle
    score = A.score_vector(W, G, "siege")
    first = A.rank_table(score, W, rng(11)).order
    assert A.rank_table(score, W, rng(11)).order == first
    orders = {A.rank_table(score, W, rng(s)).order for s in range(40)}
    assert len(orders) >= 4        # 3! = 6 mögliche Reihenfolgen der drei Gleichstehenden; das Los nutzt sie


def test_head_to_head_wins_counts_only_inside_the_group(mini_headtohead):
    W, _ = mini_headtohead
    assert A.head_to_head_wins(W, [0, 1]).tolist() == [1, 0]
    assert A.head_to_head_wins(W, [2, 3]).tolist() == [1, 0]


# --- Kennzahlen ------------------------------------------------------------------------------------------------------------------------

def test_kendall_tau_hand_examples():
    assert A.kendall_tau((0, 2, 1), (3, 1, 2)) == 1.0           # Tabelle ordnet wie die Werte
    assert A.kendall_tau((2, 1, 0), (1, 2, 3)) == 1.0           # größerer Wert = kleinerer Rang
    assert A.kendall_tau((0, 1, 2), (1, 2, 3)) == -1.0          # genau umgekehrt
    assert A.kendall_tau((1, 0, 2), (3, 2, 1)) == pytest.approx(1 / 3)


def test_kendall_tau_ignores_pairs_with_equal_values():
    # Werte (1, 1, 2): nur die zwei Paare mit Index 2 zählen; Tabelle stellt 2 an die Spitze: beide konkordant
    assert A.kendall_tau((1, 2, 0), (1.0, 1.0, 2.0)) == 1.0
    assert np.isnan(A.kendall_tau((0, 1), (5.0, 5.0)))


def test_contradiction_share_hand_instance(mini_contradiction):
    W, G = mini_contradiction
    score = A.wins_vector(W)
    assert score.tolist() == [2, 1, 2, 1]
    assert A.contradiction_share(score, W) == pytest.approx(0.25)


def test_contradiction_share_is_zero_without_contradiction_and_nan_without_pairs(mini_cycle):
    W, _ = mini_cycle
    assert A.contradiction_share(A.wins_vector(W), W) == 0.0
    W0 = np.zeros((3, 3), dtype=int)
    assert np.isnan(A.contradiction_share(np.zeros(3), W0))


def test_tie_share(mini_cycle):
    W, _ = mini_cycle
    assert A.tie_share(A.wins_vector(W)) == pytest.approx(0.5)      # 3 von 6 Paaren


def test_cyclic_triads_hand_instances(mini_cycle, mini_headtohead):
    assert A.cyclic_triads(mini_cycle[0]) == 1
    assert A.cyclic_triads(mini_headtohead[0]) == 2
    assert A.cyclic_triads_formula(A.wins_vector(mini_cycle[0])) == 1
    assert A.cyclic_triads_formula(A.wins_vector(mini_headtohead[0])) == 2


def test_cyclic_triads_need_strict_majorities():
    """Gleichstand im direkten Vergleich und ungespielte Paare bilden keine Kante: kein Zyklus."""
    W = np.zeros((3, 3), dtype=int)
    W[0, 1] = W[1, 0] = 1          # 0 und 1 je ein Sieg: keine Kante
    W[1, 2] = 1
    W[2, 0] = 1
    assert A.cyclic_triads(W) == 0
    W2 = np.zeros((3, 3), dtype=int)
    W2[0, 1] = 1
    W2[1, 2] = 1                   # 2 gegen 0 nicht gespielt
    assert A.cyclic_triads(W2) == 0


def test_components_of_two_islands_and_of_a_connected_plan():
    G = np.zeros((4, 4), dtype=int)
    G[0, 1] = G[1, 0] = 1
    G[2, 3] = G[3, 2] = 2
    assert T.components(G) == 2
    G[1, 2] = G[2, 1] = 1
    assert T.components(G) == 1
    assert T.components(np.zeros((3, 3), dtype=int)) == 3


def test_crossing_errors_hand_instance():
    # n = 4, schwach = Index 0, 1 (Ränge 0 und 3), stark = Index 2, 3 (Ränge 1 und 2): vier Paare, zwei davon falsch geordnet
    assert A.crossing_errors((0, 3, 1, 2), 2) == pytest.approx(0.5)
    assert A.crossing_errors((3, 2, 1, 0), 2) == 0.0
    assert A.crossing_errors((0, 1, 2, 3), 2) == 1.0
