"""Eigenschaftstests (metamorph) der Tabelle nach Siegen (ORACLE-PLAYBOOK §5.2).

Keine Einzelwerte, sondern Beziehungen, die für das Verfahren gelten müssen: Man verändert die Eingabe auf eine Weise, deren Wirkung
auf die Ausgabe bekannt ist, und prüft diese Beziehung an 25 bis 40 Zufallsinstanzen mit festem Seed (SplitMix64) plus Randfällen.

Belegte Aussagen (Begründung jeweils am Test):
1. Umbenennen der Teilnehmer ändert nur die Namen der Rangfolge (Plätze, die nicht das Los entscheidet, wandern mit).
2. Spiegeln aller Ergebnisse (W -> W.T) kehrt die Tabelle um, sofern jeder gegen jeden gleich oft spielt; Zyklen, Gleichstandsanteil
   und Widerspruchsanteil bleiben unverändert.
3. Ein zusätzlicher Sieg (ein Ausgang von Niederlage auf Sieg gedreht) verschlechtert den Sieger nie, verbessert den Verlierer nie.
4. Jede Partie m-fach wiederholen (W, G mit m multipliziert) ändert die Tabelle nicht.
5. Ein transitives Turnier ergibt genau die zugrunde liegende Reihenfolge, ohne Los, ohne Zyklus.
6. Das Los vertauscht nur innerhalb von Gruppen, die nach Wertung und Direktvergleich gleichstehen; es ist bei gleichem Seed
   reproduzierbar.
7. Kendall-τ ist invariant unter streng monotonen Transformationen der Vergleichsgröße und gemeinsamem Umnummerieren, wechselt das
   Vorzeichen bei Negation der Größe oder Umkehr der Tabelle.
8. Randfälle: ein Teilnehmer, zwei ohne und mit Partie, Turnier ohne jede Partie.
"""

import math

import numpy as np
import pytest

import stb_algorithm as A
import turnier_generator as T

N_INST = 30


# --- Zufallsinstanzen -----------------------------------------------------------------------------------------------------------------

def random_instance(seed, n, kmax=3, p_play=1.0, equal_games=None):
    """(W, G): jedes Paar spielt mit Wahrscheinlichkeit p_play 1..kmax Partien (oder genau `equal_games`), Siege gleichverteilt."""
    rng = T.SplitMix64(seed)
    iu = np.triu_indices(n, 1)
    m = len(iu[0])
    u = rng.uniform(3 * m).reshape(3, m) if m else np.zeros((3, 0))
    if equal_games is None:
        games = np.where(u[0] < p_play, 1 + np.floor(u[1] * kmax).astype(int), 0)
    else:
        games = np.full(m, int(equal_games))
    wins = np.floor(u[2] * (games + 1)).astype(int)
    W = np.zeros((n, n), dtype=int)
    W[iu] = wins
    W[iu[1], iu[0]] = games - wins
    return W, W + W.T


def instances(base_seed, count=N_INST, **kw):
    """Liste (W, G) mit wechselnder Teilnehmerzahl 2..8."""
    out = []
    for i in range(count):
        n = 2 + (i * 5 + base_seed) % 7
        out.append(random_instance(base_seed * 1000 + i, n, **kw))
    return out


def table_of(W, G, wertung="siege", seed=7):
    return A.rank_table(A.score_vector(W, G, wertung), W, T.SplitMix64(seed))


def positions_not_by_lot(tab):
    """{Teilnehmer: Platz} nur für Teilnehmer, deren Platz nicht das Los entschieden hat."""
    return {p: pos for pos, p in enumerate(tab.order) if tab.decided_by[pos] != "Los"}


def has_lot(tab):
    return "Los" in tab.decided_by


def lot_runs(tab, score):
    """Mengen von Teilnehmern in zusammenhängenden Los-Läufen gleicher Wertung."""
    runs, cur = [], []
    for pos, p in enumerate(tab.order):
        if tab.decided_by[pos] == "Los" and (not cur or score[cur[-1]] == score[p]):
            cur.append(p)
        else:
            if cur:
                runs.append(frozenset(cur))
            cur = [p] if tab.decided_by[pos] == "Los" else []
    if cur:
        runs.append(frozenset(cur))
    return sorted(runs, key=lambda s: sorted(s))


# --- 1. Umbenennen ------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("wertung", ["siege", "quote"])
def test_relabeling_participants_only_relabels_the_table(wertung):
    """Teilnehmer a der neuen Eingabe ist der alte Teilnehmer perm[a]. Plätze ohne Los wandern mit, Entscheidungsart und
    Wertungsfolge entlang der Tabelle bleiben gleich, Kennzahlen sind invariant."""
    checked_lot_free = 0
    for idx, (W, G) in enumerate(instances(1, p_play=0.8)):
        n = len(W)
        perm = T.SplitMix64(500 + idx).permutation(n)
        W2, G2 = W[np.ix_(perm, perm)], G[np.ix_(perm, perm)]
        s1, s2 = A.score_vector(W, G, wertung), A.score_vector(W2, G2, wertung)
        assert s2 == pytest.approx(s1[perm])
        t1, t2 = table_of(W, G, wertung), table_of(W2, G2, wertung)
        assert t1.decided_by == t2.decided_by
        assert s1[list(t1.order)] == pytest.approx(s2[list(t2.order)])
        mapped = {int(perm[a]): pos for a, pos in positions_not_by_lot(t2).items()}
        assert mapped == positions_not_by_lot(t1)
        if not has_lot(t1):
            assert [int(perm[a]) for a in t2.order] == list(t1.order)
            checked_lot_free += 1
        # Kennzahlen kennen keine Namen
        assert A.cyclic_triads(W2) == A.cyclic_triads(W)
        assert A.tie_share(s2) == pytest.approx(A.tie_share(s1))
        c1, c2 = A.contradiction_share(s1, W), A.contradiction_share(s2, W2)
        assert (math.isnan(c1) and math.isnan(c2)) or c1 == pytest.approx(c2)
    assert checked_lot_free >= 5, "zu wenige Instanzen ohne Los, der strenge Zweig würde kaum geprüft"


# --- 2. Spiegeln ---------------------------------------------------------------------------------------------------------------------------

def test_mirroring_all_results_reverses_the_table_in_a_complete_plan():
    """Vollständiger Plan, jedes Paar k-mal: die Siegezahl w wird zu k(n-1)-w, im Direktvergleich einer Gleichstandsgruppe
    ebenso. Also steht jeder nicht vom Los entschiedene Teilnehmer auf dem gespiegelten Platz."""
    checked = 0
    for idx in range(N_INST):
        n = 2 + idx % 7
        k = 1 + idx % 3
        W, G = random_instance(2000 + idx, n, equal_games=k)
        Wm = W.T.copy()
        assert A.score_vector(Wm, G, "siege") == pytest.approx(k * (n - 1) - A.score_vector(W, G, "siege"))
        t, tm = table_of(W, G), table_of(Wm, G)
        assert tm.decided_by == tuple(reversed(t.decided_by))
        mirrored = {p: n - 1 - pos for p, pos in positions_not_by_lot(t).items()}
        assert mirrored == positions_not_by_lot(tm)
        if not has_lot(t):
            assert tm.order == tuple(reversed(t.order))
            checked += 1
        s, sm = A.score_vector(W, G, "siege"), A.score_vector(Wm, G, "siege")
        assert A.cyclic_triads(Wm) == A.cyclic_triads(W)
        assert A.tie_share(sm) == pytest.approx(A.tie_share(s))
        c, cm = A.contradiction_share(s, W), A.contradiction_share(sm, Wm)
        assert (math.isnan(c) and math.isnan(cm)) or c == pytest.approx(cm)
    assert checked >= 5


def test_mirroring_turns_the_quote_into_one_minus_quote_and_keeps_cycles_in_any_plan():
    """Auch bei Lücken und ungleich vielen Partien: Siegquote q -> 1 - q für jeden mit Partien; zyklische Dreiecke bleiben."""
    seen = 0
    for W, G in instances(3, p_play=0.7):
        has_games = G.sum(axis=1) > 0
        q, qm = A.score_vector(W, G, "quote"), A.score_vector(W.T, G, "quote")
        assert qm[has_games] == pytest.approx(1.0 - q[has_games])
        assert (qm[~has_games] == 0).all()
        assert A.cyclic_triads(W.T) == A.cyclic_triads(W)
        seen += int(has_games.all())
    assert seen >= 3


# --- 3. Zusätzlicher Sieg -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("wertung", ["siege", "quote"])
def test_one_more_win_never_hurts_the_winner_and_never_helps_the_loser(wertung):
    """Ein Ausgang j-gewinnt wird zu i-gewinnt (Partienzahl gleich). Wertung von i +, von j -, alle anderen unverändert; die Zahl der
    Teilnehmer strikt vor i wächst nie; ohne Los in beiden Tabellen steigt i nie ab und j steigt nie auf."""
    strict, compared = 0, 0
    for idx, (W, G) in enumerate(instances(4, count=40, p_play=0.85)):
        n = len(W)
        rng = T.SplitMix64(900 + idx)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j and W[j, i] > 0]
        if not pairs:
            continue
        i, j = pairs[int(rng.uniform(1)[0] * len(pairs))]
        W2 = W.copy()
        W2[i, j] += 1
        W2[j, i] -= 1
        s, s2 = A.score_vector(W, G, wertung), A.score_vector(W2, G, wertung)
        assert s2[i] > s[i] and s2[j] < s[j]
        others = [x for x in range(n) if x not in (i, j)]
        assert s2[others] == pytest.approx(s[others])
        assert (s2 > s2[i]).sum() <= (s > s[i]).sum()
        assert (s2 > s2[j]).sum() >= (s > s[j]).sum()
        t, t2 = table_of(W, G, wertung), table_of(W2, G, wertung)
        if not has_lot(t) and not has_lot(t2):
            compared += 1
            assert t2.rank[i] <= t.rank[i]
            assert t2.rank[j] >= t.rank[j]
            strict += int(t2.rank[i] < t.rank[i])
    assert compared >= 12 and strict >= 1


# --- 4. Wiederholung -----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("wertung", ["siege", "quote"])
def test_repeating_every_game_m_times_changes_nothing(wertung):
    """W und G mit m multipliziert: Wertungsordnung, Direktvergleich, Gleichstandsgruppen und damit (gleicher Seed) die ganze
    Tabelle sowie alle Anteilskennzahlen bleiben gleich."""
    for idx, (W, G) in enumerate(instances(5, p_play=0.8)):
        m = 2 + idx % 3
        t1, t2 = table_of(W, G, wertung), table_of(W * m, G * m, wertung)
        assert t1 == t2
        s1, s2 = A.score_vector(W, G, wertung), A.score_vector(W * m, G * m, wertung)
        assert A.tie_share(s1) == pytest.approx(A.tie_share(s2))
        assert A.cyclic_triads(W * m) == A.cyclic_triads(W)
        c1, c2 = A.contradiction_share(s1, W), A.contradiction_share(s2, W * m)
        assert (math.isnan(c1) and math.isnan(c2)) or c1 == pytest.approx(c2)


# --- 5. Transitive Turniere -----------------------------------------------------------------------------------------------------------

def test_transitive_tournament_gives_exactly_the_underlying_order():
    for idx in range(N_INST):
        n = 1 + idx % 9
        k = 1 + idx % 3
        sigma = T.SplitMix64(30 + idx).permutation(n)       # sigma[0] schlägt alle, sigma[1] alle außer sigma[0] ...
        W = np.zeros((n, n), dtype=int)
        for a in range(n):
            for b in range(a + 1, n):
                W[sigma[a], sigma[b]] = k
        G = W + W.T
        for wertung in ("siege", "quote"):
            tab = table_of(W, G, wertung)
            assert tab.order == tuple(int(x) for x in sigma)
            assert set(tab.decided_by) <= {"Wertung"}
            s = A.score_vector(W, G, wertung)
            strength = np.empty(n)
            strength[sigma] = np.arange(n, 0, -1)
            assert math.isnan(A.kendall_tau(tab.rank, strength)) if n == 1 else A.kendall_tau(tab.rank, strength) == pytest.approx(1.0)
            if n > 1:
                assert A.tie_share(s) == 0.0       # n = 1 hat kein Paar: Anteil nicht definiert
        assert A.cyclic_triads(W) == 0
        c = A.contradiction_share(A.score_vector(W, G, "siege"), W)
        assert math.isnan(c) or c == 0.0


# --- 6. Los -------------------------------------------------------------------------------------------------------------------------------

def test_lot_only_permutes_inside_ties_and_is_reproducible():
    """Andere Lose (anderer Seed) ändern nur die Reihenfolge innerhalb von Gruppen mit gleicher Wertung und gleichem Direktvergleich;
    die Wertung entlang der Tabelle fällt nie; gleicher Seed gibt dieselbe Tabelle."""
    differing = 0
    for idx, (W, G) in enumerate(instances(6, count=40, p_play=0.9)):
        s = A.score_vector(W, G, "siege")
        ta, tb, tc = table_of(W, G, seed=11), table_of(W, G, seed=12), table_of(W, G, seed=11)
        assert ta == tc
        for tab in (ta, tb):
            along = s[list(tab.order)]
            assert (np.diff(along) <= 0).all()
            assert sorted(tab.order) == list(range(len(W)))
            assert all(tab.rank[p] == pos for pos, p in enumerate(tab.order))
        assert ta.decided_by == tb.decided_by
        assert positions_not_by_lot(ta) == positions_not_by_lot(tb)
        assert lot_runs(ta, s) == lot_runs(tb, s)
        differing += int(ta.order != tb.order)
    assert differing >= 3, "das Los hätte in einigen Instanzen etwas ändern müssen"


def test_head_to_head_decides_before_the_lot(mini_headtohead):
    """Punktgleiche mit eindeutigem Direktvergleich bekommen nie das Los, für keinen Seed (Mini-Instanz der conftest: Siege 2,2,1,1,
    Direktvergleich 0>1 und 2>3, also Reihenfolge 0,1,2,3)."""
    W, G = mini_headtohead
    for seed in range(20):
        tab = table_of(W, G, seed=seed)
        assert tab.order == (0, 1, 2, 3)
        assert tab.decided_by == ("Direktvergleich",) * 4


# --- 7. Kendall-τ --------------------------------------------------------------------------------------------------------------------------

def test_kendall_tau_invariances_and_sign_flips():
    for idx in range(N_INST):
        n = 2 + idx % 8
        rng = T.SplitMix64(7000 + idx)
        rank = rng.permutation(n)                                       # Platz 0..n-1
        values = np.floor(rng.uniform(n) * 4)                           # viele Gleichstände in der Größe
        if len(set(values)) == 1:
            values[0] += 1
        tau = A.kendall_tau(rank, values)
        assert -1.0 <= tau <= 1.0
        assert A.kendall_tau(rank, 3.0 * values + 7.5) == pytest.approx(tau)           # monoton wachsend
        assert A.kendall_tau(rank, np.exp(values)) == pytest.approx(tau)
        assert A.kendall_tau(rank, -values) == pytest.approx(-tau)                     # Größe negiert
        assert A.kendall_tau(n - 1 - rank, values) == pytest.approx(-tau)              # Tabelle umgekehrt
        perm = rng.permutation(n)
        assert A.kendall_tau(rank[perm], values[perm]) == pytest.approx(tau)           # gemeinsam umnummeriert
        by_value = np.argsort(-values, kind="stable")                                  # Tabelle exakt nach Größe
        rank_sorted = np.empty(n, dtype=int)
        rank_sorted[by_value] = np.arange(n)
        assert A.kendall_tau(rank_sorted, values) == pytest.approx(1.0)


# --- 8. Randfälle -------------------------------------------------------------------------------------------------------------------------

def test_single_participant():
    W = np.zeros((1, 1), dtype=int)
    tab = table_of(W, W)
    assert tab.order == (0,) and tab.rank == (0,) and tab.decided_by == ("Wertung",)
    assert A.cyclic_triads(W) == 0
    assert math.isnan(A.contradiction_share(A.score_vector(W, W, "siege"), W))


def test_two_participants_with_and_without_a_game():
    W = np.array([[0, 2], [1, 0]])
    G = W + W.T
    tab = table_of(W, G)
    assert tab.order == (0, 1) and tab.decided_by == ("Wertung", "Wertung")
    assert table_of(W.T, G).order == (1, 0)
    # keine Partie: beide Wertung 0, Direktvergleich 0:0 -> Los, für jeden Seed eine gültige Reihenfolge
    Z = np.zeros((2, 2), dtype=int)
    seen = set()
    for seed in range(20):
        tz = table_of(Z, Z, seed=seed)
        assert tz.decided_by == ("Los", "Los") and sorted(tz.order) == [0, 1]
        seen.add(tz.order)
    assert seen == {(0, 1), (1, 0)}
    assert A.tie_share(A.score_vector(Z, Z, "siege")) == 1.0


def test_tournament_without_any_game_is_a_pure_lot():
    n = 6
    Z = np.zeros((n, n), dtype=int)
    tab = table_of(Z, Z, "quote", seed=3)
    assert tab.decided_by == ("Los",) * n and sorted(tab.order) == list(range(n))
    assert A.cyclic_triads(Z) == 0 and A.tie_share(np.zeros(n)) == 1.0
    assert math.isnan(A.kendall_tau(tab.rank, np.zeros(n)))
    assert table_of(Z, Z, "quote", seed=3) == tab


def test_dominant_player_always_leads_and_winless_player_trails_in_a_complete_plan():
    for idx in range(N_INST):
        n = 3 + idx % 6
        W, G = random_instance(8000 + idx, n, equal_games=2)
        top = idx % n
        for j in range(n):
            if j != top:
                W[top, j], W[j, top] = 2, 0              # top gewinnt alles
        tab = table_of(W, G)
        assert tab.order[0] == top and tab.decided_by[0] == "Wertung"
        assert A.score_vector(W, G, "siege")[top] == 2 * (n - 1)
