"""Kern der Demo (nur numpy/Standardbibliothek): Tabelle nach Siegen, Gleichstandsregel und die Kennzahlen, die sie bewerten.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); `rank_table` ruft nur der Reihe nach auf.
Zufall nur über einen übergebenen `SplitMix64`. Rang 0 = bester Platz."""

import math
from dataclasses import dataclass

import numpy as np

import turnier_generator as T


@dataclass(frozen=True)
class Table:
    order: tuple          # Teilnehmerindizes, bester zuerst
    rank: tuple           # rank[i] = Platz von Teilnehmer i (0 = bester)
    decided_by: tuple     # je Platz in `order`: "Wertung", "Direktvergleich" oder "Los"


def wins_vector(W) -> np.ndarray:
    return np.asarray(W).sum(axis=1)


def games_vector(G) -> np.ndarray:
    return np.asarray(G).sum(axis=1)


def score_vector(W, G, wertung: str) -> np.ndarray:
    """Wertung je Teilnehmer: Zahl der Siege oder Siegquote (Siege je Partie; ohne Partien 0)."""
    wins = wins_vector(W).astype(float)
    if wertung == "siege":
        return wins
    if wertung == "quote":
        games = games_vector(G).astype(float)
        return np.divide(wins, games, out=np.zeros_like(wins), where=games > 0)
    raise ValueError(f"unbekannte Wertung: {wertung}")


def head_to_head_wins(W, group) -> np.ndarray:
    """Siege innerhalb der Gruppe: die Mini-Tabelle des Direktvergleichs."""
    group = list(group)
    return np.asarray(W)[np.ix_(group, group)].sum(axis=1)


def split_ties(values, tol: float = 0.0):
    """Gruppen gleicher Werte, absteigend. Rückgabe: Liste von Index-Listen (Positionen in `values`)."""
    v = np.asarray(values, dtype=float)
    idx = sorted(range(len(v)), key=lambda i: (-v[i], i))
    groups, current = [], [idx[0]] if len(idx) else []
    for i in idx[1:]:
        if abs(v[i] - v[current[0]]) <= tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    if current:
        groups.append(current)
    return groups


def rank_table(score, W, rng: T.SplitMix64) -> Table:
    """Tabelle: nach Wertung absteigend; Gleichstand erst durch den Direktvergleich innerhalb der Gruppe, danach durch das Los."""
    n = len(score)
    order, decided = [], []
    for group in split_ties(score):
        if len(group) == 1:
            order.append(group[0])
            decided.append("Wertung")
            continue
        mini = head_to_head_wins(W, group)
        for sub in split_ties(mini):
            members = [group[i] for i in sub]
            if len(members) == 1:
                order.append(members[0])
                decided.append("Direktvergleich")
                continue
            lots = rng.uniform(len(members))
            for j in np.argsort(lots, kind="stable"):
                order.append(members[int(j)])
                decided.append("Los")
    rank = [0] * n
    for position, i in enumerate(order):
        rank[i] = position
    return Table(tuple(order), tuple(rank), tuple(decided))


def kendall_tau(rank, values) -> float:
    """τ zwischen Tabelle (kleinerer Rang = besser) und einer Größe `values` (größer = besser). Nur Paare mit verschiedenem Wert zählen."""
    r = np.asarray(rank, dtype=float)
    v = np.asarray(values, dtype=float)
    iu = np.triu_indices(len(r), 1)
    sv = np.sign(v[iu[0]] - v[iu[1]])        # +1: i hat den größeren Wert
    sr = np.sign(r[iu[1]] - r[iu[0]])        # +1: i steht in der Tabelle vor j
    counted = sv != 0
    if not counted.any():
        return float("nan")
    return float((sv[counted] * sr[counted]).mean())


def contradiction_share(score, W) -> float:
    """Anteil der gespielten, im direkten Vergleich nicht ausgeglichenen Paare mit verschiedener Wertung, in denen der in der Wertung
    Hintere den direkten Vergleich gewonnen hat. Ohne zählbares Paar: nan."""
    score = np.asarray(score, dtype=float)
    W = np.asarray(W)
    iu = np.triu_indices(len(score), 1)
    wi, wj = W[iu], W.T[iu]
    counted = (wi != wj) & (score[iu[0]] != score[iu[1]])
    if not counted.any():
        return float("nan")
    i_higher = score[iu[0]] > score[iu[1]]
    i_won = wi > wj
    return float((i_higher != i_won)[counted].mean())


def tie_share(score) -> float:
    """Anteil aller Paare (i < j) mit gleicher Wertung."""
    s = np.asarray(score, dtype=float)
    iu = np.triu_indices(len(s), 1)
    return float((s[iu[0]] == s[iu[1]]).mean())


def cyclic_triads(W) -> int:
    """Zyklische Dreiecke im Mehrheitsturnier (Kante i -> j, wenn i öfter gegen j gewonnen hat; Gleichstand und ungespielt = keine Kante):
    Spur der dritten Potenz der Kantenmatrix durch 3."""
    W = np.asarray(W)
    B = (W > W.T).astype(np.int64)
    return int(np.trace(B @ B @ B) // 3)


def cyclic_triads_formula(wins) -> int:
    """Gilt für ein vollständiges Turnier ohne Gleichstände: C(n,3) minus die Summe der C(w_i,2). Zähl-Argument: Jedes nicht zyklische
    Dreieck hat genau einen Teilnehmer mit zwei Siegen darin. Fundstelle der Formel vor jeder Textverwendung prüfen."""
    w = [int(x) for x in wins]
    n = len(w)
    return math.comb(n, 3) - sum(math.comb(x, 2) for x in w)


def crossing_errors(rank, n_weak: int) -> float:
    """Anteil der Paare (stark, schwach) aus zwei Staffeln, in denen der Schwache in der Tabelle vor dem Starken steht.
    Staffel = Indexhälfte: Indizes < n_weak schwach, alle anderen stark."""
    r = np.asarray(rank)
    weak, strong = r[:n_weak], r[n_weak:]
    if len(weak) == 0 or len(strong) == 0:
        return float("nan")
    return float((weak[:, None] < strong[None, :]).mean())
