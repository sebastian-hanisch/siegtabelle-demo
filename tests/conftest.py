"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _wins(n, results):
    """`results`: Liste (Sieger, Verlierer), je eine Partie; liefert (W, G)."""
    W = np.zeros((n, n), dtype=int)
    for a, b in results:
        W[a, b] += 1
    return W, W + W.T


@pytest.fixture
def mini_cycle():
    """4 Teilnehmer A=0, B=1, C=2, D=3, je eine Partie gegen jeden. A schlägt B, B schlägt C, C schlägt A; alle drei schlagen D.
    Handrechnung: Siege (2, 2, 2, 0); Dreier-Gleichstand, im Direktvergleich je ein Sieg, also entscheidet das Los; D ist Letzter;
    zyklische Dreiecke = C(4,3) - 3*C(2,2) - C(0,2) = 1; Gleichstandsanteil 3 von 6 Paaren; Widerspruchsanteil 0; eine Komponente."""
    return _wins(4, [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)])


@pytest.fixture
def mini_headtohead():
    """4 Teilnehmer, je eine Partie gegen jeden: 0>1, 0>2, 3>0, 1>2, 1>3, 2>3.
    Handrechnung: Siege (2, 2, 1, 1); die Gleichstände (0,1) und (2,3) entscheidet der Direktvergleich: Reihenfolge 0, 1, 2, 3;
    zyklische Dreiecke: (0,1,3) und (0,2,3), also 2 (Formel: 4 - C(2,2) - C(2,2) - C(1,2) - C(1,2) = 2)."""
    return _wins(4, [(0, 1), (0, 2), (3, 0), (1, 2), (1, 3), (2, 3)])


@pytest.fixture
def mini_contradiction():
    """4 Teilnehmer, je eine Partie gegen jeden: 1>0, 0>2, 0>3, 2>1, 3>1, 2>3.
    Handrechnung: Siege (2, 1, 2, 1). Paare mit verschiedener Wertung: (0,1), (0,3), (1,2), (2,3). Nur im Paar (0,1) hat der Hintere gewonnen
    (1 schlug 0): Widerspruchsanteil 1/4."""
    return _wins(4, [(1, 0), (0, 2), (0, 3), (2, 1), (3, 1), (2, 3)])
