"""Szenario: Reglerwerte (Settings) und das daraus erzeugte Turnier. Die Erzeugung selbst steckt im gemeinsamen turnier_generator.py."""

from dataclasses import dataclass

import stb_constants as C
import turnier_generator as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    k: int = C.DEFAULT_K
    plan: str = "vollstaendig"
    x: int = C.DEFAULT_X
    p: float = C.DEFAULT_P
    c: float = C.DEFAULT_CYCLE
    stil: str = "ausgewogen"
    wertung: str = "siege"
    seed: int = C.DEFAULT_SEED


def make_turnier(s: Settings) -> T.Turnier:
    """Deterministisch aus den Reglerwerten: gleiche Eingabe, gleiches Turnier."""
    return T.build_turnier(s.n, s.delta, s.k, s.plan, s.x, s.p, s.c, s.stil, s.seed)
