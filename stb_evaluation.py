"""Kennzahlen, Verdikt-Logik und Experimente (Sweeps über die festen Instanzen SWEEP_SEEDS)."""

import math
from dataclasses import dataclass

import numpy as np

import stb_algorithm as A
import stb_constants as C
import turnier_generator as T
from stb_scenario import Settings, make_turnier

_TABLE_SALT = 0xA5A5A5A5


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    turnier: T.Turnier
    table: A.Table
    score: np.ndarray
    tau_s: float            # Tabelle gegen die Generator-Stärke
    tau_e: float            # Tabelle gegen die erwartete Siegquote
    best_place: int         # Tabellenplatz (1 = Spitze) des nach Generator-Stärke Besten
    top_is_best_s: bool     # steht der nach Generator-Stärke Beste auf Platz 1?
    top_is_best_e: bool     # steht der nach erwarteter Siegquote Beste auf Platz 1?
    tie_share: float        # Anteil der Paare mit gleicher Wertung
    contradiction: float    # Anteil der Paare, in denen der Hintere den direkten Vergleich gewann (nan: nichts zählbar)
    cyclic: int             # zyklische Dreiecke im Mehrheitsturnier
    triples: int            # Zahl aller Dreiergruppen
    components: int
    los_places: int         # Plätze, die nur das Los entscheidet
    games_total: int
    games_min: int
    games_max: int


def analyse(settings: Settings) -> Analysis:
    t = make_turnier(settings)
    score = A.score_vector(t.wins, t.games, settings.wertung)
    table = A.rank_table(score, t.wins, T.SplitMix64(settings.seed ^ _TABLE_SALT))
    n = t.n
    games = A.games_vector(t.games)
    return Analysis(
        settings=settings,
        turnier=t,
        table=table,
        score=score,
        tau_s=A.kendall_tau(table.rank, t.strengths),
        tau_e=A.kendall_tau(table.rank, t.expected),
        best_place=table.rank[n - 1] + 1,
        top_is_best_s=table.order[0] == n - 1,
        top_is_best_e=table.order[0] == int(np.argmax(t.expected)),
        tie_share=A.tie_share(score),
        contradiction=A.contradiction_share(score, t.wins),
        cyclic=A.cyclic_triads(t.wins),
        triples=math.comb(n, 3),
        components=T.components(t.games),
        los_places=sum(1 for d in table.decided_by if d == "Los"),
        games_total=int(t.games.sum() // 2),
        games_min=int(games.min()),
        games_max=int(games.max()),
    )


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else float("nan")


def _over_seeds(make_settings, seeds=C.SWEEP_SEEDS):
    return [analyse(make_settings(seed)) for seed in seeds]


# --- Experiment 1: die Tabelle im besten Fall (vollständig, ausgeglichen) ----------------------------------------------------------------

def sweep_best_case(ks=C.K_SWEEP, n=C.EXP_N_BEST_CASE, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for k in ks:
        runs = _over_seeds(lambda s: Settings(n=n, delta=delta, k=k, plan="vollstaendig", seed=s), seeds)
        rows.append({"k": k, "tau": _mean([r.tau_s for r in runs]), "top": _mean([float(r.top_is_best_s) for r in runs]),
                     "ties": _mean([r.tie_share for r in runs]), "contradiction": _mean([r.contradiction for r in runs]),
                     "los": _mean([float(r.los_places) for r in runs])})
    return rows


# --- Experiment 2: zwei Staffeln ---------------------------------------------------------------------------------------------------------

def sweep_staffeln(xs=C.X_SWEEP, n=C.EXP_N, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for x in xs:
        runs = _over_seeds(lambda s: Settings(n=n, delta=delta, k=1, plan="staffeln", x=x, seed=s), seeds)
        rows.append({"x": x, "wrong": _mean([A.crossing_errors(r.table.rank, n // 2) for r in runs]),
                     "components": _mean([float(r.components) for r in runs]), "tau": _mean([r.tau_s for r in runs]),
                     "disconnected": _mean([float(r.components > 1) for r in runs])})
    return rows


# --- Experiment 3: Lücken im Spielplan ---------------------------------------------------------------------------------------------------

def sweep_luecken(ps=C.P_SWEEP, n=C.EXP_N, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for p in ps:
        by_siege = _over_seeds(lambda s: Settings(n=n, delta=delta, k=1, plan="luecken", p=p, wertung="siege", seed=s), seeds)
        by_quote = _over_seeds(lambda s: Settings(n=n, delta=delta, k=1, plan="luecken", p=p, wertung="quote", seed=s), seeds)
        rows.append({"p": p, "tau_siege": _mean([r.tau_s for r in by_siege]), "tau_quote": _mean([r.tau_s for r in by_quote]),
                     "disconnected": _mean([float(r.components > 1) for r in by_siege]),
                     "contradiction": _mean([r.contradiction for r in by_siege]),
                     "games_min": _mean([float(r.games_min) for r in by_siege])})
    return rows


# --- Experiment 4: die Leistung hängt am Gegner (Zyklen, Stilverteilung) ---------------------------------------------------------------

def sweep_zyklen(cs=C.C_SWEEP, stile=T.STILE, n=C.EXP_N, delta=C.EXP_DELTA, k=C.K_CYCLES, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for stil in stile:
        for c in cs:
            runs = _over_seeds(lambda s: Settings(n=n, delta=delta, k=k, plan="vollstaendig", c=c, stil=stil, seed=s), seeds)
            rows.append({"stil": stil, "c": c, "tau_s": _mean([r.tau_s for r in runs]), "tau_e": _mean([r.tau_e for r in runs]),
                         "best_place": _mean([float(r.best_place) for r in runs]),
                         "top_s": _mean([float(r.top_is_best_s) for r in runs]), "top_e": _mean([float(r.top_is_best_e) for r in runs]),
                         "contradiction": _mean([r.contradiction for r in runs]), "cyclic": _mean([float(r.cyclic) for r in runs]),
                         "triples": runs[0].triples})
    return rows


# --- Verdikte (live aus den Messwerten, keine festen Behauptungen) -------------------------------------------------------------------------

def de(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def verdict_core(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Primäransicht: Trifft die Tabelle den Stärksten, und welche Frage beantwortet sie?"""
    n = a.turnier.n
    if a.components > 1:
        return "warning", (f"Der Spielplan zerfällt in **{a.components} Teile**, die nie gegeneinander gespielt haben. Die Tabelle weist trotzdem Plätze "
                           f"1 bis {n} aus, aber über die Teile hinweg ist keine Reihenfolge belegt (τ gegen die Stärke: {de(a.tau_s)}).")
    if a.top_is_best_s:
        text = f"Der Stärkste steht auf **Platz 1**; τ gegen die Generator-Stärke {de(a.tau_s)}, gegen die erwartete Siegquote {de(a.tau_e)}."
        if a.tie_share > 0:
            text += f" {pct(a.tie_share)} % aller Paare stehen punktgleich, {a.los_places} Plätze entscheidet nur das Los."
        return "success", text
    text = (f"Der Stärkste steht auf **Platz {a.best_place}** von {n}; τ gegen die Generator-Stärke {de(a.tau_s)}, "
            f"gegen die erwartete Siegquote {de(a.tau_e)}.")
    if a.tau_e - a.tau_s > 0.1:
        text += " Die Tabelle ordnet die erwartete Siegquote deutlich besser als die Stärke: Sie misst den Erfolg gegen **dieses** Feld."
    return "warning", text


def _row(rows, **match):
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            return r
    return None


def verdict_best_case(rows) -> tuple[str, str]:
    lo, hi = rows[0], rows[-1]
    return "info", (f"Bei vollständigem, ausgeglichenem Plan steht der Stärkste mit {lo['k']} Partie je Paar in nur {pct(lo['top'])} % der Läufe auf "
                    f"Platz 1 (τ = {de(lo['tau'])}), mit {hi['k']} Partien in {pct(hi['top'])} % (τ = {de(hi['tau'])}). Das ist Rauschen und kein "
                    "Verfahrensfehler: Mehr Partien beheben es, und die Ordnung nach Siegen ist hier dieselbe, die ein Bradley-Terry-Modell ergäbe (wo dessen Schätzer existiert).")


def verdict_staffeln(rows) -> tuple[str, str]:
    r0, r1, rl = _row(rows, x=0), _row(rows, x=1), rows[-1]
    return "warning", (f"Ohne Kreuzpartien ordnet die Tabelle **{pct(r0['wrong'])} %** der Paare über die Staffeln falsch: ein Münzwurf, und sie weist "
                       f"trotzdem lückenlos Plätze aus. Mit einer Kreuzpartie je starkem Spieler sind es {pct(r1['wrong'])} %, "
                       f"wenn alle gegeneinander spielen {pct(rl['wrong'])} %.")


def verdict_luecken(rows) -> tuple[str, str]:
    half, third, fifth = _row(rows, p=0.5), _row(rows, p=0.33), _row(rows, p=0.2)
    return "warning", (f"Gewertet nach Siegquote ordnet die Tabelle bei p = 0,5 besser als nach Siegen (τ {de(half['tau_quote'])} gegen "
                       f"{de(half['tau_siege'])}), weil die Partienzahlen auseinanderlaufen. Bei p = 0,33 zerfällt der Plan in {pct(third['disconnected'])} % "
                       f"der Läufe in Teile ohne Vergleich (bei p = 0,2 in {pct(fifth['disconnected'])} %). Der Widerspruch zum direkten Vergleich "
                       f"sinkt dabei von {pct(rows[0]['contradiction'])} % auf {pct(rows[-1]['contradiction'])} %: Er verrät die Lücken nicht.")


def verdict_zyklen(rows) -> tuple[str, str]:
    k4 = _row(rows, stil="konter", c=0.4)
    a4 = _row(rows, stil="ausgewogen", c=0.4)
    return "warning", (f"Bei Konter-Stil und c = 0,4 steht der Stärkste im Mittel auf Platz {de(k4['best_place'], 1)}; τ gegen die Generator-Stärke "
                       f"{de(k4['tau_s'])}, gegen die erwartete Siegquote {de(k4['tau_e'])}. Der Widerspruch zum direkten Vergleich beträgt nur "
                       f"{pct(k4['contradiction'])} %: Die Tabelle ist in sich stimmig und beantwortet doch eine andere Frage als die nach der "
                       f"Stärke. Bei ausgewogenen Stilen bleibt die Ordnung (τ {de(a4['tau_s'])}), obwohl {de(a4['cyclic'], 0)} von "
                       f"{a4['triples']} Dreiergruppen zyklisch sind.")
