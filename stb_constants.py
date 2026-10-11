"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 4, 16, 8
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.05, 0.3
K_MIN, K_MAX, DEFAULT_K = 1, 10, 1
X_MIN, X_MAX, DEFAULT_X = 0, 8, 1                # Kreuzpartien je Spieler der starken Staffel (nur Plan "staffeln")
P_MIN, P_MAX, P_STEP, DEFAULT_P = 0.1, 1.0, 0.05, 0.5   # Spielwahrscheinlichkeit je Paar (nur Plan "luecken")
CYCLE_MIN, CYCLE_MAX, CYCLE_STEP, DEFAULT_CYCLE = 0.0, 0.5, 0.05, 0.0
SEED_MAX, DEFAULT_SEED = 999999, 35

PLAN_LABELS = {"vollstaendig": "Vollständig: jeder gegen jeden", "staffeln": "Zwei Staffeln", "luecken": "Zufällige Lücken"}
STIL_LABELS = {
    "ausgewogen": "Ausgewogen (reihum)",
    "konter": "Konter-Stil dominiert (der Stärkste wird gekontert)",
    "beute": "Beute-Stil dominiert (der Stärkste kontert)",
}
WERTUNG_LABELS = {"siege": "Zahl der Siege", "quote": "Siegquote (Siege je Partie)"}

# Feste Instanzen für alle Experimente: Median bzw. Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100200))
EXP_N_BEST_CASE = 8      # Experiment 1 (Standardinstanz)
EXP_N = 12               # Experimente 2 bis 4: feste Instanz mit 12 Teilnehmern, unabhängig vom Regler
EXP_DELTA = 0.3
K_SWEEP = (1, 2, 4, 10, 40)
X_SWEEP = (0, 1, 2, 3, 6)
P_SWEEP = (1.0, 0.75, 0.5, 0.33, 0.2, 0.12)
C_SWEEP = (0.0, 0.2, 0.4)
K_CYCLES = 50            # viele Partien je Paar, damit das Rauschen verschwindet und nur die Zyklen wirken

COLOR_MAIN = "#2F6B65"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"
COLOR_WARN = "#c0392b"

PRESETS = {
    "Ausgeglichen (Voreinstellung)": {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "k": 1, "plan": "vollstaendig", "x": DEFAULT_X, "p": DEFAULT_P,
                                      "c": 0.0, "stil": "ausgewogen", "wertung": "siege", "seed": DEFAULT_SEED},
    "Zwei Staffeln": {"n": 12, "delta": DEFAULT_DELTA, "k": 1, "plan": "staffeln", "x": 1, "p": DEFAULT_P,
                      "c": 0.0, "stil": "ausgewogen", "wertung": "siege", "seed": DEFAULT_SEED},
    "Lückenhafter Plan": {"n": 12, "delta": DEFAULT_DELTA, "k": 1, "plan": "luecken", "x": DEFAULT_X, "p": 0.33,
                          "c": 0.0, "stil": "ausgewogen", "wertung": "quote", "seed": DEFAULT_SEED},
    "Konter-Stil": {"n": 12, "delta": DEFAULT_DELTA, "k": 10, "plan": "vollstaendig", "x": DEFAULT_X, "p": DEFAULT_P,
                    "c": 0.3, "stil": "konter", "wertung": "siege", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Ausgeglichen (Voreinstellung)": "8 Teilnehmer, jeder spielt einmal gegen jeden, keine Zyklen: die Tabelle im besten Fall. Platz 1 stimmt trotzdem oft nicht.",
    "Zwei Staffeln": "12 Teilnehmer in einer starken und einer schwachen Staffel, dazu nur eine Kreuzpartie je Spieler der starken Staffel.",
    "Lückenhafter Plan": "12 Teilnehmer, jedes Paar spielt nur mit Wahrscheinlichkeit ein Drittel, gewertet nach Siegquote.",
    "Konter-Stil": "12 Teilnehmer, 10 Partien je Paar, drei Spielstile mit Zyklen: der Stärkste hat einen Stil, den die meisten kontern.",
}
