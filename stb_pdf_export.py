"""PDF-Export der aktuellen Konfiguration und Tabelle (fpdf2, Kernschrift: nur Zeichen aus Latin-1, daher „Tau“ statt τ)."""

from fpdf import FPDF

import stb_constants as C
import stb_evaluation as ev


def _latin(text: str) -> str:
    """Ersetzt Zeichen, die die Kernschrift nicht kennt (Gedankenstrich, τ, Pfeile, ≥)."""
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("≥", ">="), ("≤", "<="), ("→", "->"), ("**", ""), ("€", "EUR")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s, t, table = a.settings, a.turnier, a.table
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Tabelle nach Siegen: sie misst das Feld, nicht die Stärke"))
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    plan = C.PLAN_LABELS[s.plan]
    pdf.multi_cell(0, 5.5, _latin(
        f"Teilnehmer: {s.n}, Stärkeabstand: {ev.de(s.delta)}, Partien je Paar: {s.k}, Spielplan: {plan}, Zyklenstärke: {ev.de(s.c)}, "
        f"Stilverteilung: {C.STIL_LABELS[s.stil]}, Wertung: {C.WERTUNG_LABELS[s.wertung]}, Seed: {s.seed}."))
    pdf.ln(2)
    level, text = ev.verdict_core(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text))
    pdf.ln(2)
    pdf.multi_cell(0, 5.5, _latin(
        f"Tau gegen die Generator-Stärke: {ev.de(a.tau_s)} | Tau gegen die erwartete Siegquote: {ev.de(a.tau_e)} | Platz des Stärksten: {a.best_place} | "
        f"Gleichstandsanteil: {ev.pct(a.tie_share)} % | zyklische Dreiecke: {a.cyclic} von {a.triples} | Komponenten: {a.components}."))
    pdf.ln(3)
    headers = ("Platz", "Teilnehmer", "Siege", "Partien", "Quote", "Stärke s", "erw. Siegquote", "entschieden durch")
    widths = (14, 24, 16, 18, 18, 22, 32, 40)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, _latin(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    wins = t.wins.sum(axis=1)
    games = t.games.sum(axis=1)
    for position, i in enumerate(table.order):
        quote = wins[i] / games[i] if games[i] else 0.0
        row = (str(position + 1), t.names[i], str(int(wins[i])), str(int(games[i])), ev.de(quote), ev.de(t.strengths[i]),
               ev.de(t.expected[i]), table.decided_by[position])
        for value, w in zip(row, widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alle Ergebnisse sind simuliert (Bradley-Terry-Stärke, optional zyklische Störung über drei Spielstile). Die Generator-Stärke ist eine "
        "Konstruktion der Simulation. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."))
    return bytes(pdf.output())
