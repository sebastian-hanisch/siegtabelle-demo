"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import stb_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_matrix(a):
    """Siegmatrix in Tabellenreihenfolge: Zelle (Zeile i, Spalte j) = Siegquote von i gegen j; ungespielte Paare bleiben leer."""
    t, order = a.turnier, list(a.table.order)
    names = [t.names[i] for i in order]
    W, G = t.wins[np.ix_(order, order)], t.games[np.ix_(order, order)]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(G > 0, W / np.where(G > 0, G, 1), np.nan)
    text = [[f"{W[r, c]}/{G[r, c]}" if G[r, c] > 0 else "" for c in range(len(order))] for r in range(len(order))]
    fig = go.Figure(go.Heatmap(z=share, x=names, y=names, text=text, texttemplate="%{text}", zmin=0, zmax=1, colorscale="RdBu", showscale=False,
                               hovertemplate="%{y} gegen %{x}: %{text} Siege<extra></extra>", xgap=1, ygap=1))
    fig.update_xaxes(title_text="Gegner", side="top", type="category")
    fig.update_yaxes(title_text="Teilnehmer (Tabellenplatz von oben)", autorange="reversed", type="category")
    return _base(fig, height=max(320, 34 * len(order) + 90))


def build_place_scatter(a):
    """Tabellenplatz (x) gegen den Platz nach Generator-Stärke (y); auf der Diagonale stimmt die Tabelle."""
    t, n = a.turnier, a.turnier.n
    place_true = (np.argsort(np.argsort(-t.strengths)) + 1)          # 1 = stärkster
    place_table = np.array(a.table.rank) + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, n], y=[1, n], mode="lines", line=dict(color=C.COLOR_OTHER, dash="dash"), hoverinfo="skip", name="Tabelle = Stärke"))
    fig.add_trace(go.Scatter(x=place_table, y=place_true, mode="markers+text", text=list(t.names), textposition="top center",
                             marker=dict(size=14, color=C.COLOR_MAIN), name="Teilnehmer",
                             hovertemplate="%{text}: Platz %{x} in der Tabelle, Platz %{y} nach Stärke<extra></extra>"))
    fig.update_xaxes(title_text="Platz in der Tabelle", range=[0.5, n + 0.5], dtick=1)
    fig.update_yaxes(title_text="Platz nach Generator-Stärke", range=[n + 0.5, 0.5], dtick=1)
    return _base(fig, height=380)


def build_best_case(rows):
    xs = [str(r["k"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["tau"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LINE, width=3), name="τ gegen die Stärke",
                             hovertemplate="%{x} Partien je Paar: τ = %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["top"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=3), name="Platz 1 ist der Stärkste",
                             hovertemplate="%{x} Partien je Paar: %{y:.0%}<extra></extra>"))
    fig.update_xaxes(title_text="Partien je Paar", type="category")
    fig.update_yaxes(title_text="τ bzw. Anteil der Läufe", range=[0, 1.02])
    return _base(fig)


def build_staffeln(rows):
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure(go.Bar(x=xs, y=[r["wrong"] for r in rows], marker_color=C.COLOR_MAIN,
                           hovertemplate="%{x} Kreuzpartien: %{y:.0%} der Paare falsch<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dash", line_color=C.COLOR_WARN, annotation_text="Münzwurf", annotation_position="top right")
    fig.update_xaxes(title_text="Kreuzpartien je Spieler der starken Staffel", type="category")
    fig.update_yaxes(title_text="falsch geordnete Paare über die Staffeln", range=[0, 0.6], tickformat=".0%")
    return _base(fig)


def build_luecken(rows):
    xs = [f"{r['p']:.2f}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["tau_siege"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LINE, width=3), name="τ, Wertung: Siege"))
    fig.add_trace(go.Scatter(x=xs, y=[r["tau_quote"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=3), name="τ, Wertung: Siegquote"))
    fig.add_trace(go.Scatter(x=xs, y=[r["disconnected"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_WARN, width=2, dash="dot"),
                             name="Plan zerfällt in Teile"))
    fig.update_xaxes(title_text="Wahrscheinlichkeit, dass ein Paar spielt", type="category")
    fig.update_yaxes(title_text="τ bzw. Anteil der Läufe", range=[0, 1.02])
    return _base(fig)


def build_zyklen(rows):
    labels = [f"{r['stil']}, c = {r['c']:.1f}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["tau_s"] for r in rows], marker_color=C.COLOR_LINE, name="τ gegen die Generator-Stärke"))
    fig.add_trace(go.Bar(x=labels, y=[r["tau_e"] for r in rows], marker_color=C.COLOR_MAIN, name="τ gegen die erwartete Siegquote"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Stilverteilung und Zyklenstärke", type="category")
    fig.update_yaxes(title_text="τ", range=[0, 1.02])
    return _base(fig, height=400)
