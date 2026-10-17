"""Tabelle nach Siegen: sie misst das Feld, nicht die Stärke - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der Ranking-Linie: Ein neutrales Turnier mit bekannter Stärke zeigt, wo die Tabelle nach Siegen gut ist, wo sie nur rauscht und wo
sie leise falsch wird. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import stb_constants as C
import stb_evaluation as ev
from stb_pdf_export import build_pdf
from stb_presets import (
    apply_preset,
    bounds,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from stb_scenario import Settings
from stb_visualization import build_best_case, build_luecken, build_matrix, build_place_scatter, build_staffeln, build_zyklen

st.set_page_config(page_title="Tabelle nach Siegen – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 200 feste Turniere …")
def _best_case():
    return ev.sweep_best_case()


@st.cache_data(show_spinner="Experiment rechnet über 200 feste Turniere …")
def _staffeln():
    return ev.sweep_staffeln()


@st.cache_data(show_spinner="Experiment rechnet über 200 feste Turniere …")
def _luecken():
    return ev.sweep_luecken()


@st.cache_data(show_spinner="Experiment rechnet über 200 feste Turniere …")
def _zyklen():
    return ev.sweep_zyklen()


def verdict(level, text):
    SHOW[level](text)


st.title("🏆 Tabelle nach Siegen: sie misst das Feld, nicht die Stärke")
st.markdown(
    """
Wer gewinnt ein Turnier, wenn man nur Siege zählt? Diese Demo ist das **erste Stück der Ranking-Linie** und die **erste Demo der Reihe, die aus
Paarvergleichen eine Rangfolge schätzt**. In einem **neutralen, simulierten Turnier** ist die wahre Stärke jedes Teilnehmers bekannt, sodass sich
der Fehler der **Tabelle nach Siegen** messen lässt. Das Ergebnis ist differenzierter als „die Tabelle lügt“: Sie ist unverzerrt (nur verrauscht), wo jeder gegen jeden
gleich oft spielt, und sie wird **leise** falsch, wenn Gegner ungleich sind, Partien fehlen oder die Leistung am Gegner hängt.

Weiter unten: vier **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert eine Tabelle nach Siegen", expanded=True):
    st.markdown(
        """
1. **Partien spielen:** Jedes Paar spielt so oft, wie der Spielplan es vorsieht. Eine Partie gewinnt der Stärkere nur mit einer Wahrscheinlichkeit
   (je größer der Stärkeunterschied, desto wahrscheinlicher; bei Abstand 0,3 und 12 Teilnehmern gewinnt der Erste gegen den Letzten nur mit etwa 96 %).
2. **Siege zählen:** Die Tabelle ordnet nach der Zahl der Siege oder, wenn Sie es wählen, nach der Siegquote (sinnvoll, wenn nicht alle gleich oft gespielt haben).
3. **Gleichstände auflösen:** erst durch den **Direktvergleich** unter den Punktgleichen, danach durch das **Los**.
4. **Was sie misst:** den Erfolg gegen **die Gegner, die tatsächlich gespielt haben**. Ob das die Stärke ist, hängt vom Feld ab.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Teilnehmer**")
    n = st.slider("Teilnehmer n", *bounds("n_slider"), key="n_slider", help="Wie viele Teilnehmer im Turnier antreten.")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier benachbarter Teilnehmer (logistische Skala). Bei 0,3 gewinnt der Stärkere gegen den Nächsten mit etwa 57 %.")
    st.markdown("**Spielplan**")
    plan = st.selectbox("Spielplan", list(C.PLAN_LABELS), format_func=C.PLAN_LABELS.get, key="plan_select")
    k = st.slider("Partien je Paar", *bounds("k_slider"), key="k_slider", help="Wie oft ein Paar gegeneinander spielt, wenn es überhaupt spielt.")
    if plan == "staffeln":
        seed_widget("x_slider")
        x = st.slider("Kreuzpartien je Spieler der starken Staffel", *bounds("x_slider"), key="x_slider",
                      help="Gegen wie viele zufällige Spieler der schwachen Staffel jeder Starke zusätzlich antritt. 0 = die Staffeln begegnen sich nie.")
        remember("x_slider", x)
    if plan == "luecken":
        seed_widget("p_slider")
        p = st.slider("Wahrscheinlichkeit, dass ein Paar spielt", *bounds("p_slider"), step=C.P_STEP, key="p_slider",
                      help="Jedes Paar spielt mit dieser Wahrscheinlichkeit (dann k-mal), sonst gar nicht.")
        remember("p_slider", p)
    st.markdown("**Zyklen**")
    c = st.slider("Zyklenstärke c", *bounds("c_slider"), step=C.CYCLE_STEP, key="c_slider",
                  help="Wie stark drei Spielstile einander kontern (Stil 1 schlägt 0, 2 schlägt 1, 0 schlägt 2). 0 = keine Zyklen.")
    if c > 0:       # ohne Zyklen gibt es keine Stile: der Regler wäre wirkungslos
        seed_widget("stil_select")
        stil = st.selectbox("Stilverteilung", list(C.STIL_LABELS), format_func=C.STIL_LABELS.get, key="stil_select",
                            help="Wie die drei Spielstile verteilt sind. Bei Konter und Beute hat der Stärkste Stil 0; bei „ausgewogen“ sind die Stile reihum vergeben.")
        remember("stil_select", stil)
    st.markdown("**Auswertung**")
    wertung = st.selectbox("Wertung", list(C.WERTUNG_LABELS), format_func=C.WERTUNG_LABELS.get, key="wertung_select")
    seed = st.number_input("Zufalls-Seed des Turniers", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Turnier würfeln", width="stretch", on_click=randomize_seed)

x_value = int(current_value("x_slider"))
p_value = float(current_value("p_slider"))
stil_value = str(current_value("stil_select"))
sync_query_params({"n_slider": int(n), "delta_slider": float(delta), "k_slider": int(k), "plan_select": plan, "x_slider": x_value,
                   "p_slider": p_value, "c_slider": float(c), "stil_select": stil_value, "wertung_select": wertung, "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(k), plan, x_value, round(p_value, 2), round(float(c), 2), stil_value, wertung, int(seed))
a = _analysis(settings)
t = a.turnier

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Tabelle")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Platz des Stärksten", f"{a.best_place} von {t.n}", delta="Referenz: Platz 1", delta_color="off")
m2.metric("τ gegen die Stärke", ev.de(a.tau_s), delta="1 = gleiche Ordnung", delta_color="off",
          help="Kendall-Tau zwischen Tabelle und Generator-Stärke: +1 gleiche, −1 umgekehrte Ordnung.")
m3.metric("τ gegen die erwartete Siegquote", ev.de(a.tau_e), delta="1 = gleiche Ordnung", delta_color="off",
          help="Dieselbe Tabelle gegen die erwartete Siegquote gegen ein gleichverteiltes Feld.")
m4.metric("Gleichstände", f"{ev.pct(a.tie_share)} % der Paare", delta=f"{a.los_places} Plätze per Los", delta_color="off")

order = list(a.table.order)
games_v = t.games.sum(axis=1)
wins_v = t.wins.sum(axis=1)
place_true = (pd.Series(t.strengths).rank(ascending=False, method="min")).astype(int)
df = pd.DataFrame({
    "Platz": range(1, t.n + 1),
    "Teilnehmer": [t.names[i] for i in order],
    "Siege": [int(wins_v[i]) for i in order],
    "Partien": [int(games_v[i]) for i in order],
    "Siegquote": [round(wins_v[i] / games_v[i], 2) if games_v[i] else 0.0 for i in order],
    "Platz nach Stärke": [int(place_true[i]) for i in order],
    "Stärke s": [round(float(t.strengths[i]), 2) for i in order],
    "erwartete Siegquote": [round(float(t.expected[i]), 2) for i in order],
    "entschieden durch": [a.table.decided_by[pos] for pos in range(t.n)],
})
st.dataframe(df, hide_index=True, width="stretch")
left, right = st.columns(2)
with left:
    st.markdown("**Siegmatrix** (Zeile gewinnt gegen Spalte, in Tabellenreihenfolge)")
    st.plotly_chart(build_matrix(a), width="stretch", key="matrix_chart")
with right:
    st.markdown("**Tabellenplatz gegen Platz nach Stärke** (auf der Diagonale stimmt die Tabelle)")
    st.plotly_chart(build_place_scatter(a), width="stretch", key="place_chart")
st.caption("Die Stärke ist hier **bekannt, weil das Turnier simuliert ist**; in einem echten Turnier sieht man nur die Ausgänge der Partien.")
st.download_button("📄 PDF der aktuellen Tabelle", data=build_pdf(a), file_name="siegtabelle.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Trifft die Tabelle den Stärksten, und welche Frage beantwortet sie?")
verdict(*ev.verdict_core(a))
if not math.isnan(a.contradiction):
    st.caption(f"In {ev.pct(a.contradiction)} % der gespielten, nicht ausgeglichenen Paare mit verschiedener Wertung steht der in der Tabelle Hintere, "
               "der den direkten Vergleich gewonnen hat. Ein kleiner Wert heißt nur, dass die Tabelle in sich stimmig ist, nicht dass sie die Stärke trifft.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie gut ist die Tabelle im besten Fall?")
st.caption("Vollständiges Rundenturnier mit 8 Teilnehmern, Stärkeabstand 0,3, Mittel über 200 feste Turniere (SWEEP_SEEDS). "
           "Die Regler wirken auf die Tabelle oben, nicht auf die Experimente.")
rows1 = _best_case()
st.plotly_chart(build_best_case(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_best_case(rows1))

st.subheader("🔬 Experiment 2: Wer gegen wen? Zwei Staffeln")
st.caption("12 Teilnehmer (feste Instanz, unabhängig vom Regler): 6 starke und 6 schwache. Innerhalb jeder Staffel spielt jeder gegen jeden, dazu x "
           "Kreuzpartien je starkem Spieler.")
rows2 = _staffeln()
st.plotly_chart(build_staffeln(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_staffeln(rows2))

st.subheader("🔬 Experiment 3: Wenn Partien fehlen")
st.caption("12 Teilnehmer (feste Instanz): Jedes Paar spielt mit Wahrscheinlichkeit p genau einmal. Verglichen werden die Wertungen Siege und Siegquote.")
rows3 = _luecken()
st.plotly_chart(build_luecken(rows3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_luecken(rows3))

st.subheader("🔬 Experiment 4: Wenn die Leistung am Gegner hängt")
st.caption("12 Teilnehmer (feste Instanz), 50 Partien je Paar, damit das Rauschen verschwindet und nur die Zyklen wirken. Zwei Zielgrößen: die "
           "Generator-Stärke und die **erwartete Siegquote** gegen ein gleichverteiltes Feld.")
rows4 = _zyklen()
st.plotly_chart(build_zyklen(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_zyklen(rows4))
df4 = pd.DataFrame([{"Stilverteilung": C.STIL_LABELS[r["stil"]], "c": r["c"], "τ gegen Stärke": round(r["tau_s"], 3),
                     "τ gegen erwartete Siegquote": round(r["tau_e"], 3), "Platz des Stärksten": round(r["best_place"], 1),
                     "Widerspruch zum direkten Vergleich": f"{ev.pct(r['contradiction'])} %", "zyklische Dreiecke": f"{r['cyclic']:.1f} von {r['triples']}"}
                    for r in rows4])
st.dataframe(df4, hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt eine Stärke, die die Partien erklärt** | Mit Spielstilen, die einander kontern, hängt die Leistung am Gegner. Die Tabelle trifft dann die erwartete Siegquote, nicht die Stärke (Experiment 4). | Nicht-Transitivität |
| **Jeder spielt gegen jeden, gleich oft** | Wenn die Staffeln nie gegeneinander spielen, ordnet die Tabelle über die Staffeln wie ein Münzwurf (Experiment 2); bei Lücken zerfällt der Plan oft in Teile, ohne dass der Widerspruch zum direkten Vergleich es zeigt (Experiment 3). | Massey und Colley, Bradley-Terry |
| **Eine Partie hat nur Sieg oder Niederlage** | Tore, Remis und Heimvorteil gehen verloren. Reale Turniere haben sie. | Tore, Remis, Heimvorteil |
| **Die Stärke bleibt gleich** | Form und Alter verändern sie von Partie zu Partie; die Tabelle gewichtet alle Partien gleich. | Elo, WHR und TrueSkill Through Time |
| **Die Zahl der Siege genügt** | Sie sagt nichts darüber, wie sicher ein Platz ist. Bei einer Partie je Paar steht der Stärkste in weniger als der Hälfte der Läufe vorn (Experiment 1). | Rangintervalle und Shrinkage |
| **Die Daten sind echt** | Alles hier ist simuliert; die „wahre Stärke“ ist eine Konstruktion des Generators, kein Messwert. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Teilnehmer $i = 1, \dots, n$ haben Stärken $s_i$. Eine Partie $i$ gegen $j$ gewinnt $i$ mit
$P_{ij} = \mathrm{clip}\big(\sigma(s_i - s_j) + K_{ij},\, 0{,}02,\, 0{,}98\big)$, wobei $\sigma$ die logistische Funktion ist und $K_{ij}$ eine
antisymmetrische Störung über drei Spielstile (Stil $a$ schlägt Stil $a-1$ modulo 3 mit Zuschlag $c$). Die Begrenzung auf $[0{,}02;\ 0{,}98]$
gilt nur bei $c > 0$.

**Tabelle.** Siege $w_i = \sum_j W_{ij}$ oder Siegquote $w_i / g_i$ mit $g_i$ gespielten Partien. Gleichstand: erst die Siege innerhalb der
Gruppe der Punktgleichen (Direktvergleich), dann das Los.

**Erwartete Siegquote.** $e_i = \frac{1}{n-1}\sum_{j \ne i} P_{ij}$. Im vollständigen Plan gilt $\mathbb{E}[w_i] = k\,(n-1)\,e_i$, die Tabelle schätzt
also $e$, nicht $s$.

**Kendall-τ.** $\tau = \frac{C - D}{C + D}$ über alle Paare mit verschiedenem Vergleichswert ($C$ gleich, $D$ entgegengesetzt geordnete Paare).

**Zyklische Dreiecke.** Im Mehrheitsturnier ohne Gleichstände gilt $\binom{n}{3} - \sum_i \binom{w_i}{2}$, wobei $w_i$ hier die Zahl der Gegner ist,
gegen die $i$ in der Mehrheit gewonnen hat (bei einer Partie je Paar die Siegzahl); mit Gleichständen und ungespielten Paaren
zählt die Demo über $\mathrm{tr}(B^3)/3$ mit der Kantenmatrix $B_{ij} = [W_{ij} > W_{ji}]$.

**Zufall.** SplitMix64-Strom, Ausgang $k$ hängt nur von (Seed, $k$) ab; eine Partie ist „Sieg“, wenn die gleichverteilte Zahl unter $P_{ij}$ liegt.

**Literatur.**
- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39 (3/4), 324–345
  (das logistische Paarvergleichsmodell).
- Zermelo, E. (1929): Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der Wahrscheinlichkeitsrechnung. *Mathematische Zeitschrift* 29,
  436–460; Ford, L. R., Jr. (1957): Solution of a ranking problem from binary comparisons. *American Mathematical Monthly* 64, 28–33
  (der Schätzer existiert genau dann, wenn in jeder Zerlegung der Teilnehmer in zwei nichtleere Gruppen jemand aus der ersten gegen jemanden aus
  der zweiten gewonnen hat; ein ungeschlagener Teilnehmer verletzt das).
- Kendall, M. G., Babington Smith, B. (1940): On the method of paired comparisons. *Biometrika* 31 (3/4), 324–345 (zyklische Dreiecke und
  Konsistenzkoeffizient). Die Zählformel oben ist zu der dortigen Zählung über die Siegezahlen äquivalent und hier gegen die Aufzählung aller
  Dreiergruppen geprüft.
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472; Konstanten und Schritte nach
  Vigna, `splitmix64.c` (prng.di.unimi.it).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
