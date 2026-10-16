# Tabelle nach Siegen: sie misst das Feld, nicht die Stärke – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-siegtabelle-demo.streamlit.app/)**

Wurzel der **Ranking-Linie** (Konzepte-Demo): Aus Paarvergleichen eine Rangfolge schätzen. Das Vehikel ist ein **neutrales, simuliertes Turnier**
mit bekannter Stärke je Teilnehmer, damit sich der Fehler jedes Verfahrens messen lässt. Dieses Stück fragt nur: Was misst die **Tabelle nach
Siegen**, wo ist sie gut und wo wird sie leise falsch? Alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Wer gewinnt, wenn man nur Siege zählt? Die Antwort ist differenzierter als „die Tabelle lügt“: Sie **misst den Erfolg gegen das gespielte Feld**.
Sie ist redlich, wenn jeder gegen jeden gleich oft spielt (dann nur verrauscht). Sie versagt **leise**, wenn die Gegner ungleich sind, Partien
fehlen oder die Leistung vom Gegner abhängt, und nichts in der Tabelle verrät es.

## Modell

- n Teilnehmer mit Stärken sᵢ, äquidistant mit Abstand δ (Standard 0,3: der Stärkere gewinnt gegen den nächsten Nachbarn mit etwa 57 %).
- Eine Partie i gegen j gewinnt i mit P = σ(sᵢ − sⱼ), optional plus eine zyklische Störung über drei Spielstile (Zuschlag c, auf 0,02 bis 0,98
  begrenzt). Beobachtet werden nur Ausgänge, nie die Stärke.
- **Zwei Zielgrößen:** die Generator-Stärke s und die **erwartete Siegquote** eᵢ gegen ein gleichverteiltes Feld. Ohne Zyklen ordnen beide gleich.
  Die Stärke ist eine **Konstruktion des Generators**, kein Messwert.
- Drei Spielpläne: vollständig, zwei Staffeln (stark/schwach, dazu x Kreuzpartien), zufällige Lücken (jedes Paar mit Wahrscheinlichkeit p).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Tabelle:** Siege oder Siegquote; Gleichstand erst durch den Direktvergleich unter den Punktgleichen, dann durch das Los.
- **Maße:** Kendall-τ gegen beide Zielgrößen, Platz des Stärksten, Gleichstandsanteil, Anteil der Paare, in denen der in der Tabelle Hintere den
  direkten Vergleich gewann, zyklische Dreiecke, Zahl der Teile des Spielplans.
- **Experimente** über 200 feste Turniere (`SWEEP_SEEDS`, nie ein einzelner Seed): vier Experimente mit je einem live berechneten Verdikt.
- **Zufall:** gemeinsamer Generator `turnier_generator.py` mit SplitMix64-Strom (Ergebnis hängt nur von Seed und Position ab, plattformunabhängig).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 200 feste Turniere (`SWEEP_SEEDS`), Experiment 1 mit 8, die übrigen mit 12 Teilnehmern, Stärkeabstand 0,3.

| Frage | Befund | Test |
|---|---|---|
| Wie gut ist die Tabelle im besten Fall (vollständig, ausgeglichen)? | Mit einer Partie je Paar steht der Stärkste nur in 40,5 % der Läufe auf Platz 1 (τ = 0,53); mit 4 Partien in 64,5 % (0,78), mit 10 Partien in 74,5 % (0,88), mit 40 Partien in 93 % (0,98). **Reines Rauschen**, kein Verfahrensfehler. | `test_one_game_per_pair_the_strongest_leads_in_less_than_half_of_the_runs`, `test_more_games_help_monotonically_and_forty_games_nearly_fix_it` |
| Ist die Siege-Ordnung dort schon die Bradley-Terry-Ordnung? | **Ja**, wo der Schätzer existiert (alle auswertbaren Turniere in der Stichprobe). Bei einer Partie je Paar existiert er praktisch nie. | `test_wins_order_equals_bradley_terry_order_in_a_balanced_complete_tournament` |
| Gleichstände | Bei einer Partie je Paar stehen 14 % aller Paare punktgleich, bei 40 Partien 0,2 %. | `test_ties_are_common_with_one_game_and_rare_with_forty` |
| Zwei Staffeln ohne Kreuzpartien | 50 % der Paare über die Staffeln falsch geordnet (Münzwurf), der Plan hat zwei Teile; die Tabelle weist trotzdem Plätze 1 bis 12 aus. Mit einer Kreuzpartie je starkem Spieler 35 %, mit drei 16,5 %, wenn alle gegeneinander spielen 6 %. | `test_without_crossing_games_the_two_leagues_are_ordered_like_a_coin_flip`, `test_error_falls_with_every_crossing_game_but_never_reaches_zero` |
| Fehlende Partien | Gewertet nach Siegquote ordnet die Tabelle bei p = 0,5 besser als nach Siegen (τ 0,58 gegen 0,50). Bei p = 0,33 zerfällt der Plan in 17 % der Läufe in Teile, bei p = 0,2 in 72 %. | `test_quote_beats_wins_when_games_are_missing`, `test_sparse_plans_fall_apart_and_the_contradiction_hides_it` |
| Verrät der Widerspruch zum direkten Vergleich die Lücken? | **Nein:** er sinkt von 17 % (vollständig) auf 7 % (p = 0,12), obwohl der Plan fast ganz zerfällt. | `test_sparse_plans_fall_apart_and_the_contradiction_hides_it` |
| Zyklen bei ausgewogenen Stilen | Die Ordnung bleibt (τ 0,99 bei c = 0,4), obwohl 47 von 220 Dreiergruppen zyklisch sind und 28 % der gespielten Paare der Tabelle widersprechen. | `test_balanced_styles_leave_the_order_nearly_untouched_although_many_triples_cycle` |
| Zyklen bei unausgewogenen Stilen (der Stärkste wird von den meisten gekontert) | Bei c = 0,2 steht der Stärkste im Mittel auf Platz 3,3 (τ gegen die Stärke 0,74), bei c = 0,4 auf Platz 5,7 (τ 0,525); dabei widersprechen weniger als 5 % der direkten Vergleiche der Tabelle. | `test_countered_strongest_slides_down_but_expected_rate_is_hit` |
| Trifft die Tabelle dann wenigstens die erwartete Siegquote? | **Ja**, in allen neun Konfigurationen: τ zwischen 0,97 und 0,99 (Konter-Stil bei c = 0,4: 0,97, wo es gegen die Stärke nur 0,525 sind). | `test_table_hits_expected_rate_in_every_configuration` |
| Dominiert der Beute-Stil? | Der Stärkste bleibt im Mittel auf Platz 1,0 (c = 0,2). | `test_prey_dominated_field_keeps_the_strongest_in_front` |

## Befunde und Korrekturen gegenüber dem Plan

- **Skizze: „die Siegtabelle lügt“.** Gemessen ist das falsch: Bei vollständigem, ausgeglichenem Plan ist die Tabelle die Bradley-Terry-Ordnung
  und ihr Fehler reines Rauschen. Titel und Kernbotschaft sind entsprechend geändert („sie misst das Feld, nicht die Stärke“).
- **Skizze: „Zyklen verfälschen die Rangfolge“.** Gemessen nur, wenn das Feld unausgewogen ist (Stärkster gekontert). Ausgewogene Zyklen
  verschieben die Ordnung kaum.
- **Skizze: Widerspruch zum direkten Vergleich als Qualitätsanzeige.** Für Lücken ungeeignet (siehe oben), für ausgewogene Zyklen brauchbar.
- **Zielgröße.** Unter Zyklen ist „wahre Stärke“ mehrdeutig; die Demo zeigt deshalb beide Größen. Die Tabelle trifft die erwartete Siegquote.

## Abgrenzung

- [Axelrod-Turnier](https://sebastianhanisch-axelrod-turnier-demo.streamlit.app/) (Wiederholte Spiele) zeigt „Rang hängt am Feld“ für
  deterministische Strategien im Gefangenendilemma. Dort gibt es keine wahre Stärke, an der sich ein Fehler messen ließe; hier schon. Dazu kommen
  hier Spielplan-Ungleichgewicht und Lücken, die dort nicht vorkommen.
- [Rundenturnier](https://sebastianhanisch-round-robin-demo.streamlit.app/) (Turnierplanung) konstruiert den Spielplan (Berger-Tafeln) und
  bewertet kein Ergebnis; diese Demo nutzt vollständige und zufällig lückenhafte Pläne ohne die Zirkelmethode.

## Ehrliche Grenzen

- **Ein Generator:** Bradley-Terry-Stärke plus eine einfache additive Stil-Störung. Echte Turniere haben Remis, Heimvorteil, Form und Zeit.
- **Zielgröße:** Die Generator-Stärke ist die transitive Komponente; die erwartete Siegquote ist eine zweite, ebenso vertretbare Wahl.
- **Kleine Instanzen:** 8 und 12 Teilnehmer, ein Stärkeabstand; ab etwa 50 Teilnehmern ändert sich das Bild (Schweizer-System-Regime).
- **Alles simuliert:** Aussagen gelten für diesen Generator mit seinen Annahmen, nicht für reale Ranglisten.
- **Experimente unabhängig vom Regler:** Experiment 1 rechnet mit 8, die Experimente 2 bis 4 mit 12 Teilnehmern.

## Quellen

- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39 (3/4), 324–345.
- Zermelo, E. (1929): Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der Wahrscheinlichkeitsrechnung. *Mathematische Zeitschrift* 29,
  436–460. Ford, L. R., Jr. (1957): Solution of a ranking problem from binary comparisons. *American Mathematical Monthly* 64, 28–33
  (Existenzbedingung des Schätzers; Wortlaut der Bedingung nach einer Sekundärquelle, Original nicht eingesehen).
- Kendall, M. G., Babington Smith, B. (1940): On the method of paired comparisons. *Biometrika* 31 (3/4), 324–345. Die Zählformel der Demo ist zur
  Zählung über die Siegezahlen äquivalent und gegen die Aufzählung aller Dreiergruppen getestet.
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472. Konstanten nach Vigna,
  `splitmix64.c` (prng.di.unimi.it).

## Tests

91 Tests, rund 25 Sekunden: Einheiten gegen von Hand gerechnete Mini-Instanzen (`test_algorithm.py`), Generator (`test_generator.py`),
**Orakel** (`test_oracle_siegtabelle.py`: zyklische Dreiecke gegen Aufzählung, Kendall-τ gegen `scipy`, Komponenten gegen `networkx`,
Siege-Ordnung gegen Bradley-Terry per `scipy.optimize`, exakte Aufzählung für n = 3, Direktvergleich gegen Aufzählung, erwartete Siegquote),
Presets, ausblendbare Regler (`test_hidden_widget_state.py`), alle Zahlen dieses README (`test_claims.py`) und AppTest-Rauchtests.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie): SplitMix64, Stärken, Stile, Spielpläne, Partien |
| `stb_scenario.py` | Reglerwerte (`Settings`) und das Turnier daraus |
| `stb_algorithm.py` | Tabelle, Gleichstandsregel, Kennzahlen |
| `stb_evaluation.py` | Analyse, Experimente über die festen Instanzen, Verdikte |
| `stb_presets.py`, `stb_constants.py` | Permalink, Presets, ausblendbare Regler, Grenzen |
| `stb_visualization.py`, `stb_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Ein Modell für die Stärke** (Bradley-Terry, Elo, Glicko und TrueSkill, Massey und Colley): jeweils eigene Stücke der Linie.
- **Rangabstände** über τ hinaus (Spearman-Footrule, gewichtete Maße) und **Rangaggregation** (Borda, Copeland, Kemeny): eigene Stücke.
- **Tore, Remis, Heimvorteil:** die Demo kennt nur Sieg und Niederlage.
- **Echte Daten:** kein Datensatz, die Stärke ist Teil der Simulation.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
