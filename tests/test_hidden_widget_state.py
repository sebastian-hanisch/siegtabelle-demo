"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem
Widget-Zustand; sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück. AppTest sieht den Frontend/Backend-Desync
nicht, die Invariante auf dem Session-Zustand aber schon."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def slider_keys(at):
    return {s.key for s in at.slider}


def test_hideable_sliders_are_only_drawn_in_their_plan():
    at = fresh()
    assert at.selectbox(key="plan_select").value == "vollstaendig"
    assert not {"x_slider", "p_slider"} & slider_keys(at)
    assert "x_slider" not in at.session_state and "p_slider" not in at.session_state
    at.selectbox(key="plan_select").select("staffeln").run(timeout=TIMEOUT)
    assert "x_slider" in slider_keys(at) and "p_slider" not in slider_keys(at)
    at.selectbox(key="plan_select").select("luecken").run(timeout=TIMEOUT)
    assert "p_slider" in slider_keys(at) and "x_slider" not in slider_keys(at)
    assert not at.exception


def test_hidden_value_survives_switching_the_plan_away_and_back():
    at = fresh()
    at.selectbox(key="plan_select").select("staffeln").run(timeout=TIMEOUT)
    at.slider(key="x_slider").set_value(5).run(timeout=TIMEOUT)
    at.selectbox(key="plan_select").select("luecken").run(timeout=TIMEOUT)
    assert "x_slider" not in at.session_state                    # nicht gezeichnet: kein Widget-Zustand
    at.slider(key="p_slider").set_value(0.3).run(timeout=TIMEOUT)
    at.selectbox(key="plan_select").select("staffeln").run(timeout=TIMEOUT)
    assert at.slider(key="x_slider").value == 5                   # gemerkter Wert kommt zurück
    assert "p_slider" not in at.session_state
    at.selectbox(key="plan_select").select("luecken").run(timeout=TIMEOUT)
    assert abs(at.slider(key="p_slider").value - 0.3) < 1e-9
    assert not at.exception


def test_style_select_is_only_drawn_with_cycles_and_remembers_its_value():
    at = fresh()
    assert "stil_select" not in {s.key for s in at.selectbox}        # c = 0: keine Stile
    assert "stil_select" not in at.session_state
    at.slider(key="c_slider").set_value(0.3).run(timeout=TIMEOUT)
    assert at.selectbox(key="stil_select").value == "ausgewogen"
    at.selectbox(key="stil_select").select("konter").run(timeout=TIMEOUT)
    at.slider(key="c_slider").set_value(0.0).run(timeout=TIMEOUT)
    assert "stil_select" not in at.session_state
    at.slider(key="c_slider").set_value(0.2).run(timeout=TIMEOUT)
    assert at.selectbox(key="stil_select").value == "konter"          # gemerkt
    assert not at.exception


def test_preset_sets_the_hidden_slider_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Lückenhafter Plan"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="plan_select").value == "luecken"
    assert abs(at.slider(key="p_slider").value - 0.33) < 1e-9
    assert "x_slider" not in at.session_state
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Zwei Staffeln"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="plan_select").value == "staffeln"
    assert at.slider(key="x_slider").value == 1
    assert "p_slider" not in at.session_state
    assert not at.exception
