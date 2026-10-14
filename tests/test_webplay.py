"""The browser bridge of `mz web`: deck reading and the decision hand-off between engine and browser."""

import threading
import time

import pytest

from magezero import webplay
from magezero.webplay import Table, read_deck, strip_tags


def test_read_deck_counts_main_deck_and_names_commander(tmp_path):
    deck = tmp_path / "d.dck"
    deck.write_text(
        "NAME:test\n1 [MH3:237] Ajani, Nacatl Pariah\n24 [SNC:280] Forest\nSB: 1 [IKO:216] Winota, Joiner of Forces\n",
        encoding="utf-8",
    )
    cards, names, sideboard = read_deck(deck)
    assert cards == 25
    assert sideboard == ["Winota, Joiner of Forces"]
    assert names == {"Ajani, Nacatl Pariah", "Forest", "Winota, Joiner of Forces"}


def test_strip_tags_keeps_log_text():
    assert strip_tags("<font color='#20B2AA'>Human</font> casts Sol Ring&nbsp;[1fd]") == "Human casts Sol Ring [1fd]"


def decision(name="d1"):
    return {
        "decision": name,
        "kind": "cast",
        "prompt": "Your turn",
        "options": [{"id": "priority-0", "type": "pass", "label": "Pass priority"}],
        "state": {"turn": 3, "log": ["<b>Turn 3</b>"]},
    }


def test_engine_waits_for_the_browser_answer(monkeypatch):
    table = Table()
    replies = []
    engine = threading.Thread(target=lambda: replies.append(table.decide(decision())))
    engine.start()
    deadline = time.monotonic() + 5
    while table.view()["decision"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    view = table.view()
    assert view["decision"]["prompt"] == "Your turn" and "state" not in view["decision"]
    assert view["state"]["log"] == ["Turn 3"]
    with pytest.raises(ValueError):
        table.respond("d1", "not-offered")
    with pytest.raises(ValueError):
        table.respond("stale", "priority-0")
    table.respond("d1", "priority-0")
    engine.join(timeout=5)
    assert replies == [{"action_id": "priority-0"}]
    assert table.view()["decision"] is None


def test_unanswered_decision_is_asked_again_without_reset(monkeypatch):
    monkeypatch.setattr(webplay, "HOLD_SECONDS", 0.05)
    table = Table()
    assert table.decide(decision()) == {"wait": True}
    version = table.version
    # The engine repeats the same decision: it stays pending and an answer still counts.
    assert table.decide(decision()) == {"wait": True}
    assert table.version == version
    table.respond("d1", "priority-0")
    assert table.decide(decision()) == {"action_id": "priority-0"}


def test_concede_and_abort_reach_the_engine():
    table = Table()
    table.concede = True
    assert table.decide(decision()) == {"concede": True}
    assert table.publish({"state": {"turn": 4, "log": []}}) == {"concede": True}
    table.abort = True
    assert table.decide(decision("d2")) == {"abort": True}
    assert table.publish({"state": {"turn": 4, "log": []}}) == {"abort": True}


def test_only_the_controlling_tab_answers(monkeypatch):
    monkeypatch.setattr(webplay, "HOLD_SECONDS", 0.05)
    table = Table()
    table.decide(decision())
    table.claim("tab-a")
    table.claim("tab-b")  # the newest tab takes over
    assert table.view()["controller"] == "tab-b"
    with pytest.raises(ValueError, match="another browser tab"):
        table.respond("d1", "priority-0", "tab-a")
    table.respond("d1", "priority-0", "tab-b")
    assert table.decide(decision()) == {"action_id": "priority-0"}
