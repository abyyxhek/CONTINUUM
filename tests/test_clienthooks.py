"""Tests for the observe hook's payload extraction (``continuum.clienthooks``).

Covers the durable facts a PostToolUse hook extracts from the host's report:
tool name, file path, size and digest, and the optional correlation
identifier that links the observation to a ledger action attempt (#785).

The correlation branches exist on a trust boundary. The hook process runs
inside the host (Claude Code, Cursor, ...), so every value it forwards is
agent-controlled. The hook's own contract forbids it from failing the host
tool call, so a malformed identifier is dropped while the observation is
still recorded, and the strict check stays on the explicit CLI flag (review
of #1680, and the coverage request on #1875).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from continuum.clienthooks import (
    install_claude_code_hook,
    observe_event_payload,
    remove_claude_code_hook,
)


def test_valid_correlation_id_is_carried_into_the_observation() -> None:
    """A valid top-level correlation ID is forwarded to the event payload."""
    payload = observe_event_payload(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "make"},
            "correlation_id": "attempt-1",
        }
    )
    assert payload["correlation_id"] == "attempt-1"
    assert payload["tool"] == "Bash"


def test_correlation_id_nested_in_tool_input_is_carried() -> None:
    """A correlation ID nested in ``tool_input`` is read when the top level has none.

    Hosts pass the agent's tool arguments as ``tool_input`` and do not reserve
    a top-level slot for a correlation ID, so the nested read is the path the
    feature actually takes in production.
    """
    payload = observe_event_payload(
        {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "x.py",
                "correlation_id": "attempt-2",
            },
        }
    )
    assert payload["correlation_id"] == "attempt-2"


def test_top_level_correlation_id_wins_over_a_nested_one() -> None:
    """A top-level ID is preferred; the nested one is only a fallback."""
    payload = observe_event_payload(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "make", "correlation_id": "nested"},
            "correlation_id": "top-level",
        }
    )
    assert payload["correlation_id"] == "top-level"


@pytest.mark.parametrize("bad", [42, "", "   ", "x" * 129])
def test_malformed_correlation_id_is_dropped_not_fatal(bad: object) -> None:
    """A malformed agent-supplied ID is dropped while the observation survives.

    The hook must never fail the host tool call (its own contract), and the
    observation is durable evidence regardless of linkage quality.
    """
    payload = observe_event_payload(
        {
            "tool_name": "Write",
            "tool_input": {"file_path": "x.py", "correlation_id": bad},
        }
    )
    assert "correlation_id" not in payload
    assert payload["tool"] == "Write"


def test_observation_without_any_correlation_id_is_unchanged() -> None:
    """An observation with no identifier anywhere is recorded uncorrelated."""
    payload = observe_event_payload({"tool_name": "Write", "tool_input": {"file_path": "x.py"}})
    assert "correlation_id" not in payload
    assert payload["tool"] == "Write"


def test_non_mapping_tool_input_is_not_searched_for_a_correlation_id() -> None:
    """A ``tool_input`` that is not a mapping is skipped, not crashed on."""
    payload = observe_event_payload({"tool_name": "Bash", "tool_input": "make"})
    assert "correlation_id" not in payload
    assert payload["tool"] == "Bash"


def test_file_facts_are_still_recorded_alongside_a_correlation_id(
    tmp_path: Path,
) -> None:
    """Linking an observation does not displace the durable file facts.

    Size and digest describe what is on disk and are what recovery reads; the
    correlation ID is additive metadata on the same event.
    """
    written = tmp_path / "built.log"
    written.write_bytes(b"correlated build output" * 4)
    payload = observe_event_payload(
        {
            "tool_name": "Write",
            "tool_input": {
                "file_path": str(written),
                "correlation_id": "attempt-3",
            },
        }
    )
    assert payload["correlation_id"] == "attempt-3"
    assert payload["path"] == str(written)
    assert payload["bytes"] == len(b"correlated build output" * 4)
    assert payload["sha256"] == hashlib.sha256(b"correlated build output" * 4).hexdigest()


# --- install and remove: settings validation and the claude-code wrapper (#2050) --- #
#
# The installer must refuse to touch a settings file it cannot read as the
# documented shape, and the remover must degrade rather than crash on the same
# shapes. These operate purely on tmp_path files.


def _install_observe(settings: Path, command: str = "continuum observe") -> str:
    return install_claude_code_hook(settings, command, kind="observe")


def test_install_claude_code_hook_writes_each_kinds_event_and_matcher(tmp_path: Path) -> None:
    """Each kind lands under its own Claude Code event with the documented
    default matcher: observe on PostToolUse over the write tools, gate on
    PreToolUse over anything, briefing on SessionStart, precompact on
    PreCompact (both with no matcher)."""
    cases = {
        "observe": ("PostToolUse", "Write|Edit|MultiEdit|NotebookEdit"),
        "gate": ("PreToolUse", "*"),
        "briefing": ("SessionStart", ""),
        "precompact": ("PreCompact", ""),
    }
    for kind, (event, matcher) in cases.items():
        settings = tmp_path / f"settings-{kind}.json"
        assert install_claude_code_hook(settings, f"continuum {kind}", kind=kind) == "installed"
        hooks = json.loads(settings.read_text(encoding="utf-8"))["hooks"]
        (group,) = hooks[event]
        assert group["matcher"] == matcher
        assert group["hooks"] == [{"type": "command", "command": f"continuum {kind}"}]


def test_install_claude_code_hook_falls_back_to_the_gate_event_for_an_unknown_kind(
    tmp_path: Path,
) -> None:
    """An unrecognised kind is a gate by default, the narrowest safe reading."""
    settings = tmp_path / "settings.json"
    install_claude_code_hook(settings, "continuum gate", kind="mystery")
    hooks = json.loads(settings.read_text(encoding="utf-8"))["hooks"]
    assert "PreToolUse" in hooks


def test_install_claude_code_hook_preserves_a_custom_matcher(tmp_path: Path) -> None:
    """A caller-supplied matcher wins over the kind's default."""
    settings = tmp_path / "settings.json"
    install_claude_code_hook(settings, "continuum observe", kind="observe", matcher="Bash")
    (group,) = json.loads(settings.read_text(encoding="utf-8"))["hooks"]["PostToolUse"]
    assert group["matcher"] == "Bash"


def test_install_refuses_an_unparseable_settings_file(tmp_path: Path) -> None:
    """A file someone edited by hand is a statement of intent: refusing beats
    destroying it to save a typo."""
    settings = tmp_path / "settings.json"
    settings.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        _install_observe(settings)


def test_install_refuses_a_settings_file_that_is_not_a_json_object(tmp_path: Path) -> None:
    for contents in ("[]", '"a string"', "17"):
        settings = tmp_path / "settings.json"
        settings.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError, match="does not contain a JSON object"):
            _install_observe(settings)


def test_install_refuses_a_hooks_property_that_is_not_an_object(tmp_path: Path) -> None:
    for hooks_value in ('"hooks"', "[]"):
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"hooks": hooks_value}), encoding="utf-8")
        with pytest.raises(ValueError, match="'hooks' is not an object"):
            _install_observe(settings)


def test_install_refuses_an_event_list_that_is_not_a_list(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {"PostToolUse": {"matcher": ""}}}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"'hooks.PostToolUse' is not a list"):
        _install_observe(settings)


def test_install_ignores_unrelated_group_shapes_and_still_appends(tmp_path: Path) -> None:
    """Non-dict groups and groups whose ``hooks`` is not a list are skipped,
    not crashed on, and the new entry still lands."""
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        "a stray string",
                        {"matcher": "Bash", "hooks": "not-a-list"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert _install_observe(settings) == "installed"
    payload = json.loads(settings.read_text(encoding="utf-8"))
    groups = payload["hooks"]["PostToolUse"]
    assert groups[:2] == ["a stray string", {"matcher": "Bash", "hooks": "not-a-list"}]
    (group,) = [g for g in groups if isinstance(g, dict) and isinstance(g.get("hooks"), list)]
    assert group["hooks"] == [{"type": "command", "command": "continuum observe"}]


def test_remove_refuses_an_unparseable_settings_file(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        remove_claude_code_hook(settings)


def test_remove_returns_false_for_non_object_settings(tmp_path: Path) -> None:
    for contents in ("[]", "17"):
        settings = tmp_path / "settings.json"
        settings.write_text(contents, encoding="utf-8")
        assert remove_claude_code_hook(settings) is False


def test_remove_returns_false_when_hooks_is_not_an_object(tmp_path: Path) -> None:
    for hooks_value in ('"hooks"', "[]"):
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"hooks": hooks_value}), encoding="utf-8")
        assert remove_claude_code_hook(settings) is False


def test_remove_skips_non_list_event_arrays_without_raising(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {"PostToolUse": "not-a-list"}}), encoding="utf-8")
    assert remove_claude_code_hook(settings) is False


def test_remove_returns_false_when_nothing_is_ours(tmp_path: Path) -> None:
    """No managed entries means nothing was removed and nothing is rewritten."""
    settings = tmp_path / "settings.json"
    original = {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "",
                    "hooks": [{"type": "command", "command": "someone-else observe"}],
                }
            ]
        }
    }
    settings.write_text(json.dumps(original), encoding="utf-8")
    assert remove_claude_code_hook(settings) is False
    assert json.loads(settings.read_text(encoding="utf-8")) == original


def test_remove_preserves_unrelated_group_shapes(tmp_path: Path) -> None:
    """A stray string group and a group whose ``hooks`` is not a list survive
    the prune untouched; only the recognised entry is dropped."""
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        "a stray string",
                        {"matcher": "Bash", "hooks": "not-a-list"},
                        {
                            "matcher": "",
                            "hooks": [{"type": "command", "command": "continuum observe"}],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert remove_claude_code_hook(settings) is True
    payload = json.loads(settings.read_text(encoding="utf-8"))
    assert payload["hooks"]["PostToolUse"] == [
        "a stray string",
        {"matcher": "Bash", "hooks": "not-a-list"},
    ]


def test_remove_drops_the_hooks_key_when_nothing_is_left(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_claude_code_hook(settings, "continuum observe", kind="observe")
    assert remove_claude_code_hook(settings) is True
    assert "hooks" not in json.loads(settings.read_text(encoding="utf-8"))
