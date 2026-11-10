"""The gate's ambiguous decision arms (issue #2055).

`decide()` is the idempotency enforcer: what stands between a replayed tool
call and a second side effect. The arms pinned here are precisely the
ambiguous ones, where the current answer (deny) was an accident of the code
rather than an assertion: a claim whose outcome is UNKNOWN, one that a
compensation closed, a foreign claim whose lookup raised, and a config the
gate cannot render. The memory-key parser's short form and its too-short
guards are pinned alongside, because tenant attribution is what makes a
multi-tenant store auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from continuum.actions.ledger import ActionLedger
from continuum.events import EventType
from continuum.gate import (
    GateConfigError,
    consumed_authority_reason,
    decide,
    derive_memory_key,
    load_gate_config,
    split_memory_key,
)
from continuum.models import Action, ActionStatus, Run
from continuum.storage import SQLiteStorage

CONFIG = {
    "tools": {
        "pgvector.upsert": {
            "key_template": "mem:{store_id}:{tenant}:{record_key}",
            "action_type": "mem_write",
        }
    }
}


def _seed(path: Path, run_id: str = "run_1") -> SQLiteStorage:
    store = SQLiteStorage(str(path))
    store.create_run(Run(run_id=run_id, goal="g"))
    store.append_event(run_id, EventType.RUN_STARTED, {"goal": "g"})
    return store


def _decide(store: SQLiteStorage, run_id: str = "run_1", **overrides: object):
    from continuum.actions.ledger import fold_action_events

    folded = fold_action_events(store.read_events(run_id))
    return decide(
        CONFIG["tools"],
        "pgvector.upsert",
        {"store_id": "pgvector_main", "tenant": "acme", "record_key": "rec-42"},
        run_id=run_id,
        actions_by_key=folded,
        **overrides,  # type: ignore[arg-type]
    )


# --- the memory-key decision table ----------------------------------------- #


def test_unknown_outcome_is_denied_with_reconcile_instructions(tmp_path: Path) -> None:
    """A memory claim the ledger itself records as uncertain blocks the retry.

    The completion arms are already pinned elsewhere; UNKNOWN is the outcome a
    timeout leaves behind, and the operator's next step is to reconcile it, not
    to repeat the call and find out.
    """

    with _seed(tmp_path / "mem.db") as store:
        ledger = ActionLedger(store, "run_1")
        claim = ledger.claim(
            "mem_write", {}, key="mem:pgvector_main:acme:rec-42", scoped_to_run=False
        )
        ledger.fail(claim.key, "timed out", certain=False)
        decision = _decide(store)
    assert decision.allow is False
    assert "continuum_reconcile_action" in decision.reason


def test_compensated_attempt_is_denied_and_must_be_reclaimed(tmp_path: Path) -> None:
    """A compensated memory write is closed, not absent: the retry re-claims it.

    The trailing refusal covers every status the STARTED/COMPLETED/UNKNOWN arms
    do not, so pin it on the one status a compensation produces.
    """

    with _seed(tmp_path / "mem.db") as store:
        ledger = ActionLedger(store, "run_1")
        claim = ledger.claim(
            "mem_write", {}, key="mem:pgvector_main:acme:rec-42", scoped_to_run=False
        )
        ledger.complete(claim.key, external_id="rec-42")
        ledger.compensate(claim.key, note="rolled back")
        decision = _decide(store)
    assert decision.allow is False
    assert "closed (status compensated)" in decision.reason
    assert "continuum_intercept_action" in decision.reason


class _RaisingForeignStore:
    """A store whose index lookup fails mid-call."""

    supports_action_index = True

    def foreign_action(self, key: str, *, exclude_run: str) -> Action | None:
        raise RuntimeError("index is unavailable")


class _ForeignStartedStore:
    """A store whose index reports a claim another run opened."""

    supports_action_index = True

    def foreign_action(self, key: str, *, exclude_run: str) -> Action | None:
        return Action(
            run_id="run_other",
            action_type="mem_write",
            status=ActionStatus.STARTED,
        )


def test_a_foreign_lookup_that_raises_reads_as_no_foreign_claim(tmp_path: Path) -> None:
    """The broad guard around the index lookup denies rather than waves through.

    A storage that raises mid-lookup keeps the gate's answer the same as one
    with no index at all: the call is unclaimed, so it is denied with the claim
    instructions, never allowed by the failure.
    """

    with _seed(tmp_path / "mem.db") as store:
        with pytest.raises(RuntimeError):
            _RaisingForeignStore().foreign_action("k", exclude_run="run_1")
        decision = _decide(store, storage=_RaisingForeignStore())
    assert decision.allow is False
    assert "has no ledger claim" in decision.reason


def test_a_foreign_started_claim_is_denied_not_adopted(tmp_path: Path) -> None:
    """A STARTED claim from another run is a concurrent write, not a live claim."""

    with _seed(tmp_path / "mem.db") as store:
        decision = _decide(store, storage=_ForeignStartedStore())
    assert decision.allow is False
    assert "in another run (started)" in decision.reason


def test_a_config_error_denies_rather_than_waves_the_call_through() -> None:
    """A template the call cannot satisfy is a configuration problem: deny.

    The gate names the problem as a configuration error rather than falling
    through to "no gate configured", which would let a broken registry pass
    every call it was meant to gate.
    """

    decision = decide(
        CONFIG["tools"],
        "pgvector.upsert",
        {"store_id": "pgvector_main", "tenant": "acme"},  # no record_key
        run_id="run_1",
        actions_by_key={},
    )
    assert decision.allow is False
    assert decision.reason.startswith("gate configuration error:")
    assert "record_key" in decision.reason


# --- the memory-key parser -------------------------------------------------- #


def test_split_memory_key_keeps_the_record_key_verbatim() -> None:
    """Both documented shapes carry the tenant third and the record key whole."""

    # mem:{store_id}:{tenant}:{record_key...}
    assert split_memory_key("mem:store:tenant:record") == ("tenant", "record")
    assert split_memory_key("mem:store:tenant:a:b:c") == ("tenant", "a:b:c")
    # memory:{store_id}:{tenant}:{namespace}:{record_key...}
    assert split_memory_key("memory:store:tenant:ns:doc:section:1") == (
        "tenant",
        "doc:section:1",
    )


def test_split_memory_key_returns_none_for_unattributable_keys() -> None:
    """A key that cannot place a tenant belongs to nobody, so to nobody it falls."""

    for unattributable in (
        "mem:store:tenant",  # too short even for the mem: form
        "memory:store:tenant:ns",  # too short for the memory: form
        "mem:store",  # memory-prefixed but no tenant segment at all
        "plain:store:tenant:record",  # not a memory key
    ):
        assert split_memory_key(unattributable) is None, unattributable


def test_derive_memory_key_rejects_a_non_memory_render() -> None:
    """A template that renders off the mem: convention is a config error."""

    with pytest.raises(GateConfigError, match="memory key must start with 'mem:'"):
        derive_memory_key({"a": "x"}, template="plain:{a}")


# --- the shared denial reason and the config reader ------------------------- #


def test_consumed_authority_reason_matches_across_the_event_forms() -> None:
    """The dict form a caller uses must render the same message as the event form."""

    ev = SimpleNamespace(sequence=7, payload={"consumer_run_id": "run-x"})
    from_dict = consumed_authority_reason(
        "auth-1", {"sequence": 7, "payload": {"consumer_run_id": "run-x"}}
    )
    assert from_dict == consumed_authority_reason("auth-1", ev)
    assert "seq 7" in from_dict
    assert "'run-x'" in from_dict


def test_load_gate_config_rejects_a_non_dict_tools_object(tmp_path: Path) -> None:
    """The registry must be {'tools': {...}}; anything else fails closed.

    A top level that is not an object, or a ``tools`` value that is not a
    mapping, both miss the shape the reader promises to return.
    """

    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"tools": "not-a-mapping"}), encoding="utf-8")
    with pytest.raises(GateConfigError, match=r"expected \{'tools'"):
        load_gate_config(path)

    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(GateConfigError, match=r"expected \{'tools'"):
        load_gate_config(path)


def test_load_gate_config_rejects_a_non_string_action_type(tmp_path: Path) -> None:
    """An action_type that is not a string cannot name the claim to check."""

    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps({"tools": {"t": {"key_template": "k:{a}", "action_type": 5}}}),
        encoding="utf-8",
    )
    with pytest.raises(GateConfigError, match="'action_type' must be a string"):
        load_gate_config(path)
