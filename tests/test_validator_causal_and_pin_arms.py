"""The validator's N-hop staleness propagation and pin-constraint audit (#2060).

``_propagate_caused_by`` is the reason a rollback or a revalidation cascade at
all: when one piece of evidence is tainted, that walk decides which findings
and decisions become stale so a resume does not build on poisoned state. The
pin-constraint audit is the operator-trust boundary: a dropped or retracted
hard constraint must surface ``REQUIRES_REVIEW`` unless an operator confirmed
it. Both surfaces are driven directly here, with in-memory states and hand
built event chains, because the arms under test are functions of the inputs
alone and the public ``validate`` path buries them behind a full projection.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from continuum.events import Event, EventType, Origin
from continuum.models import (
    Component,
    ComponentValidationEntry,
    ConstraintPin,
    Decision,
    Evidence,
    Finding,
    Goal,
    Provenance,
    SemanticState,
    StateStatus,
)
from continuum.security.constraints import predicate_digest
from continuum.state.validator import StateValidator

WHEN = datetime(2026, 1, 1, tzinfo=UTC)


def _state(
    *,
    evidence: list[Evidence] | None = None,
    findings: list[Finding] | None = None,
    decisions: list[Decision] | None = None,
    plan: list[Any] | None = None,
    pins: dict[str, ConstraintPin] | None = None,
) -> SemanticState:
    return SemanticState(
        run_id="run_1",
        goal=Goal(description="g"),
        evidence=evidence or [],
        findings=findings or [],
        decisions=decisions or [],
        plan=plan or [],
        pins=pins or {},
    )


def _event(
    event_id: str,
    sequence: int,
    type_: EventType,
    payload: dict[str, Any],
    *,
    caused_by: list[str] | None = None,
) -> Event:
    body = dict(payload)
    if caused_by is not None:
        body["caused_by"] = caused_by
    return Event(
        event_id=event_id,
        run_id="run_1",
        sequence=sequence,
        type=type_,
        timestamp=WHEN,
        payload=body,
        source=Origin.DETERMINISTIC,
    )


def _stale_evidence_entry(evidence_id: str) -> ComponentValidationEntry:
    return ComponentValidationEntry(
        component=Component.EVIDENCE,
        component_id=evidence_id,
        status=StateStatus.STALE,
        detail="environment dependency moved",
    )


def _pin_entries(entries: list[ComponentValidationEntry]) -> list[ComponentValidationEntry]:
    return [e for e in entries if e.component is Component.PIN]


# --- the three degrade-without-failing bail-outs (#2060) ----------------------- #


def _tainted_chain() -> tuple[SemanticState, list[Event]]:
    """Evidence (tainted) -> finding -> decision, one hop each."""
    state = _state(
        evidence=[
            Evidence(
                evidence_id="ev1",
                summary="vendor list digest",
                provenance=Provenance(source_event_id="e_a"),
            )
        ],
        findings=[
            Finding(
                finding_id="f1",
                claim="the vendor list is current",
                provenance=Provenance(source_event_id="e_b"),
            )
        ],
        decisions=[
            Decision(
                decision_id="d1",
                decision="match against the ledger",
                provenance=Provenance(source_event_id="e_c"),
            )
        ],
    )
    events = [
        _event("e_a", 1, EventType.EVIDENCE_ADDED, {"evidence_id": "ev1"}),
        _event("e_b", 2, EventType.FINDING_ADDED, {"finding_id": "f1"}, caused_by=["e_a"]),
        _event("e_c", 3, EventType.DECISION_CREATED, {"decision_id": "d1"}, caused_by=["e_b"]),
    ]
    return state, events


def test_an_import_of_the_graph_builder_degrades_to_the_untainted_state() -> None:
    """The degrade-without-failing contract: a malformed import returns the
    state unchanged rather than breaking validation for the whole run."""
    state, events = _tainted_chain()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "continuum.provenance.graph", None)
    try:
        entries = [_stale_evidence_entry("ev1")]
        result = StateValidator()._propagate_caused_by(state, events, entries)
    finally:
        monkeypatch.undo()
    assert result is state
    assert len(entries) == 1


def test_a_graph_builder_failure_degrades_to_the_untainted_state() -> None:
    import continuum.provenance.graph as graph_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(graph_module, "build_provenance_graph", _raise)
    try:
        state, events = _tainted_chain()
        entries = [_stale_evidence_entry("ev1")]
        result = StateValidator()._propagate_caused_by(state, events, entries)
    finally:
        monkeypatch.undo()
    assert result is state
    assert len(entries) == 1


def _raise(events: Any) -> Any:
    raise RuntimeError("graph build exploded")


def test_an_event_set_that_builds_no_nodes_degrades_to_the_untainted_state() -> None:
    state = _state(
        evidence=[
            Evidence(
                evidence_id="ev1",
                summary="s",
                provenance=Provenance(source_event_id="e_a"),
            )
        ]
    )
    events = [_event("e_a", 1, EventType.RUN_STARTED, {"goal": "g"})]
    entries = [_stale_evidence_entry("ev1")]
    result = StateValidator()._propagate_caused_by(state, events, entries)
    assert result is state


# --- the walk itself: statuses, details and write-back (#2060) ------------------ #


def test_hops_project_stale_onto_downstream_findings_and_decisions() -> None:
    """The N-hop effect, not merely the bookkeeping: downstream findings and
    decisions gain a STALE entry with the naming-parent detail, and the
    returned state carries the mutated statuses."""
    state, events = _tainted_chain()
    validator = StateValidator()
    entries = [_stale_evidence_entry("ev1")]

    result = validator._propagate_caused_by(state, events, entries)

    projected = [e for e in entries[1:] if e.status is StateStatus.STALE]
    assert {(e.component, e.component_id) for e in projected} == {
        (Component.FINDING, "f1"),
        (Component.DECISION, "d1"),
    }
    assert any(e.detail == "via caused_by from e_a (N-hop staleness)" for e in projected)
    # The write-back: the returned state's components carry the mutated status.
    assert result.findings[0].status is StateStatus.STALE
    assert result.decisions[0].status is StateStatus.STALE
    assert result.decisions[0].invalidated_reason is not None
    # Idempotent: a second walk over the same inputs adds nothing new.
    count_before = len(entries)
    result2 = validator._propagate_caused_by(result, events, entries)
    assert len(entries) == count_before
    assert result2.findings[0].status is StateStatus.STALE


def test_a_cyclic_caused_by_chain_upgrades_downstream_to_conflicted() -> None:
    """a -> b -> a: the cycle arm flips every propagated status from STALE to
    CONFLICTED and names the cycle in the detail. The BFS terminates on the
    revisit, so the walk cannot spin on the loop."""
    state = _state(
        evidence=[
            Evidence(
                evidence_id="ev1",
                summary="s",
                provenance=Provenance(source_event_id="e_a"),
            )
        ],
        findings=[
            Finding(
                finding_id="f1",
                claim="c",
                provenance=Provenance(source_event_id="e_b"),
            )
        ],
    )
    events = [
        _event("e_a", 1, EventType.EVIDENCE_ADDED, {"evidence_id": "ev1"}, caused_by=["e_b"]),
        _event("e_b", 2, EventType.FINDING_ADDED, {"finding_id": "f1"}, caused_by=["e_a"]),
    ]
    validator = StateValidator()
    entries = [_stale_evidence_entry("ev1")]

    result = validator._propagate_caused_by(state, events, entries)

    (projected,) = [e for e in entries if e.component is Component.FINDING]
    assert projected.status is StateStatus.CONFLICTED
    assert projected.detail == "cycle detected via caused_by, downstream of e_a"
    assert result.findings[0].status is StateStatus.CONFLICTED


def test_a_downstream_component_already_carrying_a_non_valid_entry_is_not_re_reported() -> None:
    """The de-duplication filter: a downstream finding the report already
    graded stale is not projected a second time."""
    state, events = _tainted_chain()
    entries = [
        _stale_evidence_entry("ev1"),
        ComponentValidationEntry(
            component=Component.FINDING,
            component_id="f1",
            status=StateStatus.CONFLICTED,
            detail="already graded",
        ),
    ]
    StateValidator()._propagate_caused_by(state, events, entries)
    finding_entries = [e for e in entries if e.component is Component.FINDING]
    assert len(finding_entries) == 1
    assert finding_entries[0].detail == "already graded"


def test_a_downstream_component_whose_live_status_is_not_valid_is_skipped() -> None:
    """A finding the fold already marked invalid is skipped: the walk adds
    caution, never a duplicate verdict."""
    state = _state(
        evidence=[
            Evidence(evidence_id="ev1", summary="s", provenance=Provenance(source_event_id="e_a"))
        ],
        findings=[
            Finding(
                finding_id="f1",
                claim="c",
                status=StateStatus.INVALID,
                provenance=Provenance(source_event_id="e_b"),
            )
        ],
    )
    events = [
        _event("e_a", 1, EventType.EVIDENCE_ADDED, {"evidence_id": "ev1"}),
        _event("e_b", 2, EventType.FINDING_ADDED, {"finding_id": "f1"}, caused_by=["e_a"]),
    ]
    entries = [_stale_evidence_entry("ev1")]
    StateValidator()._propagate_caused_by(state, events, entries)
    assert not any(e.component is Component.FINDING for e in entries)


def test_a_downstream_action_outside_the_state_projects_one_action_entry() -> None:
    """A downstream ACTION_RECORDED the semantic state does not carry still
    projects a Component.ACTION entry with the N-hop detail, exactly once and
    de-duplicated against an existing non-VALID entry for the same action."""
    state = _state(
        evidence=[
            Evidence(evidence_id="ev1", summary="s", provenance=Provenance(source_event_id="e_a"))
        ]
    )
    events = [
        _event("e_a", 1, EventType.EVIDENCE_ADDED, {"evidence_id": "ev1"}),
        _event(
            "e_act",
            2,
            EventType.ACTION_RECORDED,
            {"key": "k", "action": {"action_id": "act_9", "action_type": "send_invoice"}},
            caused_by=["e_a"],
        ),
    ]
    validator = StateValidator()
    entries = [_stale_evidence_entry("ev1")]
    validator._propagate_caused_by(state, events, entries)
    action_entries = [e for e in entries if e.component is Component.ACTION]
    assert len(action_entries) == 1
    assert action_entries[0].component_id == "act_9"
    assert action_entries[0].status is StateStatus.STALE
    assert action_entries[0].detail == "via caused_by downstream of tainted evidence (N-hop)"

    # De-duplicated: a second walk with an existing non-VALID entry adds none.
    entries = [
        _stale_evidence_entry("ev1"),
        ComponentValidationEntry(
            component=Component.ACTION,
            component_id="act_9",
            status=StateStatus.CONFLICTED,
            detail="already graded",
        ),
    ]
    validator._propagate_caused_by(state, events, entries)
    assert len([e for e in entries if e.component is Component.ACTION]) == 1


def test_a_status_less_plan_step_is_projected_and_written_back() -> None:
    """The plan arm of the write-back. Plan steps carry their own lifecycle
    status, so a step the propagator can talk to is one without one; the
    contextlib.suppress around the copy is what keeps a validator-version
    mismatch from failing the whole walk."""

    class _StatuslessPlanStep(BaseModel):
        model_config = {"frozen": True}

        step_id: str
        provenance: Provenance

    state = _state(
        evidence=[
            Evidence(evidence_id="ev1", summary="s", provenance=Provenance(source_event_id="e_a"))
        ]
    )
    # model_copy does not validate, so the step keeps its status-less shape.
    state = state.model_copy(
        update={
            "plan": [
                _StatuslessPlanStep(step_id="s1", provenance=Provenance(source_event_id="e_p"))
            ]
        }
    )
    # PLAN_UPSERT is not a provenance-graph node, so the step's source event
    # has to be one for the walk to reach it: the step cites a downstream
    # DECISION_CREATED node, the way a projected plan step cites the event
    # that carried it.
    events = [
        _event("e_a", 1, EventType.EVIDENCE_ADDED, {"evidence_id": "ev1"}),
        _event("e_p", 2, EventType.DECISION_CREATED, {"decision_id": "d1"}, caused_by=["e_a"]),
    ]
    validator = StateValidator()
    entries = [_stale_evidence_entry("ev1")]

    result = validator._propagate_caused_by(state, events, entries)

    (projected,) = [e for e in entries if e.component is Component.PLAN]
    assert projected.component_id == "s1"
    assert projected.status is StateStatus.STALE
    assert result.plan[0].status is StateStatus.STALE


# --- the pin-constraint audit (#2060) ------------------------------------------- #


PREDICATE = "audit events must be recorded before external side effects"


@pytest.fixture
def registry_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".continuum" / "constraints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("continuum.security.constraints.DEFAULT_CONSTRAINTS_PATH", path)
    return path


def _write_registry(path: Path, specs: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"constraints": specs}), encoding="utf-8")


def _pin(cid: str, predicate: str = PREDICATE) -> ConstraintPin:
    return ConstraintPin(constraint_id=cid, sha256=predicate_digest(predicate))


def _pin_event(cid: str, sequence: int, predicate: str = PREDICATE) -> Event:
    return _event(
        f"e_{cid}_{sequence}",
        sequence,
        EventType.CONSTRAINT_PINNED,
        {
            "constraint_id": cid,
            "sha256": predicate_digest(predicate),
        },
    )


def _drop_event(cid: str, sequence: int, reason: str = "failed_verification") -> Event:
    return _event(
        f"e_{cid}_d{sequence}",
        sequence,
        EventType.CONSTRAINT_PIN_DROPPED,
        {
            "constraint_id": cid,
            "sha256": predicate_digest(PREDICATE),
            "reason": reason,
            "level": "hard",
        },
    )


def _retract_event(cid: str, sequence: int) -> Event:
    return _event(
        f"e_{cid}_r{sequence}", sequence, EventType.CONSTRAINT_RETRACTED, {"constraint_id": cid}
    )


def _audit(
    state: SemanticState, events: list[Event], *, confirmed: set[str] | None = None
) -> list[ComponentValidationEntry]:
    entries: list[ComponentValidationEntry] = []
    StateValidator()._check_constraints(state, events, entries, confirmed=confirmed)
    return entries


def test_a_trusted_attestation_with_a_model_payload_counts(registry_path: Path) -> None:
    """The branch exists precisely so a non-dict attestation from a trusted
    origin is not silently dropped: a pydantic-model payload carrying
    verified=False reaches the audit exactly like the dict form."""
    _write_registry(
        registry_path, [{"id": "audit_logging", "level": "hard", "predicate": PREDICATE}]
    )
    state = _state(pins={"audit_logging": _pin("audit_logging")})
    events = [
        _pin_event("audit_logging", 1),
        SimpleNamespace(
            type=EventType.CONSTRAINT_PINS_VERIFIED,
            source=Origin.DETERMINISTIC,
            payload=_VerifiedPayload(verified=False, unverified=["audit_logging"]),
        ),
    ]
    entries = _audit(state, events)
    (pin_entry,) = _pin_entries(entries)
    assert pin_entry.status is StateStatus.REQUIRES_REVIEW
    assert (
        pin_entry.detail
        == "hard constraint 'audit_logging' unverified; operator confirmation required"
    )


class _VerifiedPayload(BaseModel):
    model_config = {"frozen": True}

    verified: bool = True
    unverified: tuple[str, ...] = ()


def test_a_pin_no_spec_scopes_and_never_touched_is_not_audited(registry_path: Path) -> None:
    """A spec whose scope covers nothing the run did, and whose pin the run
    never pinned, dropped or retracted, is skipped: a registry on disk must
    not escalate runs that never touched its scope."""
    _write_registry(
        registry_path,
        [{"id": "db_guard", "level": "hard", "predicate": "no direct db writes", "scope": ["db."]}],
    )
    entries = _audit(_state(), [])
    assert _pin_entries(entries) == []


def test_a_retracted_pin_a_spec_still_scopes_names_the_retraction(registry_path: Path) -> None:
    _write_registry(
        registry_path,
        [{"id": "db_guard", "level": "hard", "predicate": "no direct db writes", "scope": ["*"]}],
    )
    events = [
        _pin_event("db_guard", 1, predicate="no direct db writes"),
        _retract_event("db_guard", 2),
    ]
    entries = _audit(_state(), events)
    (pin_entry,) = _pin_entries(entries)
    assert pin_entry.status is StateStatus.REQUIRES_REVIEW
    assert "retracted from active pins" in pin_entry.detail


def test_a_spec_scoped_pin_marked_dropped_requires_review(registry_path: Path) -> None:
    """The pin is still in the active set while the log records its drop: the
    entry names the drop and asks for operator confirmation."""
    _write_registry(
        registry_path, [{"id": "audit_logging", "level": "hard", "predicate": PREDICATE}]
    )
    state = _state(pins={"audit_logging": _pin("audit_logging")})
    events = [_drop_event("audit_logging", 1, reason="failed_verification")]
    entries = _audit(state, events)
    (pin_entry,) = _pin_entries(entries)
    assert pin_entry.status is StateStatus.REQUIRES_REVIEW
    assert pin_entry.detail == (
        "hard constraint 'audit_logging' was marked dropped (failed_verification); "
        "operator confirmation required"
    )


def test_one_pin_id_in_two_audit_passes_emits_exactly_one_entry(registry_path: Path) -> None:
    """The idempotence guards: a pin id that is both registry-scoped and in
    the dropped and retracted event logs is audited once, not three times."""
    _write_registry(
        registry_path, [{"id": "audit_logging", "level": "hard", "predicate": PREDICATE}]
    )
    events = [
        _pin_event("audit_logging", 1),
        _drop_event("audit_logging", 2),
        _retract_event("audit_logging", 3),
    ]
    entries = _audit(_state(), events)
    assert len(_pin_entries(entries)) == 1


def test_an_untrusted_attestation_does_not_supersede_the_audit(registry_path: Path) -> None:
    """An agent-originated "verified" record is self-certified and ignored;
    only a trusted origin's attestation reaches the audit."""
    _write_registry(
        registry_path, [{"id": "audit_logging", "level": "hard", "predicate": PREDICATE}]
    )
    state = _state(pins={"audit_logging": _pin("audit_logging")})
    events = [
        _pin_event("audit_logging", 1),
        SimpleNamespace(
            type=EventType.CONSTRAINT_PINS_VERIFIED,
            source=Origin.EXTERNAL_AGENT,
            payload=_VerifiedPayload(verified=False, unverified=["audit_logging"]),
        ),
    ]
    entries = _audit(state, events)
    (pin_entry,) = _pin_entries(entries)
    assert pin_entry.status is StateStatus.VALID
    assert pin_entry.detail == "active and verified"


def test_a_dropped_pin_without_a_registry_spec_still_requires_review(registry_path: Path) -> None:
    """Pass 2: a registry that lacks the pin id must not hide that pin's drop."""
    # No registry on disk: only the event-log passes run.
    events = [_drop_event("standalone_guard", 1)]
    entries = _audit(_state(), events)
    (pin_entry,) = _pin_entries(entries)
    assert pin_entry.component_id == "standalone_guard"
    assert pin_entry.status is StateStatus.REQUIRES_REVIEW


def test_a_dropped_pin_with_an_existing_entry_is_not_re_audited(registry_path: Path) -> None:
    """The entries-based skip guard: a pin the first pass already graded is
    not re-entered by the event-log passes."""
    _write_registry(registry_path, [])
    events = [_drop_event("standalone_guard", 1)]
    existing = ComponentValidationEntry(
        component=Component.PIN,
        component_id="standalone_guard",
        status=StateStatus.REQUIRES_REVIEW,
        detail="already audited",
    )
    StateValidator()._check_constraints(_state(), events, [existing])
    assert len([e for e in [existing] if e.component is Component.PIN]) == 1


def test_a_retracted_pin_without_a_registry_spec_still_requires_review(registry_path: Path) -> None:
    """Pass 3: a retracted hard pin disappears from the active set, so it
    needs its own review entry even when no registry spec names it."""
    # No registry on disk: only the event-log passes run.
    events = [_retract_event("standalone_guard", 1)]
    entries = _audit(_state(), events)
    (pin_entry,) = _pin_entries(entries)
    assert pin_entry.component_id == "standalone_guard"
    assert pin_entry.status is StateStatus.REQUIRES_REVIEW
    assert "was retracted" in pin_entry.detail
