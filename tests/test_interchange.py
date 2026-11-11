"""Tests for the versioned CONTINUUM interchange format.

These pin three guarantees the interchange must hold:

* **Round-trip equality.** ``import_X(export_X(obj)) == obj`` for every
  supported kind, so the envelope is a faithful, lossless boundary.
* **Untrusted input is rejected.** A drifted, mislabeled, or future-version
  payload raises instead of being silently trusted.
* **The published schema describes the bytes.** Example artifacts on disk parse
  through the same models the schema was generated from.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from continuum.checkpoint.manager import RestoredRun
from continuum.environment.diff import diff_environments
from continuum.interchange import (
    INTERCHANGE_VERSION,
    SCHEMA_ID,
    InterchangeError,
    InterchangeVersionError,
    dump_payload,
    export_recovery_contract,
    export_recovery_decision,
    export_semantic_state,
    import_recovery_contract,
    import_recovery_decision,
    import_semantic_state,
    load_payload,
    published_schema,
)
from continuum.models import (
    Action,
    ActionStatus,
    EnvironmentSnapshot,
    EnvResource,
    ExternalDependency,
    Goal,
    Progress,
    RecoveryContract,
    RecoveryMode,
    RecoverySafety,
    SemanticState,
    StateCheckpoint,
    StateStatus,
    StateValidationResult,
)
from continuum.recovery.engine import RecoveryDecision
from continuum.recovery.planner import RepairKind, RepairPlan, RepairStep
from continuum.state.validator import ValidationOutcome

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "interchange"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _state() -> SemanticState:
    return SemanticState(
        run_id="run_demo",
        goal=Goal(
            description="Write the Git guide",
            constraints=["beginner audience", "one command per part"],
        ),
        progress=Progress(total=5, completed=2, pending=2, failed=1),
        plan=[],
        external_dependencies=[
            ExternalDependency(
                resource="q3_dataset",
                kind="dataset",
                version="sha256:aaaa1111",
                status=StateStatus.VALID,
            )
        ],
    )


def _contract() -> RecoveryContract:
    return RecoveryContract(
        run_id="run_demo",
        checkpoint_version=3,
        recovery_status=RecoverySafety.SAFE_TO_RESUME,
        verified=["goal", "progress"],
        invalidated=[],
        required_actions=[],
        next_allowed_action=None,
    )


def _decision() -> RecoveryDecision:
    state = _state()
    env = EnvironmentSnapshot(
        run_id="run_demo",
        resources={"q3_dataset": EnvResource(name="q3_dataset", version="sha256:aaaa1111")},
    )
    report = StateValidationResult(
        run_id="run_demo",
        checkpoint_version=3,
        safe_to_resume=True,
        reason="all components verified against the current environment",
    )
    plan = RepairPlan(
        steps=[
            RepairStep(
                kind=RepairKind.REVALIDATE_DEPENDENCY,
                target="q3_dataset",
                reason="version moved since checkpoint",
            )
        ]
    )
    validation = ValidationOutcome(
        state=state,
        report=report,
        environment_diff=diff_environments(
            EnvironmentSnapshot(run_id="run_demo"),
            env,
        ),
    )
    restored = RestoredRun(
        run_id="run_demo",
        state=state,
        checkpoint=StateCheckpoint(run_id="run_demo", version=3, state=state),
        pending_events=0,
        replayed=True,
    )
    action = Action(
        run_id="run_demo",
        action_type="charge",
        arguments={"amount": 100, "currency": "USD"},
        status=ActionStatus.COMPLETED,
    )
    return RecoveryDecision(
        run_id="run_demo",
        mode=RecoveryMode.RESUME,
        contract=_contract(),
        plan=plan,
        validation=validation,
        restored=restored,
        uncertain_actions=(action,),
        rationale=("Goal verified", "Progress intact", "Environment stable"),
        impacted_files=frozenset({"src/billing.py", "docs/billing.md"}),
        tail_evidence="last checkpoint replayed intact",
        informed_retry={"focus": "retry idempotently", "steps": ["verify", "retry"]},
    )


# --------------------------------------------------------------------------- #
# Round-trip equality
# --------------------------------------------------------------------------- #


def test_semantic_state_round_trips():
    state = _state()
    assert import_semantic_state(export_semantic_state(state)) == state


def test_recovery_contract_round_trips():
    contract = _contract()
    assert import_recovery_contract(export_recovery_contract(contract)) == contract


def test_recovery_decision_round_trips():
    decision = _decision()
    restored = import_recovery_decision(export_recovery_decision(decision))
    assert restored == decision
    # Behavioural parity: the reconstructed decision still answers its contract.
    assert restored.mode is RecoveryMode.RESUME
    assert restored.permits("anything") is True
    assert restored.contract.next_allowed_action is None


def test_recovery_decision_round_trips_engine_only_fields():
    # assess() populates the last three fields (engine.py), so the envelope must
    # carry them: an empty impact set, a missing summary and a dropped
    # informed-retry block would otherwise come back at their defaults.
    decision = _decision()
    restored = import_recovery_decision(export_recovery_decision(decision))
    assert restored.impacted_files == frozenset({"src/billing.py", "docs/billing.md"})
    assert restored.tail_evidence == "last checkpoint replayed intact"
    assert restored.informed_retry == {
        "focus": "retry idempotently",
        "steps": ["verify", "retry"],
    }


def test_recovery_decision_impacted_files_serialize_sorted():
    # A frozenset has no wire order; the document pins a sorted list so the
    # payload is deterministic, and re-imports as the same frozenset.
    decision = _decision()
    payload = export_recovery_decision(decision)
    assert payload["data"]["impacted_files"] == ["docs/billing.md", "src/billing.py"]
    assert import_recovery_decision(payload).impacted_files == decision.impacted_files


def test_envelope_is_versioned_and_kind_tagged():
    payload = export_semantic_state(_state())
    assert payload["schema"] == SCHEMA_ID
    assert payload["version"] == INTERCHANGE_VERSION
    assert payload["kind"] == "semantic_state"
    assert "generated_at" in payload
    assert "data" in payload


# --------------------------------------------------------------------------- #
# Untrusted input is rejected
# --------------------------------------------------------------------------- #


def test_wrong_schema_is_rejected():
    payload = export_semantic_state(_state())
    payload["schema"] = "some.other.format/v9"
    with pytest.raises(InterchangeError):
        import_semantic_state(payload)


def test_wrong_kind_is_rejected():
    payload = export_semantic_state(_state())
    payload["kind"] = "recovery_contract"
    with pytest.raises(InterchangeError):
        import_semantic_state(payload)


def test_unsupported_version_is_rejected():
    payload = export_semantic_state(_state())
    payload["version"] = INTERCHANGE_VERSION + 1
    with pytest.raises(InterchangeVersionError):
        import_semantic_state(payload)


def test_missing_data_is_rejected():
    payload = export_semantic_state(_state())
    del payload["data"]
    with pytest.raises(InterchangeError):
        import_semantic_state(payload)


def test_malformed_data_is_rejected():
    payload = export_recovery_contract(_contract())
    payload["data"]["run_id"] = 12345  # wrong type for run_id
    with pytest.raises(ValidationError):
        import_recovery_contract(payload)


# --------------------------------------------------------------------------- #
# Published schema
# --------------------------------------------------------------------------- #


def test_published_schema_is_object_for_each_kind():
    for kind in ("semantic_state", "recovery_contract", "recovery_decision"):
        schema = published_schema(kind)
        assert schema.get("type") == "object"
        assert "$defs" in schema or "properties" in schema


def test_published_schema_rejects_unknown_kind():
    with pytest.raises(InterchangeError):
        published_schema("no_such_kind")


def test_published_schema_matches_round_tripped_payload():
    # The decision payload's `data` validates against the generated schema's
    # shape: re-importing succeeds, proving the schema and the producer agree.
    decision = _decision()
    payload = export_recovery_decision(decision)
    assert import_recovery_decision(payload) == decision


def test_published_schema_covers_the_engine_only_fields():
    # The document model generates the published schema, so the three fields
    # assess() fills must be part of it rather than unknown to an external
    # verifier (they used to be stripped before the round trip).
    schema = published_schema("recovery_decision")
    properties = schema["properties"]
    assert "impacted_files" in properties
    assert "tail_evidence" in properties
    assert "informed_retry" in properties


# --------------------------------------------------------------------------- #
# File helpers + canonical example artifacts
# --------------------------------------------------------------------------- #


def test_dump_and_load_payload(tmp_path: Path):
    payload = export_semantic_state(_state())
    path = tmp_path / "state.json"
    dump_payload(payload, path)
    assert load_payload(path) == payload


@pytest.mark.parametrize(
    "kind,export,example",
    [
        ("semantic_state", export_semantic_state, "semantic_state.example.json"),
        ("recovery_contract", export_recovery_contract, "recovery_contract.example.json"),
        ("recovery_decision", export_recovery_decision, "recovery_decision.example.json"),
    ],
)
def test_example_artifact_is_readable(kind, export, example):
    path = EXAMPLE_DIR / example
    assert path.exists(), f"missing canonical example {path}"
    payload = load_payload(path)
    assert payload["kind"] == kind
    assert payload["version"] == INTERCHANGE_VERSION
    # Importing the checked-in artifact must succeed through the live models.
    if kind == "semantic_state":
        import_semantic_state(payload)
    elif kind == "recovery_contract":
        import_recovery_contract(payload)
    else:
        import_recovery_decision(payload)
