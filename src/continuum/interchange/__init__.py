"""A stable, versioned JSON interchange format for CONTINUUM's durable output.

CONTINUUM records everything as an event log and projects durable state from it.
This package turns that durable output into a *portable* representation an
external tool can read, verify, and re-import without embedding Python or the
CONTINUUM core:

* ``SemanticState`` -- the compact, crash-surviving projection of a run.
* ``RecoveryContract`` -- what recovery permitted and what it gated on.
* ``RecoveryDecision`` -- the engine's verdict plus its full justification.

None of this is a new storage backend. It is a serialization boundary with a
versioned envelope, so an interchange file written today keeps meaning when the
models evolve, and an unrecognized version fails loudly instead of silently
mis-parsing.

Conventions
-----------
* Every payload is wrapped in an envelope carrying ``schema``, ``kind``,
  ``version`` and ``generated_at``. The ``data`` field holds the payload proper.
* The on-disk shape is plain JSON produced by pydantic's ``model_dump``; import
  re-validates through the same models, so a malformed or drifted payload raises
  instead of being trusted.
* No third-party dependencies beyond what the core already uses (pydantic). The
  published JSON schema is generated from the same models, so it cannot drift
  from the code that produces it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from continuum.recovery.engine import RecoveryDecision

from continuum.checkpoint.manager import RestoredRun
from continuum.environment.diff import EnvironmentDiff
from continuum.models import (
    Action,
    RecoveryContract,
    RecoveryMode,
    SemanticState,
    StateCheckpoint,
    StateValidationResult,
)
from continuum.recovery.planner import RepairPlan
from continuum.state.validator import ValidationOutcome

__all__ = [
    "INTERCHANGE_VERSION",
    "SCHEMA_ID",
    "InterchangeError",
    "InterchangeVersionError",
    "ValidationDocument",
    "RestoredRunDocument",
    "RecoveryDecisionDocument",
    "export_semantic_state",
    "import_semantic_state",
    "export_recovery_contract",
    "import_recovery_contract",
    "export_recovery_decision",
    "import_recovery_decision",
    "published_schema",
    "dump_payload",
    "load_payload",
]

#: The interchange format version this build speaks. Bump only when an older
#: payload can no longer be read from the current models.
INTERCHANGE_VERSION = 1

#: The schema identifier carried in every envelope.
SCHEMA_ID = "continuum.interchange/v1"

_KIND_TO_MODEL = {
    "semantic_state": SemanticState,
    "recovery_contract": RecoveryContract,
    "recovery_decision": "RecoveryDecisionDocument",
}


class InterchangeError(ValueError):
    """Base class for interchange failures."""


class InterchangeVersionError(InterchangeError):
    """Raised when a payload asks for a version this build cannot honor."""


# --------------------------------------------------------------------------- #
# Envelope + validation
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _envelope(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_ID,
        "kind": kind,
        "version": INTERCHANGE_VERSION,
        "generated_at": _now(),
        "data": data,
    }


def _unwrap(payload: dict[str, Any], expected_kind: str) -> Any:
    """Validate the envelope and return its ``data`` field.

    Raises ``InterchangeError`` on a malformed envelope and
    ``InterchangeVersionError`` when the version is unsupported, so a stale or
    forged payload is never silently trusted.
    """
    if not isinstance(payload, dict):
        raise InterchangeError("interchange payload must be a JSON object")
    if payload.get("schema") != SCHEMA_ID:
        raise InterchangeError(
            f"unexpected schema {payload.get('schema')!r}; expected {SCHEMA_ID!r}"
        )
    kind = payload.get("kind")
    if kind != expected_kind:
        raise InterchangeError(f"expected kind {expected_kind!r}, found {kind!r}")
    version = payload.get("version")
    if version != INTERCHANGE_VERSION:
        raise InterchangeVersionError(
            f"unsupported interchange version {version!r}; "
            f"this build speaks version {INTERCHANGE_VERSION}"
        )
    if "data" not in payload:
        raise InterchangeError("payload is missing its 'data' field")
    return cast(dict[str, Any], payload["data"])


# --------------------------------------------------------------------------- #
# RecoveryDecision document (the only non-pydantic kind, mirrored for the wire)
# --------------------------------------------------------------------------- #


class ValidationDocument(BaseModel):
    """The validated state plus the report that justified every downgrade."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: SemanticState
    report: StateValidationResult
    environment_diff: EnvironmentDiff


class RestoredRunDocument(BaseModel):
    """How much of the run recovery managed to reconstruct."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    state: SemanticState
    checkpoint: StateCheckpoint | None = None
    pending_events: int = 0
    replayed: bool = False


class RecoveryDecisionDocument(BaseModel):
    """A self-describing, serializable view of a ``RecoveryDecision``.

    ``RecoveryDecision`` is a frozen dataclass whose fields are all pydantic
    models, so this document round-trips back into the original decision without
    loss. It exists so the decision has a stable, schema-validated wire shape.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    mode: RecoveryMode
    contract: RecoveryContract
    plan: RepairPlan
    validation: ValidationDocument
    restored: RestoredRunDocument
    uncertain_actions: list[Action] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    #: The files the engine's assessment touched or produced. A frozenset on the
    #: decision; the wire form is a sorted list so the payload is deterministic.
    impacted_files: list[str] = Field(default_factory=list)
    #: Summary of the tail of the log the decision was based on (#265).
    tail_evidence: str | None = None
    #: Informed-retry block (#265): a pure projection of prior recovery-path
    #: events plus current failure signals, or None when there is no history.
    informed_retry: dict[str, Any] | None = None


def _decision_to_document(decision: RecoveryDecision) -> RecoveryDecisionDocument:
    return RecoveryDecisionDocument(
        run_id=decision.run_id,
        mode=decision.mode,
        contract=decision.contract,
        plan=decision.plan,
        validation=ValidationDocument(
            state=decision.validation.state,
            report=decision.validation.report,
            environment_diff=decision.validation.environment_diff,
        ),
        restored=RestoredRunDocument(
            run_id=decision.restored.run_id,
            state=decision.restored.state,
            checkpoint=decision.restored.checkpoint,
            pending_events=decision.restored.pending_events,
            replayed=decision.restored.replayed,
        ),
        uncertain_actions=list(decision.uncertain_actions),
        rationale=list(decision.rationale),
        impacted_files=sorted(decision.impacted_files),
        tail_evidence=decision.tail_evidence,
        informed_retry=decision.informed_retry,
    )


def _document_to_decision(document: RecoveryDecisionDocument) -> RecoveryDecision:
    from continuum.recovery.engine import RecoveryDecision

    validation = ValidationOutcome(
        state=document.validation.state,
        report=document.validation.report,
        environment_diff=document.validation.environment_diff,
    )
    restored = RestoredRun(
        run_id=document.restored.run_id,
        state=document.restored.state,
        checkpoint=document.restored.checkpoint,
        pending_events=document.restored.pending_events,
        replayed=document.restored.replayed,
    )
    return RecoveryDecision(
        run_id=document.run_id,
        mode=document.mode,
        contract=document.contract,
        plan=document.plan,
        validation=validation,
        restored=restored,
        uncertain_actions=tuple(document.uncertain_actions),
        rationale=tuple(document.rationale),
        impacted_files=frozenset(document.impacted_files),
        tail_evidence=document.tail_evidence,
        informed_retry=document.informed_retry,
    )


# --------------------------------------------------------------------------- #
# Public export / import
# --------------------------------------------------------------------------- #


def export_semantic_state(state: SemanticState) -> dict[str, Any]:
    """Serialize a ``SemanticState`` to a versioned interchange envelope."""
    return _envelope("semantic_state", state.model_dump(mode="json"))


def import_semantic_state(payload: dict[str, Any]) -> SemanticState:
    """Parse a ``SemanticState`` envelope, re-validating through the model."""
    data = _unwrap(payload, "semantic_state")
    return SemanticState.model_validate(data)


def export_recovery_contract(contract: RecoveryContract) -> dict[str, Any]:
    """Serialize a ``RecoveryContract`` to a versioned interchange envelope."""
    return _envelope("recovery_contract", contract.model_dump(mode="json"))


def import_recovery_contract(payload: dict[str, Any]) -> RecoveryContract:
    """Parse a ``RecoveryContract`` envelope, re-validating through the model."""
    data = _unwrap(payload, "recovery_contract")
    return RecoveryContract.model_validate(data)


def export_recovery_decision(decision: RecoveryDecision) -> dict[str, Any]:
    """Serialize a ``RecoveryDecision`` to a versioned interchange envelope."""
    return _envelope(
        "recovery_decision",
        _decision_to_document(decision).model_dump(mode="json"),
    )


def import_recovery_decision(payload: dict[str, Any]) -> RecoveryDecision:
    """Parse a ``RecoveryDecision`` envelope back into a ``RecoveryDecision``.

    The envelope is re-validated through ``RecoveryDecisionDocument`` first, so a
    drifted or partial payload raises rather than yielding a half-built decision.
    """
    data = _unwrap(payload, "recovery_decision")
    document = RecoveryDecisionDocument.model_validate(data)
    return _document_to_decision(document)


# --------------------------------------------------------------------------- #
# Published schema + file helpers
# --------------------------------------------------------------------------- #


def published_schema(kind: str) -> dict[str, Any]:
    """The JSON Schema an external verifier can check a payload against.

    Generated from the same pydantic models that produce the payload, so the
    schema cannot silently drift from the code. The ``data`` field of an
    envelope validates against the schema returned for the matching ``kind``.
    """
    if kind == "semantic_state":
        return SemanticState.model_json_schema()
    if kind == "recovery_contract":
        return RecoveryContract.model_json_schema()
    if kind == "recovery_decision":
        return RecoveryDecisionDocument.model_json_schema()
    raise InterchangeError(f"no published schema for kind {kind!r}")


def dump_payload(payload: dict[str, Any], path: str | Path) -> None:
    """Write an interchange payload to ``path`` as JSON."""
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_payload(path: str | Path) -> dict[str, Any]:
    """Read an interchange payload from ``path``."""
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
