"""The recovery ledger's compaction, reconcile arms, and budget autoloads (issue #2058).

The ledger is the sealed, tamper-evident record a resume decision is made
against, so the arms pinned here are the ones that decide what a resume looks
like: compaction rewriting (never appending) the file, the drift details
reconcile reports, the dependency merge loops, the default budget registry the
three autoload sites share, and the base backend's declines.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from continuum.models import RecoveryContract, RecoverySafety
from continuum.recovery import (
    FileLedgerBackend,
    LedgerEntryKind,
    MemoryLedgerBackend,
    RecoveryLedger,
    RecoveryLedgerEntry,
)
from continuum.recovery.ledger import (
    LedgerBackend,
    _scoped_limit,
    dependencies_for_action,
    derive_recovery_dependencies,
)


def _contract(
    version: int = 0, status: RecoverySafety = RecoverySafety.REQUIRES_REPAIR
) -> RecoveryContract:
    return RecoveryContract(run_id="run_1", checkpoint_version=version, recovery_status=status)


def _seed_file_ledger(directory: Path, count: int, run_id: str = "run_1") -> RecoveryLedger:
    ledger = RecoveryLedger(FileLedgerBackend(str(directory)))
    for version in range(count):
        ledger.append_decision(run_id, _contract(version))
    return ledger


@pytest.fixture
def ledger() -> Iterator[RecoveryLedger]:
    yield RecoveryLedger(MemoryLedgerBackend())


# --- compaction on the file backend ---------------------------------------- #


def test_compact_rewrites_so_entries_never_duplicate(tmp_path: Path) -> None:
    """Sealed compact rewrites the run's file: appending would double its history."""

    ledger = _seed_file_ledger(tmp_path, 60)
    ledger_file = tmp_path / "ledger-run_1.jsonl"

    removed = ledger.compact("run_1", keep=10)
    assert removed == 50

    raw = ledger_file.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 10, "compaction rewrites, never appends"
    hashes = [json.loads(line)["content_hash"] for line in raw]
    assert len(set(hashes)) == len(hashes), "no entry appears twice after compaction"

    # Survivors keep their original sequence numbers, sparse after compaction.
    assert [e.sequence for e in ledger.entries("run_1")] == list(range(50, 60))
    ok, trusted = ledger.verify("run_1")
    assert ok is True and trusted == 10
    assert ledger.entries("run_1")[0].prev_hash == "genesis"


def test_compact_clamps_keep_and_short_circuits_a_small_run(tmp_path: Path) -> None:
    """keep < 1 clamps to 1, and a run at or under keep is left alone."""

    ledger = _seed_file_ledger(tmp_path, 5)
    ledger_file = tmp_path / "ledger-run_1.jsonl"

    # A run with fewer entries than keep: nothing is removed or rewritten.
    before = ledger_file.read_text(encoding="utf-8")
    assert ledger.compact("run_1", keep=100) == 0
    assert ledger_file.read_text(encoding="utf-8") == before

    # keep clamps to 1: the newest entry alone survives.
    assert ledger.compact("run_1", keep=0) == 4
    assert len(ledger.entries("run_1")) == 1
    assert ledger.entries("run_1")[0].sequence == 4
    assert ledger.verify("run_1")[0] is True


# --- reconcile's three drift details --------------------------------------- #


def test_reconcile_reports_a_broken_chain(ledger: RecoveryLedger) -> None:
    ledger.append_decision("run_1", _contract(0))
    ledger.append_decision("run_1", _contract(1))

    entries = ledger.entries("run_1")
    ledger._backend.replace("run_1", [replace(entries[0], note="hacked"), entries[1]])

    report = ledger.reconcile("run_1", SimpleNamespace(version=1))
    assert report.drift is True
    assert any("ledger chain broken at index" in d for d in report.details), report.details


def test_reconcile_without_a_decision_reports_nothing_to_reconcile_against(
    ledger: RecoveryLedger,
) -> None:
    """A run with no DECISION entry cannot be compared; the early return says so."""

    report = ledger.reconcile("run_1", SimpleNamespace(version=1))
    assert report.drift is True
    assert report.details == ["no ledger decision to reconcile against"]


def test_reconcile_reports_a_state_that_exposes_no_version(ledger: RecoveryLedger) -> None:
    ledger.append_decision("run_1", _contract(version=5))

    report = ledger.reconcile("run_1", SimpleNamespace())
    assert report.drift is True
    assert "live state exposes no version to compare" in report.details


# --- dependency derivation ------------------------------------------------- #


def test_derive_recovery_dependencies_merges_without_duplicating() -> None:
    """dependency, dependencies and scope merge in order, de-duplicating."""

    merged = derive_recovery_dependencies(
        dependency="a", dependencies=["a", "a", "b"], scope=["b", "c"]
    )
    assert merged == ["a", "b", "c"]


def test_dependencies_for_action_declines_what_carries_no_scope() -> None:
    """A plain action name, and no action at all, carry no external dependency."""

    assert dependencies_for_action("plain-action") == []
    assert dependencies_for_action(None) == []
    # A dep_scope without a comma is a single dependency, not a list to split.
    assert dependencies_for_action({"dep_scope": "only"}) == ["only"]
    assert dependencies_for_action("ext:db") == ["ext:db"]


# --- the base backend's declines ------------------------------------------- #


def test_the_agnostic_backend_declines_every_operation() -> None:
    """load, save and replace are the seam; the base names it as unimplemented."""

    base = LedgerBackend()
    with pytest.raises(NotImplementedError):
        base.load("run_1")
    with pytest.raises(NotImplementedError):
        base.save(
            RecoveryLedgerEntry(
                entry_id="e1",
                run_id="run_1",
                sequence=0,
                prev_hash="genesis",
                content_hash="h",
                kind="decision",
                contract=None,
                gate=None,
                anchor=False,
                created_at=datetime(2026, 1, 1),
            )
        )
    with pytest.raises(NotImplementedError):
        base.replace("run_1", [])


# --- the scoped limit fallback --------------------------------------------- #


def test_scoped_limit_falls_back_to_the_callers_ceiling() -> None:
    """Without a registry the caller's limit stands; the run-wide default applies."""

    assert _scoped_limit(None, 5, None) == 5
    assert _scoped_limit(None, 5, {"default_max_attempts": 9}) == 9
    assert _scoped_limit(None, 5, {"default_max_attempts": True}) == 5, "booleans are not ceilings"


# --- pending gate and last decision ---------------------------------------- #


def test_pending_gate_and_last_decision_answer_none_without_data(ledger: RecoveryLedger) -> None:
    """A run with no required decision and no decision at all answers None."""

    assert ledger.pending_gate("run_1") is None
    assert ledger.last_decision("run_1") is None

    # A required decision that a later gate approved is no longer pending.
    ledger.append_decision("run_1", _contract(), gate="required")
    ledger.append_decision("run_1", _contract())  # a later, unrequired decision
    ledger.record_gate("run_1", "approved")
    assert ledger.pending_gate("run_1") is None
    assert ledger.last_decision("run_1").kind == LedgerEntryKind.DECISION.value  # type: ignore[union-attr]


# --- the default-budgets autoload at all three sites ----------------------- #


def _point_at(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr("continuum.budgets.DEFAULT_BUDGETS_PATH", str(path))


def test_the_default_budgets_registry_is_autoloaded_where_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three autoload sites read the same registry and honour its limit."""

    registry = tmp_path / "budgets.json"
    registry.write_text(json.dumps({"dependency_budgets": {"dep": 1}}))
    _point_at(monkeypatch, registry)

    ledger = RecoveryLedger(MemoryLedgerBackend())
    # record_attempt: the budgeted ceiling of 1 escalates on the first attempt.
    ledger.record_attempt("run_1", dependency="dep")
    assert ledger.requires_human("run_1", dependency="dep") is True
    assert ledger.budget("run_1", dependency="dep", max_attempts=5).max_attempts == 1


def test_a_missing_registry_falls_back_uniformly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No registry on disk: every site keeps the caller's ceiling."""

    _point_at(monkeypatch, tmp_path / "absent.json")

    ledger = RecoveryLedger(MemoryLedgerBackend())
    ledger.record_attempt("run_1", dependency="dep")
    ledger.record_attempt("run_1", dependency="dep")
    assert ledger.requires_human("run_1", dependency="dep", max_attempts=5) is False
    assert ledger.budget("run_1", dependency="dep", max_attempts=5).max_attempts == 5


def test_an_unreadable_registry_is_swallowed_uniformly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed registry is swallowed by every site, not raised past one of them."""

    registry = tmp_path / "budgets.json"
    registry.write_text("{not json", encoding="utf-8")
    _point_at(monkeypatch, registry)

    ledger = RecoveryLedger(MemoryLedgerBackend())
    ledger.record_attempt("run_1", dependency="dep")
    assert ledger.requires_human("run_1", dependency="dep", max_attempts=5) is False
    assert ledger.budget("run_1", dependency="dep", max_attempts=5).max_attempts == 5
