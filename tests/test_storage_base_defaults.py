"""Pin the defaults every capability-less storage engine inherits (issue #2057).

``Storage`` is an interface built on inherited defaults rather than pure
abstract methods, so what a store with no index and no archive answers is a
contract the rest of the codebase leans on: compaction callers get "no
archive" instead of an exception, the gate gets ``None`` instead of a raise,
and claim resolution silently becomes a no-op. These tests pin those defaults
against a minimal concrete subclass, so a refactor changes the failure mode of
every engine at once only if it also fails a test here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from continuum.events import Event, EventType, IntegrityReport, Origin
from continuum.models import Run, RunStatus, SemanticState, StateCheckpoint
from continuum.storage.base import (
    CheckpointNotFound,
    ConcurrentWriteError,
    RunNotFound,
    Storage,
    StorageError,
)


class _MinimalStore(Storage):
    """The capability-less case: no index, no archive, no transactions.

    Implements the abstract methods with in-memory dicts so the defaults under
    test are the ones inherited from ``Storage``, not overridden.
    """

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.events: dict[str, list[Event]] = {}
        self.versions: dict[tuple[str, int], SemanticState] = {}
        self.checkpoints: dict[str, StateCheckpoint] = {}
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def create_run(self, run: Run) -> Run:
        if run.run_id in self.runs:
            raise ConcurrentWriteError(f"run {run.run_id!r} already exists")
        self.runs[run.run_id] = run
        self.events.setdefault(run.run_id, [])
        return run

    def create_run_started(self, run: Run, *, source: Origin = Origin.DETERMINISTIC) -> Run:
        self.create_run(run)
        self.append_event(run.run_id, EventType.RUN_STARTED, {"goal": run.goal}, source=source)
        return run

    def get_run(self, run_id: str) -> Run:
        try:
            return self.runs[run_id]
        except KeyError:
            raise RunNotFound(f"run {run_id!r} not found") from None

    def update_run(self, run: Run) -> Run:
        if run.run_id not in self.runs:
            raise RunNotFound(f"run {run.run_id!r} not found")
        self.runs[run.run_id] = run
        return run

    def list_runs(self, *, limit: int | None = None) -> Sequence[Run]:
        runs = list(self.runs.values())
        return runs[:limit] if limit is not None else runs

    def get_active_run(self) -> Run | None:
        terminal = {
            RunStatus.COMPLETED,
            RunStatus.CRASHED,
            RunStatus.ABORTED,
            RunStatus.FAILED,
        }
        candidates = [r for r in self.runs.values() if r.status not in terminal]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.created_at)

    def append_event(
        self,
        run_id: str,
        type: EventType,
        payload: Mapping[str, Any] | None = None,
        *,
        causer_event_id: str | None = None,
        expected_sequence: int | None = None,
        source: Origin = Origin.DETERMINISTIC,
    ) -> Event:
        chain = self.events.setdefault(run_id, [])
        sequence = len(chain) + 1
        if expected_sequence is not None and expected_sequence != sequence:
            raise ConcurrentWriteError("run has already advanced past the expected sequence")
        event = Event(
            run_id=run_id,
            sequence=sequence,
            type=type,
            payload=dict(payload or {}),
            causer_event_id=causer_event_id,
            source=source,
            prev_hash=chain[-1].hash if chain else None,
        )
        event = event.model_copy(update={"hash": _digest(event)})
        chain.append(event)
        return event

    def read_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        upto: int | None = None,
        rehydrate: bool = True,
    ) -> Sequence[Event]:
        top = upto if upto is not None else float("inf")
        return [e for e in self.events.get(run_id, []) if after_sequence < e.sequence <= top]

    def last_sequence(self, run_id: str) -> int:
        chain = self.events.get(run_id, [])
        return chain[-1].sequence if chain else 0

    def verify_events(self, run_id: str, *, deep: bool = False) -> IntegrityReport:
        chain = self.events.get(run_id, [])
        ok = all(e.hash == _digest(e) for e in chain)
        return IntegrityReport(
            ok=ok,
            checked=len(chain),
            trusted_through={run_id: len(chain)} if ok and chain else {},
        )

    def put_version(self, state: SemanticState, *, reason: str = "", force: bool = False) -> int:
        version = max((v for r, v in self.versions if r == state.run_id), default=0) + 1
        self.versions[(state.run_id, version)] = state
        return version

    def get_version(self, run_id: str, version: int) -> SemanticState:
        try:
            return self.versions[(run_id, version)]
        except KeyError:
            raise StorageError(f"no version {version} for run {run_id!r}") from None

    def latest_version(self, run_id: str) -> SemanticState | None:
        versions = [v for r, v in self.versions if r == run_id]
        if not versions:
            return None
        return self.versions[(run_id, max(versions))]

    def list_versions(self, run_id: str) -> Sequence[int]:
        return sorted(v for r, v in self.versions if r == run_id)

    def put_checkpoint(self, checkpoint: StateCheckpoint) -> StateCheckpoint:
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> StateCheckpoint:
        try:
            return self.checkpoints[checkpoint_id]
        except KeyError:
            raise CheckpointNotFound(f"checkpoint {checkpoint_id!r} not found") from None

    def latest_checkpoint(self, run_id: str) -> StateCheckpoint | None:
        candidates = [c for c in self.checkpoints.values() if c.run_id == run_id]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.version)

    def list_checkpoints(self, run_id: str) -> Sequence[StateCheckpoint]:
        return [c for c in self.checkpoints.values() if c.run_id == run_id]

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        try:
            del self.checkpoints[checkpoint_id]
        except KeyError:
            raise CheckpointNotFound(f"checkpoint {checkpoint_id!r} not found") from None

    def append_sealed(self, event: Event) -> Event:
        chain = self.events.setdefault(event.run_id, [])
        if event.sequence != len(chain) + 1:
            raise ConcurrentWriteError("sealed event does not continue the chain")
        chain.append(event)
        return event


def _digest(event: Event) -> str:
    """A stable stand-in hash, mirroring the real chain's content shape."""
    content = event.model_dump(mode="json")
    body = {"sequence": content["sequence"], "payload": content["payload"]}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _memory_action_event(
    store: _MinimalStore, run_id: str, rendered_key: str, *, record_key: str | None = None
) -> Event:
    """An ACTION_RECORDED carrying a memory key, the shape the gate records."""
    payload: dict[str, Any] = {
        "rendered_key": rendered_key,
        "action": {"status": "completed", "key": record_key or rendered_key},
    }
    return store.append_event(run_id, EventType.ACTION_RECORDED, payload)


# --- capability declines --------------------------------------------------- #


def test_capability_less_defaults_decline_or_default() -> None:
    """compact_run and the index ops decline; the archive and lookup default."""

    store = _MinimalStore()
    with pytest.raises(NotImplementedError):
        store.compact_run("run_1")
    with pytest.raises(NotImplementedError):
        store.action_index_drift()
    with pytest.raises(NotImplementedError):
        store.rebuild_action_index()

    # Engines without an archive answer "no archived events" rather than raise,
    # which is what keeps exactly-once folds intact across compaction.
    assert list(store.read_archived_events("run_1")) == []

    # None means both "not found" and "no index"; callers gate on the flag.
    assert store.foreign_action("memory:s:t:k", exclude_run="run_1") is None


def test_atomic_claim_is_a_no_op_for_stores_without_transactions() -> None:
    """The context manager every claim-less engine inherits yields cleanly."""

    store = _MinimalStore()
    with store.atomic_claim("run_1", "key") as yielded:
        assert yielded is None, "the claim-less default is a transparent no-op"


def test_directory_and_offload_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The auxiliary directory and offload-threshold defaults every engine serves."""

    store = _MinimalStore()
    assert store.storage_dir == Path(".continuum")
    assert store.blob_dir == Path(".continuum") / "blobs"

    from continuum.storage.blob import PAYLOAD_OFFLOAD_ENV_VAR

    monkeypatch.setenv(PAYLOAD_OFFLOAD_ENV_VAR, "1234")
    assert store.payload_offload_bytes == 1234
    monkeypatch.delenv(PAYLOAD_OFFLOAD_ENV_VAR, raising=False)
    assert store.payload_offload_bytes == 0


def test_blank_run_ids_are_refused_at_the_entry_point() -> None:
    """A blank id wins get_active_run, so it must never enter the store."""

    store = _MinimalStore()
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="run_id must not be blank"):
            Storage.require_usable_run_id(Run(run_id=blank, goal="anything"))

    # A well-formed run passes the guard and behaves normally.
    Storage.require_usable_run_id(Run(run_id="run_1", goal="anything"))
    assert store.create_run(Run(run_id="run_1", goal="anything")).run_id == "run_1"


# --- the memory-scan fallback ---------------------------------------------- #


def test_tenant_memory_scan_falls_back_to_the_event_log() -> None:
    """The base scan answers for stores with no index, skipping what is not memory."""

    store = _MinimalStore()
    store.create_run_started(Run(run_id="run_a", goal="g"))
    _memory_action_event(store, "run_a", "memory:store:tenant_a:ns:record:1")
    # A non-memory action key is skipped, not mistaken for a tenant record.
    _memory_action_event(store, "run_a", "plain-action-key")
    # An action with no rendered_key at all is skipped.
    store.append_event("run_a", EventType.ACTION_RECORDED, {"action": {"status": "ok"}})
    # A non-action event is skipped.
    store.append_event("run_a", EventType.WORK_COMPLETED, {})

    hits = store.enumerate_tenant_memory("tenant_a")
    assert [h["run_id"] for h in hits] == ["run_a"]
    assert hits[0]["rendered_key"] == "memory:store:tenant_a:ns:record:1"
    assert hits[0]["record_key"] == "record:1"
    assert hits[0]["status"] == "completed"


def test_tenant_memory_scan_skips_non_tenants_and_deduplicates() -> None:
    """Only the requested tenant is reported, and one key is one row."""

    store = _MinimalStore()
    store.create_run_started(Run(run_id="run_a", goal="g"))
    _memory_action_event(store, "run_a", "memory:store:tenant_b:ns:record")
    _memory_action_event(store, "run_a", "memory:store:tenant_a:ns:record")
    # A duplicate write of the same key collapses: the erasure order asks for
    # the record, not for every attempt that touched it.
    _memory_action_event(store, "run_a", "memory:store:tenant_a:ns:record")

    hits = store.enumerate_tenant_memory("tenant_a")
    assert len(hits) == 1
    assert {h["run_id"] for h in store.enumerate_tenant_memory("tenant_b")} == {"run_a"}


def test_tenant_memory_scan_is_store_wide_unless_narrowed() -> None:
    """Memory identity is global to the store, so the default scan sees every run."""

    store = _MinimalStore()
    store.create_run_started(Run(run_id="run_a", goal="g"))
    store.create_run_started(Run(run_id="run_b", goal="g"))
    _memory_action_event(store, "run_a", "memory:store:tenant_a:ns:from_a")
    _memory_action_event(store, "run_b", "memory:store:tenant_a:ns:from_b")

    assert sorted(h["run_id"] for h in store.enumerate_tenant_memory("tenant_a")) == [
        "run_a",
        "run_b",
    ]
    assert [h["run_id"] for h in store.enumerate_tenant_memory("tenant_a", run_id="run_b")] == [
        "run_b"
    ]
