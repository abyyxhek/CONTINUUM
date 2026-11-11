"""The SQLite reference engine's integrity arms (issue #2059).

``sqlite.py`` is the reference engine: every other engine is expected to
behave like it, so an untested arm here is a hole in the contract everywhere.
This suite pins the ``caused_by`` validation (both copies), the archive
audit's corruption shapes, the fail-closed compaction raises, and the small
capability guarantees the storage interface documents but nothing asserted:
an idempotent ``close``, the ``CONTINUUM_STORAGE_DIR`` override, and a NULL
``rendered_key`` skipped rather than crashed on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from continuum.events import Event, EventType, Origin
from continuum.models import Run
from continuum.security.hashing import make_id
from continuum.storage import SQLiteStorage
from continuum.storage.base import CorruptedRecord


@pytest.fixture
def db(tmp_path: Path) -> str:
    return str(tmp_path / "integrity.db")


def _seed(db: str, run_id: str = "run_1", events: int = 2) -> SQLiteStorage:
    """A store holding a run whose first ``events`` events are chainable."""
    store = SQLiteStorage(db)
    store.create_run(Run(run_id=run_id, goal="g"))
    store.append_event(run_id, EventType.RUN_STARTED, {"goal": "g"})
    for i in range(events - 1):
        store.append_event(
            run_id, EventType.EVIDENCE_ADDED, {"evidence_id": f"e{i}", "summary": "s"}
        )
    return store


def _sealed_next(store: SQLiteStorage, run_id: str, payload: dict) -> Event:
    """A pre-sealed DECISION_CREATED event citing ``caused_by``, chain-valid."""
    head = store._connection.execute(
        "SELECT sequence, hash FROM events WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    event = Event(
        event_id=make_id("event"),
        run_id=run_id,
        sequence=(head["sequence"] if head else 0) + 1,
        type=EventType.DECISION_CREATED,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        payload=payload,
        causer_event_id=None,
        source=Origin.DETERMINISTIC,
        prev_hash=head["hash"] if head else None,
    )
    return event.sealed()


# --- caused_by validation, both copies (#2059) --------------------------------- #


def test_append_event_rejects_a_caused_by_that_is_not_a_list(db: str) -> None:
    """The copy on the atomic append path (the one compaction reuses)."""
    store = _seed(db)
    with pytest.raises(ValueError, match="caused_by must be a list"):
        store.append_event("run_1", EventType.DECISION_CREATED, {"caused_by": "event_1"})
    store.close()


def test_append_event_rejects_a_caused_list_longer_than_32(db: str) -> None:
    store = _seed(db)
    with pytest.raises(ValueError, match="caused_by must contain at most 32 ids"):
        store.append_event(
            "run_1", EventType.DECISION_CREATED, {"caused_by": [f"e{i}" for i in range(33)]}
        )
    store.close()


@pytest.mark.parametrize("bad_entry", ["", "x" * 129])
def test_append_event_rejects_out_of_bounds_caused_by_entries(db: str, bad_entry: str) -> None:
    store = _seed(db)
    with pytest.raises(ValueError, match="caused_by entries must be 1-128 chars"):
        store.append_event("run_1", EventType.DECISION_CREATED, {"caused_by": [bad_entry]})
    store.close()


def test_append_event_rejects_a_caused_by_id_the_store_does_not_hold(db: str) -> None:
    """The causal claim is validated against what the store actually has."""
    store = _seed(db)
    with pytest.raises(ValueError, match="unknown caused_by id 'no-such-event'"):
        store.append_event("run_1", EventType.DECISION_CREATED, {"caused_by": ["no-such-event"]})
    store.close()


def test_append_sealed_rejects_a_caused_by_that_is_not_a_list(db: str) -> None:
    """The same three guards are duplicated on the pre-sealed path; all must
    agree or an import through one path would bypass the other."""
    store = _seed(db)
    with pytest.raises(ValueError, match="caused_by must be a list"):
        store.append_sealed(_sealed_next(store, "run_1", {"caused_by": "event_1"}))
    store.close()


def test_append_sealed_rejects_an_unknown_caused_by_id(db: str) -> None:
    store = _seed(db)
    with pytest.raises(ValueError, match="unknown caused_by id 'no-such-event'"):
        store.append_sealed(_sealed_next(store, "run_1", {"caused_by": ["no-such-event"]}))
    store.close()


def test_append_sealed_accepts_a_caused_by_id_that_lives_only_in_the_archive(
    db: str,
) -> None:
    """The archive half of the existence UNION is what compaction makes
    load-bearing: an id moved out of the live log by compaction must still
    satisfy the check on re-import."""
    with _seed(db) as store:
        archived_id = store.read_events("run_1")[0].event_id
        store.compact_run("run_1", through_sequence=1)
        assert not any(e.event_id == archived_id for e in store.read_events("run_1")), (
            "precondition: the cited id must live only in the archive"
        )

        event = _sealed_next(store, "run_1", {"caused_by": [archived_id]})
        store.append_sealed(event)
        assert store.read_events("run_1")[-1].event_id == event.event_id


# --- _audit_archive's corruption shapes (#2059) --------------------------------- #


def _compacted_store(db: str) -> SQLiteStorage:
    """A store whose run prefix sits in events_archive and live tail follows."""
    store = _seed(db, events=3)
    store.compact_run("run_1", through_sequence=2)
    return store


def test_a_clean_archive_verifies_and_supplies_the_edge(db: str) -> None:
    """A verified archive keeps the whole run green and the audit's edge is
    the last archived row, which the live chain continues from."""
    store = _compacted_store(db)
    try:
        archived = store._connection.execute(
            "SELECT sequence, hash FROM events_archive WHERE run_id = 'run_1' "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        live = store._connection.execute(
            "SELECT COUNT(*) AS n FROM events WHERE run_id = 'run_1'"
        ).fetchone()
        report = store.verify_events("run_1")
        assert report.ok, report.violations
        assert report.checked == 2 + live["n"]
        # The verified archive stays the trusted prefix, and the live chain
        # that continues from its edge extends the trust to the run's head.
        assert report.trusted_through == {"run_1": store.last_sequence("run_1")}
        assert archived["sequence"] <= report.trusted_through["run_1"]
    finally:
        store.close()


def test_an_unreadable_archived_row_is_reported_and_resets_the_edge(db: str) -> None:
    """The shape a forged or half-written archive hits: the row cannot even be
    parsed, so it is UNREADABLE_RECORD and the edge the live chain would
    continue from is dropped."""
    store = _compacted_store(db)
    try:
        store._connection.execute(
            "UPDATE events_archive SET payload = '{not json' WHERE run_id = 'run_1'"
        )
        report = store.verify_events("run_1")
        assert not report.ok
        kinds = [v.kind for v in report.violations]
        assert "UNREADABLE_RECORD" in kinds
        unreadable = next(v for v in report.violations if v.kind == "UNREADABLE_RECORD")
        assert unreadable.detail.startswith("archived row unreadable:")
        # The corrupted row still counts as audited.
        assert report.checked >= 2
    finally:
        store.close()


def test_an_archived_sequence_gap_is_reported(db: str) -> None:
    store = _compacted_store(db)
    try:
        store._connection.execute(
            "UPDATE events_archive SET sequence = sequence + 10 WHERE run_id = 'run_1' "
            "AND sequence = 2"
        )
        report = store.verify_events("run_1")
        assert not report.ok
        gap = next(v for v in report.violations if v.kind == "SEQUENCE_GAP")
        assert gap.detail == "archived: expected sequence 2"
    finally:
        store.close()


def test_an_archived_broken_chain_link_is_reported(db: str) -> None:
    store = _compacted_store(db)
    try:
        store._connection.execute(
            "UPDATE events_archive SET prev_hash = 'deadbeef' WHERE run_id = 'run_1' "
            "AND sequence = 2"
        )
        report = store.verify_events("run_1")
        assert not report.ok
        broken = next(v for v in report.violations if v.kind == "BROKEN_CHAIN")
        assert broken.detail.startswith("archived: prev_hash 'deadbeef' does not match")
    finally:
        store.close()


# --- compact_run's fail-closed raises (#2059) ----------------------------------- #


def test_compacting_a_run_with_no_projectable_state_refuses(db: str) -> None:
    """A run whose log carries nothing to anchor cannot be compacted: the
    failure is raised, not silently faked."""
    store = SQLiteStorage(db)
    store.create_run(Run(run_id="run_1", goal="g"))
    try:
        with pytest.raises(ValueError, match="run 'run_1' could not be anchored"):
            store.compact_run("run_1")
    finally:
        store.close()


def test_compacting_a_run_whose_version_cannot_be_read_refuses(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second fail-closed arm: an anchor that left no readable version is
    refused rather than compacting on top of a state nobody can replay."""
    store = _seed(db, events=1)
    try:
        monkeypatch.setattr(store, "latest_version", lambda run_id: None)
        with pytest.raises(ValueError, match="no projectable state"):
            store.compact_run("run_1")
    finally:
        store.close()


def test_compacting_through_zero_compacts_nothing(db: str) -> None:
    store = _seed(db)
    try:
        with pytest.raises(ValueError, match="nothing to compact: anchor would be empty"):
            store.compact_run("run_1", through_sequence=0)
    finally:
        store.close()


# --- capability guarantees the interface documents (#2059) ----------------------- #


def test_close_is_idempotent_and_safe_to_call_twice(db: str) -> None:
    """The storage interface documents close() as idempotent; only this
    engine's guard implements it, so the guarantee is asserted, not assumed."""
    store = SQLiteStorage(db)
    store.close()
    store.close()


def test_foreign_action_converts_a_corrupt_index_row(db: str) -> None:
    """An unparseable action_index row raises CorruptedRecord (with the key
    truncated) rather than escaping a raw JSON error."""
    store = _seed(db)
    try:
        store._connection.execute(
            "INSERT INTO action_index(key, run_id, action_id, status, updated_seq, action_json) "
            "VALUES ('bad-key-123456', 'run_1', 'a', 'started', 1, 'not json')"
        )
        with pytest.raises(CorruptedRecord, match="bad-key-1234"):
            store.foreign_action("bad-key-123456", exclude_run="other")
    finally:
        store.close()


def test_enumerate_tenant_memory_skips_a_null_rendered_key(db: str) -> None:
    """A NULL rendered_key is skipped rather than handed to the key splitter
    to crash on, and the tenant scan still finds a real memory row."""
    store = _seed(db)
    try:
        store._connection.execute(
            "INSERT INTO action_index(key, run_id, action_id, status, updated_seq, "
            "action_json, rendered_key) "
            "VALUES ('k1', 'run_1', 'a', 'started', 1, '{}', NULL)"
        )
        store._connection.execute(
            "INSERT INTO action_index(key, run_id, action_id, status, updated_seq, "
            "action_json, rendered_key) "
            "VALUES ('k2', 'run_1', 'a', 'started', 1, '{}', 'mem:store:tenant:rec:1')"
        )
        rows = store.enumerate_tenant_memory("tenant")
        assert [r["rendered_key"] for r in rows] == ["mem:store:tenant:rec:1"]
    finally:
        store.close()


def test_storage_dir_follows_the_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented override for the auxiliary-file directory."""
    aux = tmp_path / "aux"
    monkeypatch.setenv("CONTINUUM_STORAGE_DIR", str(aux))
    with SQLiteStorage(":memory:") as store:
        assert store.storage_dir == aux
