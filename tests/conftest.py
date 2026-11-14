"""Shared test setup. pytest runs the fixtures here for every test, in every file."""
import pytest


@pytest.fixture(autouse=True)
def no_trace_copy_from_env(monkeypatch):
    """traffic.py reads KESTREL_TRACES_COPY_DIR (the OneDrive test, set in .env) as the
    default for --copy-to. With it set, every test replay also wrote a copy into that
    real folder, and the lost-trace test in test_traces.py dropped the wrong export
    (the 2nd write was t001's copy, not t002's trace). Found in Phase 7; tests that
    need a copy pass --copy-to themselves. The suite must pass whatever .env says."""
    monkeypatch.delenv("KESTREL_TRACES_COPY_DIR", raising=False)
