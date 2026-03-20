import threading

import pytest

from bluesky.fsm import _TRANSITIONS, MachineDescriptor, REState, RunEngineStateMachine


class Owner:
    fsm = MachineDescriptor(RunEngineStateMachine)

    def __init__(self) -> None:
        self._state_lock = threading.RLock()


class Borrower:
    fsm = MachineDescriptor()

    def __init__(self, fsm: RunEngineStateMachine, lock: threading.RLock) -> None:
        # the expected type of fsm.__set__ is RunEngine, but
        # for testing the functionality we can ignore the type mismatch...
        self.fsm = fsm  # type: ignore[arg-type]
        self._state_lock = lock


class OwnerNoLock:
    """Raises attribute error because the state machine needs a _state_lock to work."""

    fsm = MachineDescriptor(RunEngineStateMachine)


def test_states():
    assert REState.states() == [
        "idle",
        "running",
        "pausing",
        "paused",
        "halting",
        "stopping",
        "aborting",
        "suspending",
        "panicked",
    ]


def test_transitions():

    assert _TRANSITIONS["idle"] == ["running", "panicked"]
    assert _TRANSITIONS["running"] == [
        "idle",
        "pausing",
        "halting",
        "stopping",
        "aborting",
        "suspending",
        "panicked",
    ]
    assert _TRANSITIONS["pausing"] == ["paused", "idle", "halting", "aborting", "panicked"]
    assert _TRANSITIONS["suspending"] == ["running", "halting", "aborting", "panicked"]
    assert _TRANSITIONS["paused"] == ["idle", "running", "halting", "stopping", "aborting", "panicked"]
    assert _TRANSITIONS["halting"] == ["idle", "panicked"]
    assert _TRANSITIONS["stopping"] == ["idle", "panicked"]
    assert _TRANSITIONS["aborting"] == ["idle", "panicked"]
    assert _TRANSITIONS["panicked"] == []


def test_multi_obj_fsm():

    owner = Owner()

    assert isinstance(owner.fsm, RunEngineStateMachine)
    assert owner.fsm.is_idle

    borrower = Borrower(owner.fsm, owner._state_lock)
    assert borrower.fsm is owner.fsm


def test_no_lock():
    owner = OwnerNoLock()

    with pytest.raises(AttributeError):
        _ = owner.fsm.is_idle
