"""RunEngine state machine module."""

import threading
from enum import Enum
from typing import TYPE_CHECKING, overload
from weakref import WeakKeyDictionary

from .log import state_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from .run_engine import RunEngine


class TransitionError(RuntimeError):
    """Raised when an invalid transition is attempted."""


# mapping of valid state transitions.
_TRANSITIONS: dict[str, list[str]] = {
    "idle": ["running", "panicked"],
    "running": ["idle", "pausing", "halting", "stopping", "aborting", "suspending", "panicked"],
    "pausing": ["paused", "idle", "halting", "aborting", "panicked"],
    "suspending": ["running", "halting", "aborting", "panicked"],
    "paused": ["idle", "running", "halting", "stopping", "aborting", "panicked"],
    "halting": ["idle", "panicked"],
    "stopping": ["idle", "panicked"],
    "aborting": ["idle", "panicked"],
    "panicked": [],
}


class REState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    HALTING = "halting"
    STOPPING = "stopping"
    ABORTING = "aborting"
    SUSPENDING = "suspending"
    PANICKED = "panicked"

    @classmethod
    def states(cls) -> list[str]:
        return [s.value for s in cls]


class RunEngineStateMachine:
    def __init__(self, on_transition: "Callable[[REState, REState], None] | None" = None) -> None:
        self._state = REState.IDLE
        self._lock = threading.RLock()
        self._on_transition = on_transition

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._state.value == other
        if isinstance(other, REState):
            return self._state == other
        if isinstance(other, RunEngineStateMachine):
            return self._state == other._state
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._state)

    def __str__(self) -> str:
        return self._state.value

    @property
    def is_idle(self) -> bool:
        return self._state == REState.IDLE

    @property
    def is_running(self) -> bool:
        return self._state == REState.RUNNING

    @property
    def is_pausing(self) -> bool:
        return self._state == REState.PAUSING

    @property
    def is_paused(self) -> bool:
        return self._state == REState.PAUSED

    @property
    def is_halting(self) -> bool:
        return self._state == REState.HALTING

    @property
    def is_stopping(self) -> bool:
        return self._state == REState.STOPPING

    @property
    def is_aborting(self) -> bool:
        return self._state == REState.ABORTING

    @property
    def is_suspending(self) -> bool:
        return self._state == REState.SUSPENDING

    @property
    def is_panicked(self) -> bool:
        return self._state == REState.PANICKED

    @property
    def can_pause(self) -> bool:
        return "pausing" in _TRANSITIONS[self._state.value]

    def set(self, new_state: str | REState) -> tuple[REState, REState]:
        """Put the FSM into a new state.

        Parameters
        ----------
        new_state: str | REState
            The new state to transition to.

        Returns
        -------
        tuple[REState, REState]
            A tuple of (old_state, new_state) after the transition.

        Raises
        ------
        TransitionError
            If the transition from the current state
            to the new state is not allowed.

        Returns (old, new) for the caller to use (e.g. logging)."""
        new = REState(new_state)
        with self._lock:
            old = self._state
            allowed = _TRANSITIONS[old.value]
            if new.value not in allowed:
                raise TransitionError(f"Cannot transition from '{old.value}' to '{new.value}'. Allowed: {allowed}")
            self._state = new
        return old, new


class MachineDescriptor:
    """Descriptor-accessor for the state machine.

    When providing the state machine type, the descriptor will
    create a new instance of that type.
    When no type is provided, it is expected that
    the owning class will set an instance
    of the state machine on itself before any access.

    i.e.

    class MyClass:
        # the machine instance will be
        # created automatically
        state = MachineDescriptor(MyStateMachine)

    class MyOtherClass:
        # the class must provide the machine
        # instance itself before any access
        state = MachineDescriptor()

        def __init__(self, fsm: MyStateMachine | None = None):
            self.state = fsm or MyStateMachine()
    """

    def __init__(self, fsm_type: type[RunEngineStateMachine] | None = None) -> None:
        self._fsm_type = fsm_type
        self._memory: WeakKeyDictionary[RunEngine, RunEngineStateMachine] = WeakKeyDictionary()

    def _get_or_create(self, obj: "RunEngine") -> RunEngineStateMachine:
        try:
            return self._memory[obj]
        except KeyError as exc:
            if self._fsm_type is None:
                raise RuntimeError(f"{obj!r} has no FSM assigned and no FSM type was provided.") from exc
            fsm = self._fsm_type()
            self._memory[obj] = fsm
            return fsm

    @overload
    def __get__(self, obj: None, owner: type) -> "MachineDescriptor": ...
    @overload
    def __get__(self, obj: "RunEngine", owner: type) -> RunEngineStateMachine: ...
    def __get__(self, obj: "RunEngine | None", owner: type) -> "RunEngineStateMachine | MachineDescriptor":
        if obj is None:
            return self
        with obj._state_lock:
            return self._get_or_create(obj)

    def __set__(self, obj: "RunEngine", value: str | REState | RunEngineStateMachine) -> None:
        if isinstance(value, RunEngineStateMachine):
            self._memory[obj] = value
            return
        with obj._state_lock:
            old, new = self._get_or_create(obj).set(value)
        tags = {"old_state": old, "new_state": new, "RE": self}
        state_logger.info("Change state on %r from %r -> %r", obj, old, new, extra=tags)
        if obj.state_hook is not None:
            obj.state_hook(new, old)
