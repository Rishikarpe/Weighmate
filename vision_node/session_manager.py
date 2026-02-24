"""
session_manager.py

State machine for a single weighing session.

States:
    IDLE          Scale is empty or near zero. Waiting for a reel.
    STABILIZING   Weight detected above threshold. Waiting for it to settle.
    CONFIRMED     Weight has been stable for STABLE_DURATION_SECONDS.
                  Confirmed weight is locked. Waiting for reel to be removed.
    ERROR         Unexpected weight drop mid-session (forklift partially left).
                  Operator must rescan.

Transitions:
    IDLE        → STABILIZING   weight rises above IDLE_THRESHOLD_KG
    STABILIZING → CONFIRMED     StabilityFilter confirms stable weight
    STABILIZING → ERROR         weight drops below IDLE_THRESHOLD_KG before stable
    CONFIRMED   → IDLE          weight drops below IDLE_THRESHOLD_KG (reel removed)
    ERROR       → IDLE          on rescan command from HMI
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable, Optional

from stability_filter import StabilityFilter


# ─── Configuration ────────────────────────────────────────────────────────────

# Weight above which we consider a reel to be on the scale.
IDLE_THRESHOLD_KG = 30.0


class State(Enum):
    IDLE        = auto()
    STABILIZING = auto()
    CONFIRMED   = auto()
    ERROR       = auto()


class SessionManager:
    """
    Manages the weighing session state machine.

    Usage:
        def on_confirmed(weight):
            mqtt.publish_stable_weight(weight)

        def on_error(reason):
            mqtt.publish_error(reason)

        sm = SessionManager(on_confirmed=on_confirmed, on_error=on_error)

        # In your main loop:
        sm.update(current_weight)   # call with each new weight reading (or None)
    """

    def __init__(
        self,
        on_confirmed: Callable[[float], None],
        on_error: Callable[[str], None],
        on_state_change: Optional[Callable[[State], None]] = None,
    ) -> None:
        """
        Args:
            on_confirmed:    Called once with the stable weight when CONFIRMED.
            on_error:        Called with an error reason string.
            on_state_change: Optional callback whenever state changes.
        """
        self._state = State.IDLE
        self._on_confirmed = on_confirmed
        self._on_error = on_error
        self._on_state_change = on_state_change

        self._stability = StabilityFilter()
        self._confirmed_weight: Optional[float] = None

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    @property
    def confirmed_weight(self) -> Optional[float]:
        """The confirmed stable weight. Only set in CONFIRMED state."""
        return self._confirmed_weight

    def update(self, weight: Optional[float]) -> None:
        """
        Feed the latest weight reading into the state machine.

        Args:
            weight: Current weight in kg, or None if detection failed this frame.
        """
        if self._state == State.IDLE:
            self._handle_idle(weight)

        elif self._state == State.STABILIZING:
            self._handle_stabilizing(weight)

        elif self._state == State.CONFIRMED:
            self._handle_confirmed(weight)

        elif self._state == State.ERROR:
            pass  # waiting for rescan command from HMI

    def rescan(self) -> None:
        """
        Called when the HMI operator presses RESCAN.
        Resets state machine to IDLE regardless of current state.
        """
        self._stability.reset()
        self._confirmed_weight = None
        self._transition(State.IDLE)

    # ─── State handlers ───────────────────────────────────────────────────────

    def _handle_idle(self, weight: Optional[float]) -> None:
        if weight is not None and weight >= IDLE_THRESHOLD_KG:
            self._stability.reset()
            self._transition(State.STABILIZING)
            # Feed this first reading in immediately
            self._stability.add(weight)

    def _handle_stabilizing(self, weight: Optional[float]) -> None:
        if weight is None or weight < IDLE_THRESHOLD_KG:
            # Weight disappeared before stabilising — forklift not fully clear
            self._on_error("Weight lost before stable. Ensure reel is fully on scale.")
            self._stability.reset()
            self._transition(State.ERROR)
            return

        self._stability.add(weight)
        stable = self._stability.get_stable_weight()

        if stable is not None:
            self._confirmed_weight = stable
            self._on_confirmed(stable)
            self._transition(State.CONFIRMED)

    def _handle_confirmed(self, weight: Optional[float]) -> None:
        if weight is None or weight < IDLE_THRESHOLD_KG:
            # Reel has been removed — reset for next session
            self._confirmed_weight = None
            self._stability.reset()
            self._transition(State.IDLE)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _transition(self, new_state: State) -> None:
        if new_state != self._state:
            self._state = new_state
            if self._on_state_change:
                self._on_state_change(new_state)
