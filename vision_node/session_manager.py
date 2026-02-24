"""
session_manager.py

State machine for a single weighing session.

States:
    IDLE            Scale is empty. Waiting for a reel.
    WEIGHT_DETECTED Weight rose above detect threshold. Brief hold to confirm
                    it is a real placement and not a passing disturbance.
    STABILIZING     Weight is sustained. Running stability filter.
    CONFIRMED       Stable weight locked and published. Waiting for removal.
    RESET           Reel removed. One-frame cleanup before next session.
    ERROR           Weight lost mid-session (forklift not fully clear).
                    Operator must press RESCAN on HMI.

Transitions:
    IDLE            → WEIGHT_DETECTED   weight >= DETECT_THRESHOLD_KG
    WEIGHT_DETECTED → STABILIZING       weight sustained for DETECT_HOLD_SEC
    WEIGHT_DETECTED → IDLE              weight dropped before hold completes
    STABILIZING     → CONFIRMED         StabilityFilter.get_stable_weight() is set
    STABILIZING     → ERROR             weight drops below IDLE_THRESHOLD_KG
    CONFIRMED       → RESET             weight drops below IDLE_THRESHOLD_KG
    RESET           → IDLE              immediate (one-frame housekeeping)
    ERROR           → IDLE              on rescan() from HMI operator
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable, Optional

from stability_filter import StabilityFilter
from config import (
    IDLE_THRESHOLD_KG,
    DETECT_THRESHOLD_KG,
    DETECT_HOLD_SEC,
)


class State(Enum):
    IDLE            = auto()
    WEIGHT_DETECTED = auto()
    STABILIZING     = auto()
    CONFIRMED       = auto()
    RESET           = auto()
    ERROR           = auto()


class SessionManager:
    """
    Weighing session state machine.

    Usage:
        sm = SessionManager(
            on_confirmed=lambda w: mqtt.publish_stable_weight(w),
            on_error=lambda r: log_error(r),
            on_state_change=lambda s: mqtt.publish_state(s.name),
        )
        # In the main loop:
        sm.update(weight_kg_or_None)
    """

    def __init__(
        self,
        on_confirmed: Callable[[float], None],
        on_error: Callable[[str], None],
        on_state_change: Optional[Callable[[State], None]] = None,
    ) -> None:
        self._state              = State.IDLE
        self._on_confirmed       = on_confirmed
        self._on_error           = on_error
        self._on_state_change    = on_state_change
        self._stability          = StabilityFilter()
        self._confirmed_weight: Optional[float] = None
        self._detect_since: Optional[float]     = None

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    @property
    def confirmed_weight(self) -> Optional[float]:
        """Confirmed stable weight, set only while in CONFIRMED state."""
        return self._confirmed_weight

    def update(self, weight: Optional[float]) -> None:
        """Feed the latest weight reading into the state machine."""
        if   self._state == State.IDLE:            self._handle_idle(weight)
        elif self._state == State.WEIGHT_DETECTED: self._handle_weight_detected(weight)
        elif self._state == State.STABILIZING:     self._handle_stabilizing(weight)
        elif self._state == State.CONFIRMED:       self._handle_confirmed(weight)
        elif self._state == State.RESET:           self._handle_reset()
        # State.ERROR: wait passively for rescan()

    def rescan(self) -> None:
        """Operator pressed RESCAN on HMI. Unconditionally resets to IDLE."""
        self._stability.reset()
        self._confirmed_weight = None
        self._detect_since     = None
        self._transition(State.IDLE)

    # ─── State handlers ───────────────────────────────────────────────────────

    def _handle_idle(self, weight: Optional[float]) -> None:
        if weight is not None and weight >= DETECT_THRESHOLD_KG:
            self._detect_since = time.monotonic()
            self._transition(State.WEIGHT_DETECTED)

    def _handle_weight_detected(self, weight: Optional[float]) -> None:
        if weight is None or weight < IDLE_THRESHOLD_KG:
            # Weight disappeared before the hold expired — false trigger
            self._detect_since = None
            self._transition(State.IDLE)
            return

        elapsed = time.monotonic() - (self._detect_since or time.monotonic())
        if elapsed >= DETECT_HOLD_SEC:
            self._stability.reset()
            self._stability.add(weight)
            self._detect_since = None
            self._transition(State.STABILIZING)

    def _handle_stabilizing(self, weight: Optional[float]) -> None:
        if weight is None or weight < IDLE_THRESHOLD_KG:
            self._on_error(
                "Weight lost before stable. Ensure reel is fully on scale."
            )
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
            self._confirmed_weight = None
            self._stability.reset()
            self._transition(State.RESET)

    def _handle_reset(self) -> None:
        # One-frame pause — gives callbacks time to fire before cycling back.
        self._transition(State.IDLE)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _transition(self, new_state: State) -> None:
        if new_state != self._state:
            self._state = new_state
            if self._on_state_change:
                self._on_state_change(new_state)
