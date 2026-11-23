"""Choose fixed joystick presses from settled, signed camera X measurements."""
from __future__ import annotations

from .joystick import Direction


class DirectXSearch:
    """Probe right, then use the outcome of each nudge to choose the next one."""

    def __init__(self, tolerance_px: float = 0.):
        self._previous_dx: float | None = None
        self._last_crossing: tuple[float, float] | None = None
        self._direction = Direction.RIGHT
        self._change_threshold = max(.05, tolerance_px * .1)
        self._best_error = float("inf")
        self._no_progress = 0
        self._probing = True
        self.crossed_target = False

    def next_direction(self, dx: float) -> Direction:
        """Select one press for an out-of-tolerance measurement."""
        error = abs(dx)
        previous = self._previous_dx
        self.crossed_target = previous is not None and (dx > 0) != (previous > 0)
        if self._probing and previous is not None and dx != previous:
            self._probing = False
            # Count progress from the first measured response, after any
            # initial slack, so reversing the probe has time to take up slack.
            self._best_error = float("inf")
        if self.crossed_target:
            if self._last_crossing is not None:
                start, end = self._last_crossing
                same_endpoints = (abs(dx - start) <= self._change_threshold and
                                  abs(previous - end) <= self._change_threshold)
                if same_endpoints and error <= abs(previous):
                    raise ValueError("Requested X tolerance is smaller than one joystick nudge can resolve")
            self._last_crossing = previous, dx
            self._direction = Direction(-self._direction)
        elif previous is not None and error > abs(previous):
            self._direction = Direction(-self._direction)

        if error < self._best_error:
            self._best_error = error
            self._no_progress = 0
        else:
            self._no_progress += 1
            if self._no_progress >= 4:
                raise ValueError("Direct X alignment stopped after 4 joystick nudges without progress")
        self._previous_dx = dx
        return self._direction
