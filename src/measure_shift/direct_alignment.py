"""Choose fixed joystick presses from settled, signed camera measurements."""
from __future__ import annotations

from .joystick import Direction


class AxisSearch:
    """Probe in one direction, then use the outcome of each nudge to choose the next one."""

    def __init__(self, axis: str, tolerance_px: float):
        self.axis = axis
        self._previous_offset: float | None = None
        self._last_crossing: tuple[float, float] | None = None
        self._direction = Direction.RIGHT if axis == "X" else Direction.UP
        self._change_threshold = max(.05, tolerance_px * .1)
        self._best_error = float("inf")
        self._no_progress = 0
        self._probing = True
        self.crossed_target = False

    def next_direction(self, offset: float) -> Direction:
        """Select one press for an out-of-tolerance measurement."""
        error = abs(offset)
        previous = self._previous_offset
        self.crossed_target = previous is not None and (offset > 0) != (previous > 0)
        if self._probing and previous is not None and offset != previous:
            self._probing = False
            # Count progress from the first measured response, after any
            # initial slack, so reversing the probe has time to take up slack.
            self._best_error = float("inf")
        if self.crossed_target:
            if self._last_crossing is not None:
                start, end = self._last_crossing
                same_endpoints = (abs(offset - start) <= self._change_threshold and
                                  abs(previous - end) <= self._change_threshold)
                if same_endpoints and error <= abs(previous):
                    raise ValueError(f"Requested {self.axis} tolerance is smaller than one joystick nudge can resolve")
            self._last_crossing = previous, offset
            self._direction = Direction(-self._direction)
        elif previous is not None and error > abs(previous):
            self._direction = Direction(-self._direction)

        if error < self._best_error:
            self._best_error = error
            self._no_progress = 0
        else:
            self._no_progress += 1
            if self._no_progress >= 4:
                raise ValueError(f"Direct {self.axis} alignment stopped after 4 joystick nudges without progress")
        self._previous_offset = offset
        return self._direction
