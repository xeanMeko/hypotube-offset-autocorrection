"""Experimental X search using settled target error, without a pixel scale."""
from __future__ import annotations


class DirectXSearch:
    """Grow finite moves, then halve on target crossings to refine alignment.

    Repeat through backlash without assuming driver travel means image travel.
    The caller handles tolerance, fresh feedback, and operation limits. Keep
    experiment-specific tuning here so removing the extra mode is straightforward.
    """

    INITIAL_STEPS = 8

    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self._raw_position: int | None = None
        self._previous_dx: float | None = None
        self._direction = 1
        self._step = min(self.INITIAL_STEPS, max_steps)
        self._refining = False
        self._best_error = float("inf")
        self._limited_no_progress = 0

    def next_steps(self, position: int, dx: float) -> int:
        if self._raw_position is not None:
            print(f"Direct X alignment: position {position}, dx {dx:.3f}, step {self._step}, direction {self._direction}, best error {self._best_error:.3f}")
            if position == self._raw_position:
                raise ValueError("Direct X alignment stopped: motor position did not change")
        self._raw_position = position

        error = abs(dx)
        if error < self._best_error:
            # print(f"Direct X alignment: new best error {error:.3f} < previous {self._best_error:.3f}")
            self._best_error = error
            self._limited_no_progress = 0
        elif self._step == self.max_steps:
            # print(f"Direct X alignment: no progress for {self._limited_no_progress + 1} maximum-size moves")
            self._limited_no_progress += 1
            if self._limited_no_progress >= 5:
                raise ValueError("Direct X alignment stopped after 5 maximum-size moves without progress")

        if self._previous_dx is not None:
            # print(f"Direct X alignment: previous dx {self._previous_dx:.3f}, current dx {dx:.3f}")
            crossed = (dx > 0) != (self._previous_dx > 0)
            if crossed and self._step == 1:
                raise ValueError("Requested X tolerance is smaller than one driver step can resolve")
            if crossed or error > abs(self._previous_dx):
                self._direction *= -1
                self._step = max(1, self._step // 2)
                self._refining |= crossed
                # print(f"Direct X alignment: crossed target, reversing direction to {self._direction}, halving step to {self._step}, refining {self._refining}")
            elif not self._refining:
                # print(f"Direct X alignment: no crossing, doubling step from {self._step} to {min(self.max_steps, self._step * 2)}")
                self._step = min(self.max_steps, self._step * 2)
            # During refinement, unchanged image error can be backlash on a
            # reversal. Keep taking finite moves until the camera sees motion;
            # never shrink a motor-position bracket around this dead travel.
        self._previous_dx = dx
        return self._direction * self._step
