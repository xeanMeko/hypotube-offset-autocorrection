"""Timed Dual Relay contacts wired to a fake Digital In, independent of the motor."""
import time


class JoystickHarness:
    output_channel = None
    pulse_ends = 0.

    def press_output(self, channel, duration_ms):
        assert channel in (0, 1)
        self.output_channel = channel
        self.pulse_ends = time.monotonic() + duration_ms / 1000

    def release_outputs(self):
        self.output_channel = None

    def read_inputs(self):
        levels = list(self.inputs)
        if self.output_channel is not None and time.monotonic() < self.pulse_ends:
            levels[{0: 1, 1: 2}[self.output_channel]] = True
        return tuple(levels)
