"""Machine-side implementation of one fixed left or right bench nudge."""
from __future__ import annotations

import time

from .joystick import Direction, JoystickState
from .motion_hardware import MotionHardwareError


class SilentStepperBench:
    # One full motor step per press, using the bench's 1/8 microstep setting.
    STEPS_PER_NUDGE = 8

    def __init__(self, device):
        self._device = device

    def setup(self) -> None:
        device = self._device
        device.set_motor_current(1200)
        device.set_step_configuration(device.STEP_RESOLUTION_32, True)
        device.set_time_base(1)
        device.set_speed_ramping(0, 0)
        device.set_max_velocity(200)

    def read_state(self) -> JoystickState:
        device = self._device
        data = device.get_all_data()
        status = device.get_driver_status()
        fault = None
        if status.short_to_ground:
            fault = "Bench motor driver short to ground"
        elif status.over_temperature == device.OVER_TEMPERATURE_LIMIT:
            fault = "Bench motor driver overtemperature"
        elif status.motor_stalled:
            fault = "Bench motor stalled"
        elif data.input_voltage < 8000:
            fault = "Bench motor supply is below 8 V"
        return JoystickState(busy=bool(data.current_velocity or data.remaining_steps), fault=fault)

    def prepare_nudge(self) -> None:
        state = self.read_state()
        if state.fault:
            raise MotionHardwareError(state.fault)
        if state.busy:
            raise MotionHardwareError("Wait for the bench nudge to finish")
        if not self._device.get_enabled():
            self._device.set_enabled(True)

    def nudge(self, direction: Direction) -> None:
        if not isinstance(direction, Direction) or direction not in (Direction.LEFT, Direction.RIGHT):
            raise ValueError("A bench nudge must be LEFT or RIGHT")
        self._device.set_steps(-int(direction) * self.STEPS_PER_NUDGE)

    def stop_and_disable(self) -> None:
        device = self._device
        if not device.get_enabled():
            # Firmware rejects a stop command while the driver is off.
            data = device.get_all_data()
            if data.current_velocity or data.remaining_steps:
                raise MotionHardwareError("Bench driver is disabled but reports movement")
            return
        device.stop()
        deadline = time.monotonic() + 2
        while device.get_current_velocity():
            if time.monotonic() >= deadline:
                raise MotionHardwareError("Bench motor did not confirm stopping")
            time.sleep(.01)
        device.set_enabled(False)
