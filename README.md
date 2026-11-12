# Measure Shift

A Linux desktop application for aligning live images against a fixed reference,
with optional joystick/keyboard motor nudging and mechanical X alignment. It
measures X/Y translation in native image pixels using phase
correlation or template matching, with optional image analytics and experimental
slot detection. The original notebooks and sample images remain available.

## Install and launch

Use Python 3.11 or newer and a local virtual environment. The package installs
PySide6, pyqtgraph, NumPy, and **one** OpenCV distribution,
`opencv-python-headless`. Qt supplies the desktop interface; OpenCV's own GUI
package is unnecessary.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,cameras]'
.venv/bin/measure-shift --demo
```

To reproduce the dependency versions validated on Linux with Python 3.12, use the
supplied lock file as constraints when installing into a fresh virtual environment:

```bash
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[dev,cameras]'
```

The lock includes pytest, both optional camera bindings, and Tinkerforge motion
bindings. For a different
supported Python version, use the regular installation command above so pip can
select compatible versions from the package's declared ranges.

Use `-e '.[dev]'` to omit camera bindings, or `-e .` for the application alone.
Camera SDKs are imported only when discovering or opening the corresponding
camera. Installing the application does not configure network adapters, change
camera firmware, or install manufacturer drivers.

In PyCharm, select this project's `.venv/bin/python` as the interpreter. You can
also launch the package with `.venv/bin/python -m measure_shift --demo`.

The supplied project `.venv` includes development tools, both optional Python
camera bindings, and Tinkerforge motion bindings. It isolates the application from any OpenCV packages already
installed in the system or Conda environment. On a Linux desktop, the system
must also provide Qt's display libraries; a missing `xcb` library reported by Qt
must be installed using your distribution's package manager.

If an earlier version segfaulted immediately in both Demo and camera modes when
using a Conda-based virtual environment, the application now imports `QtWidgets`
before `QtGui` or pyqtgraph. On the affected Linux installation, importing
`QtGui` first mixed Conda's `libxkbcommon` with the system's
`libxkbcommon-x11`, crashing during `QApplication` initialization before any
camera was opened. Keep this import order when extending the interface. A
virtual environment created with `/usr/bin/python3` also avoids inheriting the
Conda interpreter's native library search path.

Other launch examples:

```bash
.venv/bin/measure-shift --source images --path images/horizontal/1.bmp
.venv/bin/measure-shift --source images --path images/horizontal/
.venv/bin/measure-shift --source video --path recording.avi
.venv/bin/measure-shift --source basler --device CAMERA_SERIAL
.venv/bin/measure-shift --source ids --device DEVICE_ID
.venv/bin/measure-shift --source images --path images/horizontal/2.bmp --reference images/horizontal/1.bmp
```

Run `.venv/bin/measure-shift --help` for available options. The default source is
the synthetic demo, which includes a textured circular object, dark slot, and a
known moving translation. Files and video replay in a loop. Image directories
play supported images in filename order; all frames used with one reference must
have the same dimensions.

## Build a Linux executable

Build on Linux with Python 3.11 or newer. On Ubuntu 24.04, install the build
tools and common Qt display libraries first:

```bash
sudo apt install python3-venv binutils libgl1 libegl1 libxcb-cursor0 libxkbcommon-x11-0
bash scripts/build_linux.sh
./dist/measure-shift/measure-shift --demo
```

The script creates a separate system-Python environment in `build/linux-venv`,
installs the app with camera and motion bindings plus PyInstaller, and bundles
the dynamically imported hardware modules. Set `MEASURE_SHIFT_BUILD_PYTHON` to
another Python executable if needed. The build does not use the existing
development `.venv`.

After the first build, use `bash scripts/build_linux.sh --skip-install` to
rebuild the current source without accessing the package index, provided the
dependencies have not changed.

Copy the entire `dist/measure-shift/` directory, or extract the generated
`dist/measure-shift-linux-<architecture>.tar.gz` archive on the target computer.
The bundle includes Python and the app's Python dependencies; camera drivers
and manufacturer runtimes still need to be installed on the target computer.
The resolved build dependencies are recorded in the bundle's
`build-requirements.txt`.

Build on the oldest Ubuntu release you intend to support and use the target
CPU architecture. PyInstaller bundles still depend on the operating system's
glibc and desktop libraries. See
[PyInstaller's Linux compatibility notes](https://pyinstaller.org/en/stable/usage.html#making-gnu-linux-apps-forward-compatible).

## Align a sample

1. Choose Demo, Images, Video, Basler, or IDS. Select a file/directory for replay,
   or use **Discover** to list cameras. Use **Connect / Open** to begin acquisition.
   Use **File → Open image sequence folder…** for a folder rather than one image.
2. Connecting an IDS or Basler camera automatically applies its highest supported
   pixel clock (IDS), then highest frame rate, then maximum exposure allowed at
   that frame rate. Limits are queried again between changes, before the first
   frame is published. These startup timing defaults take precedence over saved
   timing values. The controls show the camera's actual readback; unsupported
   settings are skipped and failed changes are reported.
   Adjust pixel clock, frame rate, exposure, and gain under **Camera controls**
   where supported. IDS pixel clock is in MHz, frame rate in fps, and exposure
   in milliseconds; gain uses the camera's own numeric units. Click **Apply
   camera settings** to send the values to the camera. Pixel clock offers only
   the camera's supported values. Applying a new clock briefly restarts capture
   and refreshes the frame rate/exposure limits and actual values. Adjust frame
   rate and exposure after applying the clock if their new limits are needed.
3. Use **Capture reference** on the desired frame, or load a saved reference.
   The demo starts with an automatic reference.
4. Watch X, Y, total displacement, tolerance status, and the correction arrow
   while physically adjusting the sample. Positive X points right and positive Y
   points down in the image. The yellow arrow shows the negative measured shift;
   its displayed length is capped to keep it inside the image. Use the numeric
   readout for the full correction magnitude.
5. Inspect the actual overlap, red/cyan overlay, or difference view as needed.
   The **compensated preview** applies a software translation and is labeled:
   visual agreement in that preview does not move the physical sample.

Reference and live views support linked zoom and pan. Enable **Show center
crosshair** beneath the **View** selector to draw cyan guides through the image
center in both views. The guides stay anchored while zooming and panning, and the
toggle is remembered across restarts.

Whole-image matching is the default. A draggable rectangular or circular region
limits the analysis to a feature. A Hann window or circular taper is applied once
during preprocessing.
Template matching is useful for a distinctive region that moves farther through
the frame. Optional half/quarter-resolution analysis reduces processing work;
the reported shift and selected regions remain in native image pixels.

The reference stays fixed while measuring. Changing source geometry clears an
incompatible reference. Weak texture, uncertain matches, insufficient overlap,
and camera disconnection have separate states. The correlation response is a
matching diagnostic, **not a probability**. Residual comparisons are diagnostic
only and do not reject a match. Repeated patterns and large motions still require visual
inspection. **Stale measurement** indicates the last comparison is older than
the allowance for the source frame rate and processing time; the measurement age
is also shown numerically.

**Pause view** pauses display and measurement updates while acquisition continues.
Resume to measure the newest available frame rather than replaying a backlog.
The optional offset-history graph shows the last 600 measurements.

Measurement age starts at application receipt; it is not a sensor-exposure
timestamp. Performance inspection separately reports source wait/read time,
measurement stages, preview generation, and Qt image painting. Acquisition and
measurement run concurrently, so adding their stage times does not give sensor-to-display latency.

In a local check of the supplied 2560 × 1920 horizontal sample pair, median
matching times over five runs after warm-up were approximately 100 ms at native
resolution, 24 ms at half resolution, and 5 ms at quarter resolution. These are
processing measurements, not camera-throughput guarantees. Reduced-resolution
estimates can differ; use native resolution when inspecting fine alignment.

## Optional analytics

Open **Image analytics** from the toolbar or View menu for deeper inspection. It
starts **Off**, with a default Live refresh rate of **1 Hz**, adjustable from
**0.1 to 10 Hz**:

| Mode | Behavior |
| --- | --- |
| Off | Stops optional diagnostic calculations and releases displayed diagnostic data. Live measurement continues. |
| Live | Analyzes the current visible tab at the selected rate. Collapsing/hiding the dock pauses live diagnostic work. |
| Frozen | Holds one exact reference/current-frame/configuration/result pair for inspection while acquisition and measurement continue. **Capture pair** selects another pair. |

Use **Maximize** to open analytics in a separate window with the full screen
available for images and plots. **Restore** lets you resize it freely, and
**Dock** returns it to the main window. The plot grid adds columns as the window
gets wider to reduce scrolling.

Only the visible tab is computed. A pending diagnostic request is replaced by a
newer one; an old job cannot overwrite a more recent selection. Essential match
validity checks remain part of normal measurement even when analytics is Off.

| Tab | Inspection tools |
| --- | --- |
| Image | Native images, pixel readout, histogram, and an adjustable line intensity profile. |
| Preprocessing | Original/processed regions, smoothing, mean removal, and the analysis window. |
| Fourier | Fourier magnitudes, phases, and normalized cross-power diagnostics. |
| Correlation | Reconstructed correlation map, peak detail, and horizontal/vertical peak slices. |
| Alignment | Before/after comparisons, overlap, and residual differences. |
| Performance | Acquisition/measurement rates, frame age, and processing/diagnostic timings. |
| Slot β | Candidate/selected slot regions, tip information, and detection state. |

The Fourier/correlation views reconstruct diagnostics independently of OpenCV's
reported displacement and are labeled accordingly. Keep Frozen inspection open
when comparing several tabs: each tab then describes the same captured pair.
Full-frame arrays can consume substantial memory, so keep live analytics at a
modest refresh rate while aligning.

## Slot detection beta

**Find slot β** proposes dark elongated features within the circular field,
using local contrast, shape, width, and interior-tip evidence. Choose a candidate
when several are plausible. Its proposed measurement region includes the tip
and adjacent texture. Tracking remains relative to the original reference.

Searching, tracking, ambiguous, and lost states make failures visible. Manual
rectangle/circle selection remains available. This is experimental detection:
scratches, multiple similar slots, changing illumination, rotation, occlusion, or
a changed slot shape can confuse it. It does not provide rotation or scale
compensation.

## Save and inspect results

- **Save snapshot…:** a `.shift.zip` bundle containing reference and current
  native PNGs plus measurement/settings metadata.
- **Export inspection…:** the Analytics dock exports the inspected pair together
  with its diagnostic arrays and a PNG of the visible inspection. Alignment
  diagnostics include the compensated/difference arrays. Arrays use portable
  NumPy files; metadata uses JSON, with no Python pickle payloads.
- **File → Export CSV history (last 10,000)…:** export the rolling measurement
  history. Rows include frame/reference/
  configuration identifiers, timestamps, X/Y/distance in pixels, response,
  validity/reasons, overlap, residuals, and processing time. History retains the
  most recent **10,000 measurements**; export during long sessions if earlier
  measurements are needed.
- **Automatic settings:** edits are saved automatically and restored on startup,
  including motor current, joystick channels and polarity, nudge distance/speed,
  calibration, matching settings, tolerance, camera settings, source selection,
  and panel visibility. On Linux the file is
  `$XDG_CONFIG_HOME/measure-shift/settings.json`, or
  `~/.config/measure-shift/settings.json` by default. Closing flushes pending
  edits. Click **Connect hardware** to prepare the motor after startup.
- **Settings profiles:** save/load the matching region, method, scale, quality
  thresholds, tolerance, diagnostic refresh rate, and camera settings in JSON.
  Captured images and references are separate from profiles. After loading a
  profile, use **Apply camera settings** to send the restored values to the camera.

CSV and profiles retain numeric data and settings; use the ZIP bundle when a
reproducible image pair and inspection context are needed. Closing the application
waits for an export already in progress to finish.

## Camera setup

**Basler ace 2:** install Basler's Linux pylon runtime and `pypylon`. Discovery
uses the camera serial number as the device identifier. Acquisition uses
`LatestImageOnly` to avoid an old-frame backlog, converts to Mono8, and copies the
image before releasing the SDK buffer. Use pylon Viewer to confirm the camera is
visible if discovery returns no devices. See
[Basler's installation guide](https://docs.baslerweb.com/pypylon-installation-and-environment-setup)
and [image acquisition documentation](https://docs.baslerweb.com/pypylon-image-acquisition).

**IDS UI-5580 RE C GL R2:** this UI-series camera uses the legacy **IDS Software
Suite/uEye** runtime with `pyueye`, rather than the IDS peak API. Discovery uses
the SDK's numeric **device ID**. The color camera is acquired as 8-bit BGR and
converted to grayscale; mono sensors use Mono8. Three sequence buffers bound
memory use, queued old frames are discarded when supported, each row's SDK pitch
is respected, and locked buffers are always released after copying. Use uEye
Cockpit/Camera Manager to confirm driver and camera visibility. See
[IDS's Python interface documentation](https://en.ids-imaging.com/manuals/ids-software-suite/ueye-manual/4.91.1/en/sdk-program-with-python.html).

Pixel clock is available for IDS cameras that expose it through the uEye SDK.
Exposure, gain, and frame-rate availability vary by camera and acquisition mode;
one unavailable parameter does not disable the others. Disconnecting releases
camera resources. This application deliberately leaves operating-system driver,
network, and firmware configuration to the manufacturer's tools.

## Development and validation

```bash
.venv/bin/python -m pytest
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
```

Fresh-process startup tests also support the actual display backend. Run these
from a Linux desktop session to catch X11 library loading failures that
offscreen tests cannot expose:

```bash
MEASURE_SHIFT_TEST_QPA_PLATFORM=xcb .venv/bin/python -m pytest -q tests/test_startup.py
```

Tests cover signed/subpixel displacement, regions/scaling/reference ownership,
diagnostic consistency and scheduling, source decoding/replay, and camera-buffer
release using fake SDKs. Offscreen GUI tests validate interaction without a
display. Run the demo and a recorded sample on the target computer to assess
actual rates: image size, analysis region, camera transport, preprocessing, and
enabled analytics all affect achievable FPS. Camera FPS and measurement FPS are
shown separately; freshness matters more than processing every captured frame.

Desktop checks with IDS device `1001` acquired 2560 × 1920 frames and verified
pixel clock, frame rate, and exposure changes with actual camera readback and
continued acquisition. Sustained acquisition and Basler hardware still need validation.
Rotation/scale compensation remains outside this version.

## Linear axis: joystick, keyboard, and Align X

Install the optional motion bindings in the same environment as the app:

```bash
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[motion]'
```

Open **Linear axis** from the toolbar or **View** menu. The defaults are Brick
Daemon at `localhost:4223`, Silent Stepper Bricklet 2.0 UID **2eoQ**, and Industrial
Digital In 4 UID **27s4**. Both generations of Digital In 4 are supported. The app
connects when you click **Connect hardware**, applies the selected motor settings,
and enables the motor. Camera-only and Demo use do not require Tinkerforge bindings.

1. Click **Connect hardware**. This applies the selected settings and enables
   movement without starting a move; no separate setup or enable click is needed.
   If a direction is held during connection, release all directions before
   pressing again. Motor current defaults to **800 mA** for
   this axis; you can edit it within the motor's rating and the driver's
   360–1640 mA range. **Apply motor setup** applies settings while leaving
   movement disabled. Driver steps refer to the selected microstep size, not
   necessarily full motor steps.
2. Channel **2** means **left / forward** (positive driver steps) and channel **1**
   means **right / backward** (negative driver steps); channels **0/3** are available
   for other inputs.
   Watch all four **Raw inputs** while testing your joystick, then
   change the left/right channel selections and **Active high** polarity as
   needed. **Invert motor direction** swaps which motor direction means right.
3. Set **Distance / nudge**, **Motor speed**, and **Repeat frequency**. The defaults
   are **1 driver step**, **200 steps/s**, and **2 Hz**. A press performs one
   nudge when sampled; holding repeats at the selected frequency. Releasing finishes
   the current nudge. Inputs are not buffered: presses released before the axis is
   ready are discarded. Each move waits for the previous move to finish, and repeats
   never accumulate.
   **STOP** interrupts movement. Nudges have no acceleration/deceleration ramp.
   Single-step nudges also handle firmware that leaves a nonzero remaining-step
   count after completion: the app verifies the target position and zero velocity,
   requests Stop to clear that state, and verifies fresh telemetry before repeating.
4. Keyboard **Left/Right arrows** behave like the joystick while the main window
   is active. Arrows retain their ordinary behavior in settings fields, menus,
   and dialogs. Leaving the window clears held keys. Keyboard and joystick
   requests in the same direction share one repeat stream; conflicting directions
   stop movement and require both to return to neutral. After Stop or reconnect,
   return to neutral before pressing again.

Under **Digital reference capture**, set **Reference capture channel** to one of
channels **0–3** (or **Disabled**, the default). With hardware connected, a raw
**LOW → HIGH** transition captures the current reference, with at least **one
second** between digital captures. Holding HIGH does not repeat; edges during
the cooldown are discarded. Connecting or selecting an already-HIGH input does
not capture: the input must first be observed LOW. This trigger is independent
of joystick **Active high** polarity and motor enable state. It uses the same
capture behavior as **Capture reference**, including the displayed image while
the view is paused. The channel selection is saved with settings. Inputs are
sampled by the motion worker, so pulses shorter than its polling interval can
be missed.

The repeat frequency controls how often a nudge starts; motor speed controls how
fast each nudge travels. For example, **1 px/nudge**, **10 px/s**, and **2 Hz**
means a 1-pixel move taking roughly 0.1 seconds, repeated at most every 0.5 seconds
while held. A move longer than the repeat interval finishes before another starts.
Changes apply to subsequent nudges. Settings profiles retain these controls,
wiring, and calibration, but loading a profile disconnects motion hardware and
never activates it.

### Calibrate pixels

Use a real IDS or Basler camera with a textured feature visible in the live
image. **Test move (steps)** defaults to **100**;
adjust this signed distance for the available travel, then click
**Run pixel calibration move**. The app
captures a fresh pre-move live image, performs that finite move, and compares the
post-move image with that live baseline after the motor stops and camera queues
are refreshed. Calibration uses the full live image, independently of the saved
reference and matching regions; no saved reference is required or changed.
If the movement is below 2 pixels or the match is uncertain, calibration is
rejected; choose another test distance and retry. No automatic return move is
made.

The signed calibration is `pixels_per_step = change in measured X / change in
motor position`. For example, a +400-driver-step move observed as +20 pixels gives
**+0.05 px/step**. A 1-pixel nudge then uses 20 driver steps. A negative result is
valid and tells automatic alignment that positive motor motion moves the image
left. Successful calibration switches the nudge controls to pixels while keeping
the same effective motor distance and speed. Keep **Invert
motor direction** consistent with the left/right behavior you want from manual
controls; alignment uses the measured calibration sign.

The UI shows the effective driver-step distance and speed after integer rounding;
an amount below one driver step is rejected. Changing units preserves the
effective distance and speed. A step-resolution change clears pixel calibration;
changing camera identity or image dimensions clears pixel calibration. Recalibrate
pixels after changing magnification or the camera/axis geometry. If the camera's
serial cannot be read, pixel calibration is cleared on reconnect.
Older profiles with millimetre nudge settings load as equivalent driver-step
settings.

### Mechanical X alignment

With valid pixel calibration, a fixed reference, and neutral manual controls,
click **Align X**. Each correction is capped at the current nudge distance and
uses the selected motor speed. After each move, the controller waits at least
200 ms (extended for exposure/frame timing), refreshes camera queues, and uses
only new feedback before correcting again. These frame timestamps describe
application acquisition, not sensor exposure timestamps.

**X aligned** requires three fresh, valid measurements within the existing **Tolerance**
in pixels. Y and total displacement remain visible; the axis cannot correct Y.
Demo, image replay, and video replay cannot drive automatic alignment. An uncertain
match may drive coarse X corrections when either **|X| or |Y| exceeds 10 px and
the current tolerance**. When both offsets are at or below that threshold, an
uncertain match pauses further corrections while the controller retries camera
feedback for up to **3 seconds**. Invalid, nonfinite, and stale feedback use the
same recovery window, both between corrections and during movement. The current
finite move may finish, but no new correction uses rejected feedback. A fresh,
usable measurement resumes the operation; continued failure stops it. Invalid
or nonfinite estimates remain blocked at any offset; pixel calibration still
requires valid matches. If X is already within tolerance
but a large Y offset permits an uncertain match, the controller waits for valid
measurements before confirming completion. The match quality details below the
readout identify which check failed. Select a textured region or a distinctive
rectangular template and check that both images share the same feature.
Unrecovered feedback loss, increasing error, ten unsuccessful corrections,
insufficient step resolution, 100 corrections, or 60 seconds stop the attempt
with a status message. Manual input cancels calibration/alignment and requires neutral before
manual movement resumes. Changing the source, reference, matching or motion
settings, pausing, disconnecting, or closing also cancels the operation.

Failed calibration or alignment attempts stop motion while keeping movement
enabled. Once the motor has stopped, retry the operation directly or release
the manual controls and press again to nudge; no **Enable motor** click is needed.
Driver faults, motor move timeouts, and motor setup changes still disable movement
and require **Enable motor** after resolving the problem or applying settings.

Motion commands are finite, and shutdown stops the motor before disabling power.
If the driver is already disabled, Stop and shutdown verify zero velocity and no
remaining steps without sending the stop command, which the firmware rejects
while the driver is off.
There is no homing or end-stop integration in this version. Hardware-free tests
exercise fake SDKs and simulated motor/camera feedback; commissioning on the
actual axis is still required.
