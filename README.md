# Measure Shift

A Linux desktop application for aligning live images against a fixed reference,
with optional joystick/keyboard motor nudging and mechanical X alignment. It
measures X/Y translation in native image pixels using phase
correlation, with optional image analytics. The notebooks and sample images
remain available.

The interface opens in German. Choose **Einstellungen → Sprache → English** to
switch immediately. The app remembers your choice for the next launch.

For everyday operation, see the [German user manual (PDF)](docs/benutzerhandbuch-de.pdf).
It assumes a configured camera and axis controller. Relay contacts request
axis nudges, and digital inputs can trigger reference capture and X alignment.
Edit the [LaTeX source](docs/benutzerhandbuch-de.tex), or use the
[complete LaTeX source archive](docs/benutzerhandbuch-de-latex.zip).
With pdfLaTeX installed, rebuild the PDF and archive using
`python scripts/build_user_manual.py`. See [build instructions](docs/LATEX-ANLEITUNG.txt).

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

The build machine needs Brick Daemon at `/usr/bin/brickd`. The script creates
a separate system-Python environment in `build/linux-venv`,
installs the app with camera and motion bindings plus PyInstaller, and bundles
the dynamically imported hardware modules. Set `MEASURE_SHIFT_BUILD_PYTHON` to
another Python executable if needed. The build does not use the existing
development `.venv`.

After the first build, use `bash scripts/build_linux.sh --skip-install` to
rebuild the current source without accessing the package index, provided the
dependencies have not changed.

Copy the entire `dist/measure-shift/` directory, or extract the generated
`dist/measure-shift-linux-<architecture>.tar.gz` archive on the target computer.
The bundle includes Python, the app's Python dependencies, and Brick Daemon.
Running `measure-shift` starts the bundled daemon on `localhost:4223` if the
local port is free and stops that subprocess on normal exit. The target needs
`libusb-1.0-0` on Debian or Ubuntu and the desktop user's account needs read and
write access to the Tinkerforge USB device. At launch, Measure Shift checks the
saved connection settings and connects the hardware automatically. If a local
Master Brick (USB ID `16d0:063d`) needs access, the desktop immediately asks for
administrator approval through polkit. After approval, Measure Shift applies a
persistent udev rule and connects. You can cancel and continue using the app.
For a Brick attached later, **Connect hardware** also offers **Enable Master
Brick access** if discovery encounters the same permission issue. If discovery still fails,
unplug and reconnect the Brick, then click **Connect hardware**.

Camera drivers and manufacturer
runtimes still need to be installed on the target computer. Brick Daemon source
is available from [Tinkerforge](https://github.com/Tinkerforge/brickd).
The resolved build dependencies are recorded in the bundle's
`build-requirements.txt`.

Build on the oldest Ubuntu release you intend to support and use the target
CPU architecture. PyInstaller bundles still depend on the operating system's
glibc and desktop libraries. See
[PyInstaller's Linux compatibility notes](https://pyinstaller.org/en/stable/usage.html#making-gnu-linux-apps-forward-compatible).

## Align a sample

1. Choose Demo, Images, Video, Basler, or IDS. Select a file/directory for replay,
   or use **Tools → Discover cameras** to list cameras. Use **Connect / Open** to begin acquisition.
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
4. Watch X and Y displacement, tolerance status, and the correction arrow
   while physically adjusting the sample. Positive X points right and positive Y
   points down in the image. The yellow arrow shows the negative measured shift;
   its displayed length is capped to keep it inside the image. Use the numeric
   readout for the full correction magnitude. X/Y readouts show one decimal
   place and average valid readings over 150 ms to reduce jitter. Tolerance
   status, motion control, analytics, and exports use the original measurements.
5. Inspect the actual overlap, red/cyan overlay, or difference view as needed.
   The **compensated preview** applies a software translation and is labeled:
   visual agreement in that preview does not move the physical sample.

Reference and live images share the full width of the main window. Alignment
readouts, image settings, and camera controls sit in a row beneath the images.
The two image panes can be resized with the divider between them.
Under **Image settings → Display scale**, choose **1:1**, **1:2**, **1:4**,
**1:6**, or **1:8**. **1:1** shows one image pixel per screen pixel; **1:2**
displays both dimensions at half size. This changes only the display size of
both images and preserves their proportions. Pixels, regions, measured shifts,
and exports keep native image coordinates. The display scale is remembered
across restarts and stays constant when resizing the window. Mouse zoom can
show **Custom**; **View → Fit images** shows **Fit** and displays the whole image.

Reference and live views support linked zoom and pan. Enable **Show crosshair**
beneath the **View** selector to draw cyan guides in both views. Enable
**Move crosshair (click image)**, then left-click either image to place both
guides at the same native image coordinates. Turn movement off to lock the
position. **Save crosshair** remembers the position across restarts and locks it;
unsaved moves last only for the current session. **Center** returns the guides
to the image center; save again to remember that choice. Saved positions are
also included in settings profiles. With smaller images, guides stay within the
image edges while retaining the saved coordinates for larger images. Zooming
and panning preserve the image position, and visibility is remembered.

Whole-image phase correlation is the default. A draggable rectangular or
circular region limits the analysis to a feature. A Hann window or circular
taper is applied once during preprocessing. Analysis uses native resolution by
default. Existing settings profiles can still specify half/quarter-resolution
analysis to reduce processing work; the reported shift and selected regions
remain in native image pixels.

The reference stays fixed while measuring. Changing source geometry clears an
incompatible reference. Weak texture, uncertain matches, insufficient overlap,
and camera disconnection have separate states. The correlation response is a
matching diagnostic, **not a probability**. Residual comparisons are diagnostic
only and do not reject a match. Repeated patterns and large motions still require visual
inspection. **Stale measurement** indicates the last comparison is older than
the allowance for the source frame rate and processing time.

**Tools → Pause view** pauses display and measurement updates while acquisition continues.
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

Open **Tools → Image analytics** or **View → Image analytics** for deeper
inspection in a separate, resizable window. It starts **Off**, with a default
Live refresh rate of **1 Hz**, adjustable from **0.1 to 10 Hz**:

| Mode | Behavior |
| --- | --- |
| Off | Stops optional diagnostic calculations and releases displayed diagnostic data. Live measurement continues. |
| Live | Analyzes the current visible tab at the selected rate. Hiding the analytics window pauses live diagnostic work. |
| Frozen | Holds one exact reference/current-frame/configuration/result pair for inspection while acquisition and measurement continue. **Capture pair** selects another pair. |

Use **Maximize** to give analytics the full screen. **Restore** lets you resize
it freely, and **Dock** places it inside the main window if preferred. The plot
grid adds columns as the window gets wider to reduce scrolling.

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

The Fourier/correlation views reconstruct diagnostics independently of OpenCV's
reported displacement and are labeled accordingly. Keep Frozen inspection open
when comparing several tabs: each tab then describes the same captured pair.
Full-frame arrays can consume substantial memory, so keep live analytics at a
modest refresh rate while aligning.

## Save and inspect results

- **Save snapshot…:** a `.shift.zip` bundle containing reference and current
  native PNGs plus measurement/settings metadata.
- **Export inspection…:** the analytics window exports the inspected pair together
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
  including relay timing and digital input channels and polarity,
  calibration, matching settings, tolerance, camera settings, source selection,
  and panel visibility. On Linux the file is
  `$XDG_CONFIG_HOME/measure-shift/settings.json`, or
  `~/.config/measure-shift/settings.json` by default. Closing flushes pending
  edits. Hardware connects automatically after these settings are restored.
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

## Motion setup

Install the optional motion bindings in the same environment as the app:

```bash
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[motion]'
```

In **Tools → Motion Setup**, set
the Brick Daemon host and port, Dual Relay UID and optional Digital In UID.
Defaults are `localhost:4223`, relay **2j7T**, and Digital In **27s4**. The
app connects at startup and waits for the controls to be released. Use
**Connect hardware** to disconnect or retry. Loading a settings profile
disconnects hardware until you reconnect.

| Default direction | Dual Relay contact | Digital In |
| --- | --- | --- |
| Right | A0 | 1 |
| Left | A1 | 2 |

The physical joystick, arrow keys, **Align X**, and **Run pixel calibration** send the same relay
nudge. The relay releases on its own timer. Open **Settings → Relay pulse timing** to see a live preview. Set **Pulse duration** (default
80 ms) and **Release interval** (default 100 ms) to suit the receiving
controller, allowing it to finish the nudge before the next press. **Nudge
repeat rate** caps how often the app sends pulses (default 2 Hz). Hold the
joystick left or right, or hold an arrow key in the active main window, to repeat.
Moving the joystick during automatic alignment cancels the operation. The app filters its relay echoes so a nudge
cannot trigger a second pulse. **STOP** releases the relays and ends the
alignment attempt. The receiving controller handles movement already accepted.

With a Silent Stepper connected on the bench, each **Align X** correction requires
feedback from Digital In. Both direction inputs must start idle. The app pulses
the relay and commands one motor nudge when it observes the matching input close
during the pulse. The next correction waits for contact release and motor
completion. If the app misses the closure, it stops alignment and reports the
input channel. The configured press must be long enough for a digital input read
to capture it. This feedback requirement applies only to bench **Align X**.

### Direct X alignment

With a live IDS or Basler camera, a reference, and released manual controls,
click **Align X** or raise the configured **Align X** digital input from LOW
to HIGH. The default channel is **3**. Holding it HIGH starts one attempt.
Starting with the input HIGH, or pressing while the app is busy, requires
release and a new rising edge. Configure or disable the trigger under
**Digital inputs → Apply digital inputs**. If you assign a separate
**Reference capture** channel, each rising edge can capture a reference, with
at least one second between captures.

If X is outside tolerance, alignment starts with one right relay nudge. It
waits for relay release and camera settling, then compares the new signed
`dx` with the previous measurement. It continues in the same direction
when the error shrinks, and reverses when the error grows or crosses zero.
The app checks for repeated endpoints and stops after 10 nudges without
progress, 100 corrections, or 60 seconds. Three fresh, valid measurements
inside the configured tolerance confirm alignment. Camera feedback uses a
freshness barrier and at least 200 ms of settling after the nudge interval.
Manual input, Stop, disconnect, or a change in camera or reference cancels
alignment.

### Optional pixel calibration

In **Motion Setup → Pixel calibration**, **Run pixel calibration** measures the signed X displacement per right relay
press for the live camera. Set **Test presses** to the number of nudges.
Positive counts pulse right and negative counts pulse left. The app compares
settled frames against a live baseline. The total X movement must reach at
least 2 pixels. Two right nudges that move the image 6 pixels left give
**−3 px/right press**. The axis stays at the endpoint. You can also enter an
existing measurement using **Known X / right press** and **Use known distance**.

Alignment works independently of the saved pixel measurement. Old profiles
keep their relay and input settings. If a profile specifies a test in driver
steps, the app preserves the direction as one relay press.
