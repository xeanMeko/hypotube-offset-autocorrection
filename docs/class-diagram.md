# Class diagram

## Capture, measurement, and analytics

`MainWindow` creates the shared `LatestValue` frame mailbox. `CaptureWorker` publishes `FramePacket` objects to it, and `MeasurementWorker` and `MainWindow` read it independently. `MainWindow` runs two `AnalyticsWorker` instances, one for diagnostic snapshots and one for display previews.

```mermaid
classDiagram
    class MainWindow
    class CaptureWorker
    class MeasurementWorker
    class AnalyticsWorker
    class LatestValue
    class MeasurementEngine
    class LiveCalibration
    class Source {
        <<protocol>>
    }
    class _PacedSource
    class DemoSource
    class ImageSource
    class VideoSource
    class BaslerSource
    class IDSSource
    class FramePacket
    class ReferenceSnapshot
    class MatchConfig
    class Region
    class MatchResult
    class InspectionPair
    class CalibrationObservation
    class AnalysisOutput
    class DiagnosticSnapshot

    Source <|.. DemoSource
    Source <|.. ImageSource
    Source <|.. VideoSource
    Source <|.. BaslerSource
    Source <|.. IDSSource
    _PacedSource <|-- DemoSource
    _PacedSource <|-- ImageSource
    _PacedSource <|-- VideoSource

    MainWindow o-- CaptureWorker : manages
    MainWindow o-- MeasurementWorker : manages
    MainWindow o-- "2" AnalyticsWorker : diagnostics and preview
    MainWindow *-- LatestValue : owns shared frame mailbox
    CaptureWorker --> Source : reads frames
    CaptureWorker --> LatestValue : writes frame mailbox
    CaptureWorker ..> FramePacket : publishes
    MeasurementWorker --> LatestValue : reads frame mailbox
    MeasurementWorker ..> MeasurementEngine : measures against saved reference
    MeasurementWorker ..> LiveCalibration : measures against live baseline
    MeasurementWorker ..> InspectionPair : publishes
    MeasurementWorker ..> CalibrationObservation : publishes during calibration
    LiveCalibration *-- MeasurementEngine : owns a separate engine
    MeasurementEngine --> ReferenceSnapshot : retains reference
    MeasurementEngine ..> MatchResult : returns
    AnalyticsWorker ..> InspectionPair : analyzes
    AnalyticsWorker ..> AnalysisOutput : publishes

    ReferenceSnapshot --> FramePacket : contains
    InspectionPair --> ReferenceSnapshot
    InspectionPair --> FramePacket : current frame
    InspectionPair --> MatchConfig
    InspectionPair --> MatchResult
    MatchConfig --> Region : reference ROI and optional live ROI
    CalibrationObservation --> FramePacket
    CalibrationObservation --> MatchResult
    AnalysisOutput --> DiagnosticSnapshot : diagnostic jobs
    DiagnosticSnapshot --> InspectionPair
```

## Motion control

`MainWindow` forwards inspection results to `MotionWorker` and routes its fresh-frame requests to `CaptureWorker`. Each request sets an acquisition cutoff and returns a generation that the motion controller uses to validate feedback. During pixel calibration, `MotionWorker` reads `CalibrationObservation` objects from `MeasurementWorker` so feedback is measured against the pre-move live baseline. The controller uses `DirectXSearch` for automatic X alignment.

```mermaid
classDiagram
    class MainWindow
    class CaptureWorker
    class MeasurementWorker
    class InspectionPair
    class CalibrationObservation
    class MotionWorker
    class MotionController
    class DirectXSearch
    class JoystickPort {
        <<protocol>>
    }
    class MotionHardware {
        <<protocol>>
    }
    class WiredJoystick
    class TinkerforgeMotionHardware
    class SilentStepperBench
    class JoystickState
    class MotionConfig
    class MotionContext
    class MotionFeedback
    class MotionAction
    class MotionStatus
    class ReferenceCaptureRequest

    MainWindow o-- MotionWorker : manages
    MainWindow ..> CaptureWorker : forwards fresh-frame requests
    MainWindow ..> InspectionPair : forwards to motion worker
    MainWindow ..> MotionStatus : reads
    MotionWorker *-- MotionController : owns
    MotionWorker --> MainWindow : calls freshness callback
    MotionWorker --> MeasurementWorker : starts and stops live calibration
    MotionWorker ..> CalibrationObservation : reads calibration feedback
    MotionWorker ..> MotionFeedback : builds feedback
    MotionWorker ..> MotionAction : executes
    MotionWorker ..> MotionStatus : publishes
    MotionFeedback ..> InspectionPair : built from measurement results
    MotionController --> MotionConfig
    MotionController --> MotionContext
    MotionController --> MotionFeedback
    MotionController --> JoystickState
    MotionController ..> MotionAction : plans
    MotionController ..> MotionStatus : creates
    MotionController *-- "0..1" DirectXSearch : runs during X alignment
    MotionStatus --> MotionConfig
    MotionStatus --> "0..1" ReferenceCaptureRequest

    MotionWorker --> JoystickPort : polls and commands
    JoystickPort <|.. WiredJoystick
    JoystickPort ..> JoystickState : returns
    WiredJoystick --> MotionHardware : reads inputs and drives relays
    MotionHardware <|.. TinkerforgeMotionHardware
    TinkerforgeMotionHardware *-- "0..1" SilentStepperBench : creates in bench mode
    WiredJoystick --> "0..1" SilentStepperBench : drives in bench mode
```
