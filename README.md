# FormPro2

Real-time barbell back squat form analysis from a live camera. The pipeline reads a webcam
stream, extracts 3D pose landmarks, normalizes them against the lifter's own body
proportions, compares the result to a labelled reference dataset, and draws corrective cues
onto the live feed.

Scope for v1 is the barbell back squat only. Thresholds, error classes and phase
segmentation are hardcoded for the squat; generalizing to other lifts is deferred.

## Architecture

The system is a one-way pipeline. Each stage owns one transformation and depends only on
the stage above it, so any stage can be swapped or tested in isolation.

```
 camera --> capture.py --> pose_estimator.py --> kinematics.py --> form_analyzer.py --> overlay
           (raw BGR       (33 3D landmarks,      (body-proportion   (compare vs.         (draw)
            frames,        metric space)          normalized         reference set)
            drop-old)                             angles/ratios)
                                                                     ^
                                                  dataset_loader.py -+
                                                  (offline reference sequences,
                                                   run through the same kinematics.py)
```

`kinematics.py` is the single normalization path, used by both the live stream and the
offline dataset loader. If live frames and reference frames were normalized by different
code, every comparison downstream would measure the difference between two implementations
rather than between two squats.

### Directory layout

```
FormPro2/
├── app.py                       Phase 6: main loop and UI
├── requirements.txt
├── configs/
│   └── squat.yaml               all tunables; no magic numbers in code
├── models/                      .task model binaries (git-ignored, see fetch_model.py)
├── data/
│   ├── reference/               Phase 4: labelled reference sequences (JSON)
│   └── recordings/              git-ignored raw captures
├── formpro/
│   ├── schema.py                data contracts and landmark indices
│   ├── config.py                typed config loading
│   ├── filters.py               One-Euro landmark smoothing
│   ├── capture.py               threaded camera, drop-old buffering
│   ├── pose_estimator.py        BlazePose backend behind a Protocol
│   ├── video_processor.py       orchestrates capture into pose
│   ├── kinematics.py            Phase 3: segment lengths, ratios, joint angles
│   ├── dataset_loader.py        Phase 4: reference set ingestion
│   ├── phases.py                Phase 4: eccentric/concentric segmentation
│   └── form_analyzer.py         Phase 5: DTW / similarity comparison engine
├── scripts/
│   ├── fetch_model.py           downloads the BlazePose .task binary
│   └── smoke_test_pose.py       ingestion diagnostic viewer, not the real UI
└── tests/
```

## Setup

Requires CPython 3.9 or newer. Verified on 3.13 with MediaPipe 1.0.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/fetch_model.py --variant heavy
```

MediaPipe 1.0 removed the legacy `mp.solutions.pose` API, so FormPro2 targets the current
Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`). That API loads weights from a
local `.task` file, which is why `fetch_model.py` exists.

Verify the camera and pose stage end to end:

```powershell
python scripts/smoke_test_pose.py
```

You should see a wireframe tracking you plus a readout of capture FPS, inference latency
and dropped-frame count. Press `q` to quit.

## Camera placement

The system assumes a **45-degree anterior oblique** view: front-diagonal, roughly waist
height. This is a fixed constraint, not a suggestion, because the Phase 3 thresholds are
calibrated against it.

The reasoning is that neither orthogonal view works alone. A pure lateral view suffers from
occlusion, since weight plates block the hips and knees, and it cannot see knee valgus at
all. A pure frontal view has to derive back angle and hip depth from the Z axis, which is
BlazePose's least reliable output. The 45-degree angle gives a usable 2D projection of both
the sagittal plane (flexion and extension) and the frontal plane (valgus and varus).

Two consequences for Phase 3:

- Joint angles are computed from BlazePose 3D vectors, but weighted toward X and Y. Z is
  used only where nothing else is available.
- Knee valgus is not measured as a 3D angle. It is measured as the horizontal distance
  between the knee joints, normalized against hip width, which keeps it in the well-observed
  X/Y plane.

## Reference dataset format

Reference data is JSON containing pre-computed, normalized angles. Raw pixel coordinates are
not used, so the dataset stays invariant to camera resolution, subject distance and lifter
height.

```json
{
  "metadata": {
    "exercise": "barbell_back_squat",
    "camera_angle": "45_oblique",
    "dataset_type": "reference_optimal"
  },
  "subject_proportions": {
    "femur_to_torso_ratio": 1.12
  },
  "frames": [
    {
      "frame_id": 1,
      "phase": "eccentric",
      "angles": {
        "hip_flexion": 175.2,
        "knee_flexion": 170.5,
        "ankle_dorsiflexion": 90.0,
        "back_to_vertical": 10.5,
        "knee_to_hip_width_ratio": 1.05
      },
      "form_label": "optimal_form"
    }
  ]
}
```

`metadata.camera_angle` is validated on load. A file recorded at a different angle is
rejected rather than silently compared against 45-degree thresholds.

`subject_proportions.femur_to_torso_ratio` is what makes cross-body comparison possible. It
is the reference subject's build, and Phase 5 has to account for the gap between it and the
live lifter's ratio, rather than assuming both were built the same way.

### Angle conventions

All angles are in degrees and are **included angles between two segments, where 180 means
fully extended**. This is not the clinical range-of-motion convention, where a straight knee
is 0 degrees of flexion. The sample frame is a lifter who has just begun descending, which
is why the values sit near the extended end:

| Field | Standing | Bottom of squat | Meaning |
|---|---|---|---|
| `hip_flexion` | ~175 | decreases | torso relative to thigh |
| `knee_flexion` | ~170 | decreases | thigh relative to calf |
| `ankle_dorsiflexion` | ~90 | decreases | calf relative to foot, 90 is neutral shin |
| `back_to_vertical` | ~10 | increases | torso away from vertical, 0 is upright |
| `knee_to_hip_width_ratio` | ~1.05 | decreases on valgus | inter-knee horizontal distance over hip width |

`knee_to_hip_width_ratio` is the valgus measure described under camera placement. It stays
in the X/Y plane rather than depending on Z, and being a ratio it is already normalized
against the lifter's own hip width.

Phase 3 will emit these exact field names and this exact convention so live and reference
frames are directly comparable without an adapter layer between them.

### Open questions on the schema

These do not block Phase 3, which computes the angles above regardless. They do shape
Phase 4 and Phase 5.

1. **No temporal field.** `frame_id` is an ordinal with no timestamp and no frame rate.
   `error_good_morning` is defined by the hips rising faster than the shoulders, which is a
   rate and cannot be computed from ordinals alone if capture rate ever varies. Suggested
   fix: `timestamp_ms` per frame, or `fps` in `metadata` if capture is reliably fixed-rate.
2. **Bilateral collapse.** `hip_flexion`, `knee_flexion` and `ankle_dorsiflexion` are single
   scalars, but a lifter has two of each and asymmetry is diagnostic. Are these averaged, or
   taken from the camera-near side, which the 45-degree view observes best? Per-side fields
   would preserve the signal.
3. **`phase` vocabulary.** `eccentric` is shown. The live segmenter must emit the identical
   vocabulary, so the full set needs stating, including whether standing and bottom holds
   get their own values or fold into the two movement phases.
4. **`dataset_type` versus `form_label`.** One is file-level, one is per-frame. May a single
   file mix frame labels? A rep that descends cleanly and breaks on the ascent is the normal
   presentation of `error_good_morning`, so mixed labelling is likely wanted.
5. **Coverage of `femur_to_torso_ratio`.** Dynamically adjusting the back-angle threshold by
   build requires reference subjects at several ratios. If every file records 1.12 there is
   only one anchor point and nothing to interpolate between.

A `schema_version` field in `metadata` would also be worth adding before the set grows.

## Coordinate conventions

Two landmark spaces come out of the pose stage and they are not interchangeable:

| Space | Field | Units | Origin | Axes | Use for |
|---|---|---|---|---|---|
| Image | `PoseFrame.image_xyz` | normalized `[0,1]` | top-left of frame | X right, Y down | drawing only |
| World | `PoseFrame.world_xyz` | metres | midpoint of the hips | X right, Y up, Z toward camera | all biomechanics |

MediaPipe emits world landmarks Y-down and Z-away. FormPro2 negates both at ingestion so
world space is a conventional right-handed Y-up system. Every downstream module assumes
Y-up, including angles, depth checks and phase segmentation. `image_xyz` keeps the raw
MediaPipe orientation because OpenCV draws in Y-down.

The capture stage never mirrors the frame. Mirroring before inference would swap the
lifter's anatomical left and right, which would invert per-side findings such as knee
valgus. Display mirroring is applied at render time only, in Phase 6.

## Build status

| Phase | Module | Status |
|---|---|---|
| 1 | Project setup and architecture | done |
| 2 | Camera and pose ingestion | done |
| 3 | Biomechanical normalization engine | not started |
| 4 | Dataset ingestion and preprocessing | not started, schema received |
| 5 | Real-time comparison logic | not started |
| 6 | UI and feedback overlay | not started |
