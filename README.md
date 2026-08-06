# FormPro2

Real-time barbell back squat form analysis from a live camera. The pipeline reads a webcam
stream, extracts 3D pose landmarks, normalizes them against the lifter's own body
proportions, compares the result to a labelled reference dataset, and draws corrective cues
onto the live feed.

Scope for v1 is the barbell back squat only. The error classes, rep-cycle vocabulary
and error signatures are specific to the squat; generalizing to other lifts is
deferred. Note that the numeric bounds are not hardcoded anywhere: they are derived
from the reference corpus at runtime. See Analysis.

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
├── app.py                       main loop
├── requirements.txt
├── configs/
│   └── squat.yaml               all tunables; no magic numbers in code
├── models/                      .task model binaries (git-ignored, see fetch_model.py)
├── data/
│   ├── reference/               labelled reference sequences (JSON)
│   └── recordings/              git-ignored raw captures
├── formpro/
│   ├── schema.py                data contracts and landmark indices
│   ├── config.py                typed config loading
│   ├── filters.py               One-Euro landmark smoothing
│   ├── capture.py               threaded camera, drop-old buffering
│   ├── pose_estimator.py        BlazePose backend behind a Protocol
│   ├── video_processor.py       orchestrates capture into pose
│   ├── kinematics.py            segment lengths, ratios, joint angles, features
│   ├── phases.py                rep cycle segmentation from hip trajectory
│   ├── dataset_loader.py        reference set ingestion and corpus indexing
│   ├── form_analyzer.py         corpus-derived bounds, error detection, DTW
│   └── overlay.py               skeleton and HUD rendering
├── scripts/
│   ├── fetch_model.py           downloads the BlazePose .task binary
│   ├── smoke_test_pose.py       ingestion diagnostic viewer, not the real UI
│   └── validate_dataset.py      checks a reference directory against the contract
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
height. This is a fixed constraint, not a suggestion: the reference corpus is recorded
at this angle, and live data is only comparable to it from the same viewpoint.

The reasoning is that neither orthogonal view works alone. A pure lateral view suffers from
occlusion, since weight plates block the hips and knees, and it cannot see knee valgus at
all. A pure frontal view has to derive back angle and hip depth from the Z axis, which is
BlazePose's least reliable output. The 45-degree angle gives a usable 2D projection of both
the sagittal plane (flexion and extension) and the frontal plane (valgus and varus).

Two consequences for the kinematics engine:

- Joint angles are computed from BlazePose 3D vectors, but weighted toward X and Y. Z is
  used only where nothing else is available.
- Knee valgus is not measured as a 3D angle. It is measured as the horizontal distance
  between the knee joints, normalized against hip width, which keeps it in the well-observed
  X/Y plane.

## Normalization

Three things make a 5'2" lifter comparable to a corpus recorded by a 6'4" one. All of
them happen in `kinematics.py`, which is the only normalization path in the system.

1. Angles instead of positions. A joint angle is already scale-free.
2. Ratios instead of distances. Knee separation is divided by that lifter's own hip
   width, hip height by that lifter's own leg length. Both come out dimensionless.
3. Proportions as context, not correction. `femur_to_torso_ratio` never adjusts a
   measurement. It is carried alongside it, because a 45-degree forward lean is sound on
   a long-femur lifter and a good-morning on a short-femur one. The analyzer interpolates
   the acceptable band from the corpus using this ratio; see Analysis below.

Segment lengths are calibrated over a rolling window and frozen once enough frames have
accumulated, rather than recomputed per frame. A per-frame estimate would let the
tolerance band wobble mid-rep on measurement noise alone.

### The depth axis is treated differently for angles and for lengths

Z is BlazePose's least reliable output, and the two uses have opposite requirements.

Angles attenuate Z by `kinematics.z_weight`. At `1.0` these are true 3D angles; at `0.0`
they are projected onto the image plane. Neither extreme suits a 45-degree view: full 3D
inherits the depth noise, while pure projection systematically under-reads flexion
because the sagittal plane sits at 45 degrees to the image plane. The default of `0.6`
sits between them.

Segment lengths use full 3D with no attenuation, because attenuating Z would shorten any
limb pointing toward the camera, and correcting exactly that foreshortening is what the
depth axis is for. Depth noise is handled there by taking a median over the window.

This leaves a known, systematic bias in the angles, which is acceptable only because the
bias is identical on both sides of the comparison. That holds if the reference corpus was
produced by this same engine at this same camera angle. **If your corpus came from a
different pipeline computing true 3D angles, it will sit at a constant offset from live
data and every threshold will be wrong by that offset.** `KinematicFrame.to_json_frame()`
exists so reference files can be generated from this engine directly; see
`tests/test_integration.py` for a worked example.

The valgus metric sidesteps the problem entirely. `knee_to_hip_width_ratio` uses X
separation only: at any camera yaw the knee axis and the hip axis are foreshortened by
the same cosine, so dividing one by the other cancels it. This is verified against
rotated poses in the test suite.

## Rep cycle

`phases.py` segments the lift by tracking hip height over time. The vocabulary is shared
verbatim with the reference dataset, since the analyzer aligns like against like:

| Phase | Meaning |
|---|---|
| `setup` | standing, un-racking, bracing |
| `eccentric` | descent |
| `bottom` | the hole; velocity approaching zero |
| `concentric` | ascent |
| `recovery` | return to standing |

Velocity is a least-squares slope over a fixed time window using capture timestamps,
never a difference over frame counts. The capture stage deliberately drops frames when inference
falls behind, so consecutive frames are not evenly spaced and a per-frame delta would
read a dropped frame as sudden acceleration. Both position and velocity are divided by
the lifter's own leg length, so one threshold set covers every body size.

Two safeguards worth knowing about:

- Every transition must hold for `min_dwell_frames` before it commits. A flickering
  phase is worse than a lagging one, because the analyzer keys its comparison window
  off the phase.
- A rep may only begin from a standing position the segmenter actually observed. After a
  tracking dropout, or at startup with the lifter already mid-squat, the partial rep is
  discarded rather than scored on the fraction that happened to be visible.

## Reference dataset format

Reference data is JSON containing pre-computed, normalized angles. Raw pixel coordinates
are not used, so the dataset stays invariant to camera resolution, subject distance and
lifter height.

```json
{
  "metadata": {
    "exercise": "barbell_back_squat",
    "camera_angle": "45_oblique_anterior",
    "dataset_type": "reference_good_morning_error",
    "fps_target": 30
  },
  "subject_proportions": {
    "femur_to_torso_ratio": 1.12,
    "tibia_to_femur_ratio": 0.85
  },
  "frames": [
    {
      "frame_id": 142,
      "timestamp_ms": 4733,
      "phase": "concentric",
      "angles": {
        "camera_near": {
          "hip_flexion": 110.5,
          "knee_flexion": 125.0,
          "ankle_dorsiflexion": 85.2,
          "back_to_vertical": 45.1
        },
        "camera_far": {
          "hip_flexion": 111.0,
          "knee_flexion": 124.5,
          "ankle_dorsiflexion": 86.0,
          "back_to_vertical": 45.3
        },
        "global": {
          "knee_to_hip_width_ratio": 0.95
        }
      },
      "form_label": "error_good_morning"
    }
  ]
}
```

Validate a directory before trusting it. This reports every bad file in one pass, rather
than stopping at the first:

```powershell
python scripts/validate_dataset.py data/reference
```

### Angle conventions

All angles are in degrees and are included angles between two segments, where 180 means
fully extended. This is not the clinical range-of-motion convention, where a straight
knee is 0 degrees of flexion.

| Field | Standing | Bottom of squat | Meaning |
|---|---|---|---|
| `hip_flexion` | 180 | decreases | torso relative to thigh |
| `knee_flexion` | 180 | decreases | thigh relative to calf |
| `ankle_dorsiflexion` | 90 | decreases | shin relative to the foot's long axis |
| `back_to_vertical` | 0 | increases | torso away from vertical |
| `knee_to_hip_width_ratio` | ~1.0 | decreases on valgus | inter-knee X separation over hip width |

`camera_near` is the side facing the camera, resolved per frame from Z depth with
hysteresis so it cannot flip mid-rep. It is the reliable side for sagittal measures.
`camera_far` is partially occluded at 45 degrees and enters the analyzer's distance metric
at reduced weight (`kinematics.camera_far_weight`) rather than being trusted equally or
discarded. Anatomical left and right are preserved throughout; the near/far split is a
viewing relationship layered on top.

`back_to_vertical` is per side because it is measured from that side's own shoulder-hip
vector, which makes asymmetric torso lean visible rather than averaged away.

### Loader behaviour

Validation is strict and failures raise rather than warn. A silently skipped reference
file does not break anything visibly, it just removes an anchor point, and the tolerance
band then narrows around whichever builds happened to survive. Rejected on load:

- `camera_angle` outside the accepted set. `45_oblique` is accepted as a legacy alias
  with a warning; anything else fails, because comparing against a differently-framed
  recording presents as consistent form error.
- Non-monotonic timestamps, or a gap wider than `dataset.max_timestamp_gap_ms`. Velocity
  is undefined on non-monotonic time, and a rep cannot be reconstructed across a long
  dropout.
- Any `phase` or `form_label` outside the enums, any angle outside 0-180, any null.

`form_label` is evaluated per frame, so a single file transitions between labels partway
through. This is the normal presentation of a good-morning squat: a clean eccentric that
breaks once the concentric begins. `dataset_type` records the file's dominant intent.
`ReferenceSequence.label_transitions()` exposes the boundary directly.

`load_corpus` reads a directory into a `ReferenceCorpus` indexed by
`femur_to_torso_ratio`, with `bracketing()` returning the nearest reference below and
above a live lifter's build so the analyzer interpolates rather than snapping to the
nearest. Either side comes back `None` when the lifter falls outside the corpus, which
the analyzer treats as extrapolation. A corpus spanning less than `dataset.min_ratio_span` warns:
the band is then nominally dynamic but effectively fixed to one body type.

## Analysis

`form_analyzer.py` is an inference engine over the corpus. It holds no baseline
thresholds and no fallback constants: every bound it applies is computed at runtime from
reference frames labelled `optimal_form`, then adjusted to the live lifter's build. An
empty or unusable corpus is a startup failure, not a degraded mode, because an analyzer
that quietly falls back to hardcoded numbers is indistinguishable from one that is
working and passes every rep.

### How a bound is produced

1. Reference frames labelled `optimal_form` are pooled by subject build, giving one
   threshold profile per distinct `femur_to_torso_ratio` in the corpus. Error files
   contribute too: the clean eccentric of a good-morning rep is still optimal evidence.
   Frames labelled with an error never contribute, since folding them in would widen the
   acceptable band to admit the very thing being detected.
2. Within a profile, frames are grouped by phase and each feature gets a percentile band.
   Bands are per phase because the acceptable knee angle at the bottom has nothing to do
   with the acceptable knee angle during setup.
3. At evaluation time the lifter's ratio goes to `ReferenceCorpus.bracketing()`. Between
   two profiles the bands blend by distance. Outside the corpus they are projected along
   the trend of the two nearest profiles, logging `[Extrapolation Warning]`, and the HUD
   marks the bounds as extrapolated. A lifter with an unusual build is exactly the case
   where a corpus-average bound would be wrong, so the projection continues rather than
   clamping.

If the corpus holds only one build, no trend exists to project. The analyzer says so and
uses that single profile, rather than inventing a slope from one point.

### Thresholds are evidence, interpretation is domain knowledge

`ERROR_SIGNATURES` maps a feature deviating in a given direction during a given phase
onto a named error. That table contains no numbers. It says *which* error a deviation
means, never *how far* is too far, and a test asserts no float ever appears in it. The
magnitudes come entirely from the corpus. This separation is what lets the bounds stay
fully dynamic while the cues stay specific enough to act on.

Two signatures are worth explaining:

- **Hips rising too fast.** The schema carries no shoulder height, and does not need to.
  If the hips outrun the shoulders the torso necessarily becomes more horizontal, so a
  back angle that is both above the corpus band and *still opening* during the ascent is
  precisely that failure. The rate requirement is what separates it from a static lean.
- **Heel lift.** As the heel rises, the heel-to-toe axis tilts away from the shin and the
  measured ankle angle *opens*, whereas normal dorsiflexion during a descent closes it.
  An unexpectedly large ankle angle at the bottom is therefore the signature.

### Feature scaling

A distance metric over raw features would discard the only frontal-plane signal
available. A full valgus collapse moves `knee_to_hip_width_ratio` by around 0.2, against
angle deviations of tens of degrees, so the frontal signal would vanish inside sagittal
noise.

`kinematics.feature_weights()` fixes this from a stated biomechanical equivalence rather
than a tuned constant: a 0.1 deviation in the width ratio is treated as about as severe
as 15 degrees of joint-angle deviation, giving a scale factor of 150. Both numbers live
in `configs/squat.yaml` as the equivalence itself, so the claim can be argued with
directly instead of appearing as an unexplained multiplier. The camera-far side carries
reduced weight in the same dictionary.

The practical effect is that findings from different features can be ranked against each
other on one severity scale, so the HUD shows the worst problem first.

### Sequence classification

Alongside the per-frame band checks, completed eccentric and concentric segments are
compared against labelled reference segments by weighted DTW, with a Sakoe-Chiba band
bounding how far the alignment may warp. Without that band a slow live descent could
align against a fast reference ascent and report a small distance between two quite
different movements.

This is the check that can catch a coordination fault no single frame violates: a rep
where every angle stays inside its band but the relationship between them is wrong. A
classification is only reported when the best label beats the runner-up by
`analyzer.min_confidence_margin`; otherwise it is withheld as inconclusive rather than
resolved by coin toss.

## Running it

```powershell
python app.py
```

Loads the corpus, opens the camera, and renders the live feed. Keys: `q` or `Esc` to
quit, `r` to reset the session for a new lifter or set.

The HUD shows the calibrated `femur/torso` and `tibia/femur` ratios, the current rep
phase, rep count, normalized depth, throughput telemetry, and where the active bounds
came from. The camera-near side of the skeleton is drawn in green and the camera-far side
in blue, so orientation can be confirmed at a glance; occluded joints drop to grey.
Coaching cues appear along the bottom, green for `OPTIMAL FORM` and red for specific
errors, worst first, each with the observed value and the corpus band it violated.

Useful flags:

| Flag | Effect |
|---|---|
| `--source PATH` | replay a video file instead of the camera |
| `--reference DIR` | corpus directory, overriding the config |
| `--no-mirror` | disable display mirroring |
| `--verbose` | debug logging, including side-switch decisions |

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
valgus. Display mirroring is applied at render time only, in `overlay.py`.

## Build status

| Phase | Module | Status |
|---|---|---|
| 1 | Project setup and architecture | done |
| 2 | Camera and pose ingestion | done |
| 3 | Biomechanical normalization engine | done |
| 4 | Dataset ingestion and preprocessing | done |
| 5 | Real-time comparison logic | done |
| 6 | UI and feedback overlay | done |
