# FormPro2

Real-time barbell back squat form analysis from a single camera. It extracts 3D pose
landmarks, normalizes them to the lifter's body proportions, compares them against a
labelled reference corpus and flags form faults live.

Detected faults: knee valgus, hips rising too fast (good morning), high squat, heel lift.

## Setup

Python 3.9+ (tested on 3.13).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/fetch_model.py --variant heavy
```

## Running

Both front ends require a reference corpus in `data/reference/`.

```powershell
python server.py        # web UI at http://127.0.0.1:8000
python app.py           # OpenCV window; q to quit, r to reset
```

Options for both: `--source PATH` to replay a video, `--reference DIR`, `--no-mirror`.

To check camera placement before recording:

```powershell
python scripts/smoke_test_pose.py
```

## Camera placement

45-degree anterior oblique (front-diagonal), roughly waist height. A lateral view cannot
see knee valgus and gets occluded by plates; a frontal view depends on depth for back
angle. Thresholds assume this angle.

## Reference corpus

Reference files are JSON with pre-computed angles. Validate a directory with
`python scripts/validate_dataset.py`.

```json
{
  "metadata": {
    "exercise": "barbell_back_squat",
    "camera_angle": "45_oblique_anterior",
    "dataset_type": "reference_optimal",
    "fps_target": 30
  },
  "subject_proportions": { "femur_to_torso_ratio": 1.12, "tibia_to_femur_ratio": 0.85 },
  "frames": [
    {
      "frame_id": 142,
      "timestamp_ms": 4733,
      "phase": "concentric",
      "angles": {
        "camera_near": { "hip_flexion": 110.5, "knee_flexion": 125.0,
                         "ankle_dorsiflexion": 85.2, "back_to_vertical": 45.1 },
        "camera_far":  { "hip_flexion": 111.0, "knee_flexion": 124.5,
                         "ankle_dorsiflexion": 86.0, "back_to_vertical": 45.3 },
        "global": { "knee_to_hip_width_ratio": 0.95 }
      },
      "form_label": "optimal_form"
    }
  ]
}
```

- Angles are included angles in degrees: standing reads hip 180, knee 180, ankle 90,
  back 0.
- `phase`: `setup`, `eccentric`, `bottom`, `concentric`, `recovery`.
- `form_label` is per frame: `optimal_form`, `error_high_squat`, `error_knee_valgus`,
  `error_good_morning`, `error_heel_lift`.
- Reference data should be recorded with this pipeline at the same camera angle, so live
  and reference angles share the same measurement bias.

With no recordings yet, generate a simulated starter corpus (a clean squat on a rigid
skeleton, widened across builds):

```powershell
python scripts/seed_reference.py
```

To widen a thin corpus, synthesize other builds from one optimal recording:

```powershell
python scripts/dataset_generator.py data/reference/subject.json
```

Generated files are marked `reference_optimal_synthetic` and should be replaced with real
recordings when available.

## How it works

| Module | Role |
|---|---|
| `capture.py` | Threaded capture with a drop-old buffer |
| `pose_estimator.py` | MediaPipe BlazePose, 3D world landmarks |
| `kinematics.py` | Joint angles, body proportions, camera-near/far side |
| `phases.py` | Rep segmentation from hip height and velocity |
| `dataset_loader.py` | Reference corpus loading and validation |
| `form_analyzer.py` | Corpus-derived bands, fault detection, DTW |
| `synthesis.py` | Build-warping model for synthetic references |
| `server.py`, `app.py` | Web and desktop front ends |

Thresholds are not hardcoded. Acceptable bands are percentiles of `optimal_form` frames,
grouped by phase and interpolated by `femur_to_torso_ratio`, so a longer-femur lifter is
allowed more forward lean. All tunables live in `configs/squat.yaml`.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest
```
