# FormPro2 — Real-Time Barbell Back Squat Form Analysis

Live-camera biomechanical form coaching. The pipeline ingests a webcam stream, extracts
3D pose landmarks, normalizes them against the lifter's own body proportions, compares the
result to a labelled reference dataset, and renders corrective cues onto the live feed.

**Exercise scope for v1: barbell back squat only.** Thresholds, error classes and phase
segmentation are hardcoded for the squat by design; generalization is deferred.

---

## Architecture

The system is a one-way pipeline. Each stage owns exactly one transformation and depends
only on the stage above it, so any stage can be swapped or unit-tested in isolation.

```
 camera ──▶ capture.py ──▶ pose_estimator.py ──▶ kinematics.py ──▶ form_analyzer.py ──▶ overlay
           (raw BGR       (33 3D landmarks,     (body-proportion    (compare vs.        (draw)
            frames,        metric space)         normalized          reference set)
            drop-old)                            angles/ratios)
                                       ▲
                        dataset_loader.py ───────┘
                        (offline reference sequences, same normalization)
```

The critical invariant: **`kinematics.py` is the single normalization path, used by both the
live stream and the offline dataset loader.** If live frames and reference frames were
normalized by different code, every comparison downstream would be measuring the difference
between two implementations rather than between two squats.

### Directory layout

```
FormPro2/
├── app.py                       # Phase 6 — main loop + UI
├── requirements.txt
├── configs/
│   └── squat.yaml               # all tunables; no magic numbers in code
├── models/                      # .task model binaries (git-ignored, see fetch_model.py)
├── data/
│   ├── reference/               # Phase 4 — labelled reference sequences (CSV/JSON)
│   └── recordings/              # git-ignored raw captures
├── formpro/
│   ├── schema.py                # ✅ Phase 2 — data contracts + landmark indices
│   ├── config.py                # ✅ Phase 1 — typed config loading
│   ├── filters.py               # ✅ Phase 2 — One-Euro landmark smoothing
│   ├── capture.py               # ✅ Phase 2 — threaded camera, drop-old buffering
│   ├── pose_estimator.py        # ✅ Phase 2 — BlazePose backend behind a Protocol
│   ├── video_processor.py       # ✅ Phase 2 — orchestrates capture → pose
│   ├── kinematics.py            # ⬜ Phase 3 — segment lengths, ratios, joint angles
│   ├── dataset_loader.py        # ⬜ Phase 4 — reference set ingestion
│   ├── phases.py                # ⬜ Phase 4 — eccentric/concentric segmentation
│   └── form_analyzer.py         # ⬜ Phase 5 — DTW / similarity comparison engine
├── scripts/
│   ├── fetch_model.py           # ✅ downloads the BlazePose .task binary
│   └── smoke_test_pose.py       # ✅ Phase 2 diagnostic viewer (not the real UI)
└── tests/
```

---

## Setup

Requires CPython 3.9+ (verified on 3.13 with MediaPipe 1.0).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/fetch_model.py --variant heavy
```

MediaPipe 1.0 removed the legacy `mp.solutions.pose` API; FormPro2 targets the current
Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`), which requires the model
weights as a local `.task` file — hence the `fetch_model.py` step.

Verify the camera + pose stage end to end:

```powershell
python scripts/smoke_test_pose.py
```

You should see a wireframe tracking you, plus a live readout of capture FPS, inference
latency and dropped-frame count. Press `q` to quit.

---

## Coordinate conventions

Two landmark spaces come out of the pose stage and they are **not** interchangeable:

| Space | Field | Units | Origin | Axes | Use for |
|---|---|---|---|---|---|
| Image | `PoseFrame.image_xyz` | normalized `[0,1]` | top-left of frame | X right, **Y down** | drawing only |
| World | `PoseFrame.world_xyz` | metres | midpoint of the hips | X right, **Y up**, Z toward camera | all biomechanics |

MediaPipe emits world landmarks Y-down / Z-away. FormPro2 negates both at ingestion so that
world space is a conventional right-handed Y-up system. Every downstream module — angles,
depth checks, phase segmentation — assumes Y-up. `image_xyz` is left in raw MediaPipe
orientation because OpenCV draws in Y-down.

**The capture stage never mirrors the frame.** Mirroring before inference would swap the
lifter's anatomical left and right, which would invert per-side findings like knee valgus.
Display mirroring is applied at render time only (Phase 6).

---

## Build status

| Phase | Module | Status |
|---|---|---|
| 1 | Project setup & architecture | ✅ done |
| 2 | Camera & pose ingestion | ✅ done |
| 3 | Biomechanical normalization engine | ⬜ pending |
| 4 | Dataset ingestion & preprocessing | ⬜ pending |
| 5 | Real-time comparison logic | ⬜ pending |
| 6 | UI & feedback overlay | ⬜ pending |
