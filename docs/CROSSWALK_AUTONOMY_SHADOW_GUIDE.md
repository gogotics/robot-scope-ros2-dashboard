# Crosswalk Autonomy Shadow Evaluation

Robot Scope now includes the crosswalk mask geometry and center-tracking policy
from `gogotics/go2-crosswalk-autonomy`.  The first integration stage is fixed to
shadow mode: it records recommended velocities but emits zero robot commands.

## Why shadow mode comes first

The source model recognizes a crosswalk only.  It does not establish traffic
signal state, pedestrian clearance, obstacle clearance, camera-to-ground metric
calibration, or a completed exit.  Robot Scope's Route Planner therefore keeps
those requirements separate and does not turn this result directly into motion.

The imported policy adds four fail-closed properties needed before Jetson work:

- one nearby crosswalk instance is selected instead of merging unrelated masks;
- both visible boundary clearances must contain the camera center plus a margin;
- a stale or missing frame immediately produces a zero recommendation;
- mask loss never means crossing complete; odometry or an end-boundary detector
  must explicitly provide `exit_confirmed`.

## Codespaces setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-crosswalk.txt
```

Put the segmentation weights at `weights/best.pt`.  Model weights and recorded
videos must remain outside Git because they can be large and may have separate
licensing terms.

Run a recorded-video evaluation:

```bash
PYTHONPATH=. python3 scripts/replay_crosswalk_autonomy.py input.webm \
  --model weights/best.pt \
  --output artifacts/crosswalk-shadow.avi \
  --class-name crosswalk \
  --confidence 0.50 \
  --device cpu \
  --crosswalk-width-m 3.0
```

Use `--device 0` in a CUDA runner.  The command creates:

- an overlay video with the selected mask and centerline;
- CSV frame decisions with proposed and emitted velocities;
- JSON Lines snapshots accepted by the Route Planner v1 perception contract;
- a JSON summary with detection ratio, hold ratio, lateral error, and inference
  latency.

Every emitted velocity remains zero.  The nonzero `recommended_*` columns are
for controller tuning only.

## Acceptance before Jetson deployment

Use Go2 camera recordings containing straight, offset, angled, partly occluded,
and false-positive scenes.  Record separate daytime and low-light sets.  Agree
numeric gates from those recordings before changing any live runtime.  At a
minimum, review detection coverage, p95 inference latency, lateral-error tails,
boundary holds, and every false-positive motion recommendation.

The `crosswalk_width_m` conversion assumes the near mask span is the known
physical crosswalk width.  This approximation is suitable for reviewing the
Route Planner data shape, not live control.  Jetson deployment requires camera
intrinsics, camera-to-`base_link` extrinsics, ground-plane projection, and a
measured body/foot clearance margin.

The live phase must also provide fresh traffic-green, pedestrian-clear, obstacle
clear, control-lease, watchdog, emergency-stop, and explicit exit evidence.  It
should feed the existing `Go2ControlBridge`; no second Unitree command publisher
should be introduced.
