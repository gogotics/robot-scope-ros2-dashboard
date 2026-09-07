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

If importing OpenCV reports `libGL.so.1` in Codespaces, install the missing
runtime library and retry:

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

The default is Ultralytics YOLOE-26s segmentation with the text prompts
`crosswalk` and `zebra crossing`, so custom weights are not required for an
initial test.  Ultralytics downloads the model and text encoder on the first
run; later runs reuse the cache.

For a smoke test when no Go2 recording is available, download this CC0
Wikimedia Commons image:

```bash
curl -L \
  'https://thumb.wikimedia.org/wikipedia/commons/thumb/a/af/Crossroad-zebra-crossing-crosswalk_%2823698506663%29.jpg/1280px-Crossroad-zebra-crossing-crosswalk_%2823698506663%29.jpg' \
  -o /tmp/crosswalk-demo.jpg
```

Run the image smoke test:

```bash
PYTHONPATH=. python3 scripts/replay_crosswalk_autonomy.py \
  /tmp/crosswalk-demo.jpg \
  --output artifacts/crosswalk-demo-shadow.avi \
  --device cpu
```

Then run a recorded-video evaluation with your Go2 footage:

```bash
PYTHONPATH=. python3 scripts/replay_crosswalk_autonomy.py input.webm \
  --output artifacts/crosswalk-shadow.avi \
  --device cpu
```

Use `--device 0` in a CUDA runner. To use a locally trained model instead, put
the segmentation weights outside Git and add, for example,
`--model weights/best.pt --class-name crosswalk --confidence 0.50`. Repeating
`--class-name` adds accepted class aliases or YOLOE text prompts. The command
creates:

- an overlay video with the selected mask and centerline;
- CSV frame decisions with proposed and emitted velocities;
- JSON Lines snapshots accepted by the Route Planner v1 perception contract;
- a JSON summary with detection ratio, hold ratio, lateral error, and inference
  latency.

Every emitted velocity remains zero.  The nonzero `recommended_*` columns are
for controller tuning only.

YOLOE is a convenient zero-shot baseline, not a validated production detector.
Its default confidence is intentionally low for offline coverage review. Tune
the threshold on Go2 recordings, then train or fine-tune a crosswalk model if
false positives or misses do not meet the acceptance gates.

Ultralytics code and downloaded weights have their own AGPL-3.0 or enterprise
licensing terms. Review them before a closed-source or commercial Jetson
deployment; installing an optional dependency does not relicense Robot Scope.

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
