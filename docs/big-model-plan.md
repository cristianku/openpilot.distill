# Our own big model on Chestnut

Plan of 31 October 2026. The open decisions are Cristian's (section 7).

## 1. Goal

Replace openpilot's big driving model (MD v3) on **Chestnut**, comma's external GPU, with **our own model of the same size**. It should:

- drive at least as well as MD v3 on our roads (Switzerland, Germany, Italy): a regional model, not a world model;
- also recognise speed limits and the road type;
- later, also do 3D perception.

We do not train from zero. We start from MD v3's own weights and keep MD v3 as the teacher, so the model never forgets what it already knows while it learns our roads and the new tasks.

## 2. Constraints

| What | Value | Source |
|---|---|---|
| Frame budget | 50 ms (20 Hz) | modeld |
| MD v3 on Chestnut | 27 ms a frame in a drive (about 40 % load), 40 ms median on the bench | sunnypilot.speedlimits, `docs/NN_VISION_SPEED_LIMIT_REMAINING_WORK.md`, 27 Oct 2026 |
| Spare time next to MD v3 | 10–13 ms | same |
| Full frame to Chestnut over USB | 115–160 ms | same: new stages must use the frame modeld already has on Chestnut |
| MD v3 size | 395 M parameters, FP16, 0.8 GB, about 50 GMAC a frame | `models/big_driving_supercombo_1365126a_12864.onnx` |
| Driving model input | 128 × 256 YUV, road + wide camera | too small to read distant signs |

Rules that do not move:

- **One owner of Chestnut:** modeld. No second process may open it.
- **No map on the vision side:** no OSM limit, road name or junction as input. Camera and the car's own signals only. OSM may only make training labels offline.
- **Radar only from `carState`** (`stockRadar*`), never from CAN. Perception outputs never go to vehicle control.
- **`modelV2.big`** stays the signal that the big model is running: Speed Limit Assist's auto speed depends on it.
- **Fallback:** without Chestnut, the comma keeps today's small models. The speed limit never depends on Chestnut.
- **Test routes stay out of training:** `annotations/test-routes.txt` and the test partition of each split of sunnypilot.speedlimits. The parity check runs on them.
- **No GPU-heavy job, no weights published, nothing installed on the comma** without Cristian's go.

## 3. Starting point (done)

`supervised/big_model.py` (commit `64e5a7f`) holds MD v3 as a trainable PyTorch model:

- `BigVision`: the vision part, 349 M parameters, rebuilt as timm `convnext_xlarge` (24 input channels, 1 × 1 head 2048 → 512, 32 tokens of 512 per step).
- `OnnxModule`: the policies, 45 M parameters, run op by op from the ONNX graph, weights as parameters.
- `BigSupercombo`: both together, the 9-step 5 Hz sequence of the dataset → outputs + `hidden_state`.

`helpers/check_big_model.py` shows that it matches onnxruntime: every output within 7e-3 of its largest value, in FP32 and FP16.

## 4. Phases

Each phase ends at a gate. The next phase starts only when the gate passes.

### Phase 1: same model, our export, on Chestnut

- Export `BigSupercombo` from PyTorch back to an ONNX with MD v3's interface (queue format, same inputs and outputs, FP16).
- Compile it with tinygrad `DEV=AMD` as modeld does. Run it on Chestnut on recorded frames, with the comma offroad.
- **Gate:** the same outputs as the original MD v3 on the same frames, and no more than 1 ms slower. That proves that our export path can replace the original without changing anything.
- Needs: Cristian's go and the comma.

### Phase 2: driving, fine-tuned on our roads

- **Data:** our drives in `data/` (about 300 one-minute segments today), more from Germany and Italy as they come. The test split is by place, not by drive: the drives keep going over the same roads.
- **Loss:** the two losses `supervised/train.py` already has:
  - *distill:* MD v3's outputs, a frozen copy, on every frame;
  - *gt:* our real driving from the logs and the localizer.

  Distill keeps a high weight, so the model stays close to MD v3 and only corrects where our roads disagree.
- **Against overfitting:** 395 M parameters and a few hours of driving is the main risk.
  - Start with the early convnext stages frozen and a low learning rate.
  - Unfreeze only if the holdout keeps improving.
- **Evaluation:** `evaluation/` on the holdout, MD v3 against our model on the same segments.
- **Gate:** better than MD v3 on the holdout, never worse on safety outputs (lead, plan near the car). Same latency as phase 1.

### Phase 3: speed limits

The networks, data and labels come from sunnypilot.speedlimits, read only:

- checkpoints:
  - `data/runs/2026-10-30_2129-cutter/signnet-best.pt` (finder);
  - `data/runs/2026-10-30_2159-classifier/classifier/reader-best.pt` (reader);
  - `data/runs/2026-10-30_1216-road-type/road-type/gru/road-type-best.pt` (road type);
- model classes in `src/nnslr_tools`;
- the dataset in `annotations/` and `derived/`, as described in its `docs/DATASET.md`.

Proposal, to be measured:

| Part | Where | Input | Rate |
|---|---|---|---|
| Road type (9 classes + junction "passed") | **head on the shared 32 × 512 features** of the big model | what the driving model already sees, plus the car's motion (yaw, steer, blinkers, vEgo) | 1 Hz (its memory was trained on frames 1 s apart) |
| Sign finder + reader | **separate stage on Chestnut**, same process as the driving model | high-resolution crop of the road frame modeld already holds (today's training input: road camera 1344 × 760, the 896-wide top band) | 5 Hz, right after `modelV2` |

- The road-type head learns from the labels and from the current network as a teacher.
- The finder and reader start from the current checkpoints. Later they may share early layers with the big model, if that is faster and just as good.
- **Output:** exactly what `speedvisiond` feeds `SpeedLimitMemory` today. Per frame, the sign boxes, each with class and confidence. Per second, the road-type probabilities and "passed". The speed-limit side then stays unchanged.
- **Gate:**
  - SPEEDLIMITS' parity check on their test routes: same signs, values and road types as today's ONNX, same F1 and limit timeline;
  - the driving model plus all new stages stays under 50 ms on Chestnut, with margin.

### Phase 4: 3D perception

- A head or stage following openpilot.perception's plan (`docs/review-decisions.md` there): temporal, several frames plus the localizer's poses, self-supervised, radar from `carState` for the lead.
- The monocular baseline of openpilot.perception must be beaten in an A/B test first.
- Outputs are displayed or logged, never sent to vehicle control.
- **Gate:** better than the baseline on the metric checks there, and still under 50 ms.

### Phase 5: in the car

- Inside modeld on Chestnut, one process:
  - the driving outputs in `modelV2`, unchanged;
  - the speed-limit observations in their own message for `speedvisiond`, not inside `modelV2`.
- The 50 ms compile check is a hard gate before any install.
- SPEEDLIMITS builds the car branch (`scripts/send_to_car.sh`, `psa-torque-sunny-testing`). Distill sends its modeld changes to them, it does not rebuild that branch.
- Cristian installs on the comma.

## 5. Who does what

| | Distill (this repo) | SPEEDLIMITS (sunnypilot.speedlimits) |
|---|---|---|
| Model | architecture, training, distillation, export | speed-limit networks, data, labels, contracts |
| Car | modeld runtime on Chestnut | `SpeedLimitMemory`, `speedvisiond`, Speed Limit Assist, UI, sunnylink settings, car builds |
| Checks | driving holdout, latency gates | speed-limit parity on their test routes |

Neither edits the other's files: requests go by message. Neither installs anything on the comma.

## 6. Risks

- **Too slow on Chestnut:** every phase has a latency gate. If the stages do not fit, the 5 Hz finder runs on fewer frames, or shares the backbone.
- **Overfitting on a few hours:** frozen early layers, high distill weight, holdout by place.
- **Leaking test routes:** the training reads the split files of sunnypilot.speedlimits and refuses its test routes.
- **Weights:** MD v3's weights come from `commaai/openpilot_driving_models`. Check their licence before publishing anything derived from them.

## 7. Open decisions (Cristian)

1. Speed limits: road type as a head and finder + reader as a separate stage (proposal above), or everything as heads.
2. Go for phase 1 on Chestnut: it needs the comma offroad.
3. How much driving to collect in Germany and Italy before phase 2.
