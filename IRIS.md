# Iris on Qwen3-VL — unofficial reproduction

Implementation of **Iris: Empowering Video MLLMs with High-Frequency Pose Priors via Spatiotemporal Binding**, Zhang et al., [OpenReview sfWwIjHvqo](https://openreview.net/forum?id=sfWwIjHvqo). Based on the supplied 31-page manuscript, especially §§3.2–3.6 and Appendices A/B. This is independent code, not the authors' release. The paper PDF is not redistributed.

This repository is a public fork of QwenLM/Qwen3-VL at `96588727e44c78b25ba03ea03b8e12f7e64fd0da`. Upstream files remain available; the implementation is in `iris/`. No training or pretrained-weight download was performed when preparing this implementation. Numerical reproduction has **not** been established.

## Implemented correspondence

| Paper | Code |
|---|---|
| Eq. 1, Table 7: 24 joints, normalized 2D + pelvis-relative 3D + camera pelvis + visibility | `iris/pose.py:descriptor` |
| §3.3: 160 dense frames, 5 semantic parts, stride-20 temporal convolution, 8-layer encoder, person/time/part embeddings, 2-layer MLP | `iris/modules.py:PoseEncoder`, `iris/model.py:Iris` |
| Eq. 4 / B.2: person-bbox masked cross-attention, 2 layers, width 1024, zero output projection | `iris/modules.py:SpatialBinding` |
| Eqs. 2/3: shared visual/pose temporal position, person/part axes offset 100, 10 tokens/person/time | `iris/model.py:Iris.forward` |
| Qwen3-VL visual features and DeepStack injection | Native Transformers vision encoder and language model, preserving DeepStack features |
| Stage 1: BERT + pose symmetric InfoNCE | `iris/contrastive.py` |
| Stages 2/3: parameter freezing, special-embedding gradient mask, differentiated AdamW groups | `iris/model.py`, `iris/train.py` |
| A.1: isolated-frame removal, bounded interpolation, EMA, configurable jitter rejection | `iris/pose.py:clean` |
| No usable pose | Native Qwen3-VL forward, without pose tokens |

## Setup and verification

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]'
source .venv/bin/activate
python -m pytest -q
python -m iris.train --help
```

Tested with Python 3.12, PyTorch 2.9.1 and Transformers 4.57.6. Tests instantiate tiny real Qwen3-VL/BERT models locally: forward/backward, DeepStack integration, temporal positions, multiple persons, checkpoint roundtrip, padding invariance, region masking, empty-region stability, RGB fallback, inference answer exclusion, and exact frozen-vocabulary preservation after an AdamW step. These are synthetic unit/integration checks, not dataset training or benchmark runs.

## Data contract

Prepare one video per record, without additional images. Relative paths resolve against the JSONL manifest directory. Stage 1 records:

```json
{"pose":"poses/clip.npy","caption":"The person raises their left arm."}
```

Stages 2/3 records:

```json
{"id":"clip-1","video":"videos/clip.mp4","pose":"poses/clip.npy","question":"What does the person do?","answer":"Raises their left arm."}
```

Stage 3 additionally accepts records without `pose`, preserving native RGB capability. Stage 2 requires valid pose-paired records. Include person-grounding and rendered body-part-discrimination QAs in the stage-2 manifest (B.1). Person grounding should supply only the target person's pose; body-part questions require an actual box overlay in the input video, not an ungrounded label. No proxy answers or appearance labels are synthesized by this package.

Pose arrays are float32 `[160,P,24,9]`, `1 <= P <= 3`, in Table-7 order (see `iris/modules.py`). Features are `[u,v,x,y,z,pelvis_x,pelvis_y,pelvis_z,visibility]`; 2D coordinates are normalized to image width/height, visibility is binary. Person identities must persist over the entire clip. Missing frames are zero-filled. `[160,0,24,9]` or wholly invisible arrays trigger RGB-only fallback. Arrays must cover the same complete clip as the video, with uniform endpoint-inclusive sampling.

To convert estimator exports:

```bash
python -m iris.pose estimator-export.npz poses/clip.npy \
  --local-scale YOUR_TRAINING_LOCAL_SCALE \
  --global-scale YOUR_TRAINING_GLOBAL_SCALE \
  --jitter-threshold YOUR_FIXED_THRESHOLD
```

The NPZ contains `uv [T,P,24,2]` in pixels, `xyz [T,P,24,3]`, `translation [T,P,3]`, and identity-specific boolean `masks [T,P,H,W]`. `xyz + translation` must equal camera-space coordinates. Export joints in Table-7 order; the SAM-3D-Body native joint indexing must not be passed unchanged. Fit normalization on training data only, record the scales, and reuse them for evaluation.

**External prerequisites:** GroundingDINO/SAM3/SAM-3D-Body detection, tracking and raw keypoint extraction are not bundled or automatically downloaded. Supply estimator exports or ready pose arrays. This repository implements the deterministic descriptor/postprocessing and MLLM side, not an end-to-end raw-video estimator stack. The paper's private ~550K mixture, ~30K motion captions, proxy labels, and author checkpoints are not included. Their exact reconstruction cannot be guaranteed from the manuscript. At test time do not use MLLM-based pose correction, manual filtering or benchmark answers.

## Progressive training (commands only; not executed)

```bash
python -m iris.train --stage 1 --manifest data/stage1.jsonl --output outputs/stage1 --bf16

# Install DeepSpeed separately for distributed training.
# On each of 3 nodes, use its NODE_RANK; set MASTER_ADDR to node 0.
torchrun --nnodes=3 --nproc_per_node=8 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  -m iris.train --stage 2 --manifest data/stage2.jsonl \
  --previous outputs/stage1 --output outputs/stage2 --bf16 \
  --deepspeed configs/iris/zero2.json

torchrun --nnodes=3 --nproc_per_node=8 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  -m iris.train --stage 3 --manifest data/stage3.jsonl \
  --previous outputs/stage2 --output outputs/stage3 --bf16 \
  --deepspeed configs/iris/zero2.json
```

Defaults follow Table 8: epochs 150/2/1; batch 512 in stage 1, per-device 12 in stages 2/3 (global 288 over 24 GPUs); cosine with 5% warmup; weight decay .01; gradient clipping 1. Stage 1 LRs: pose 1e-4, BERT 1e-5, contrastive heads 3e-4. Stage 2 LR 2e-4; pose/vision/LLM frozen except new input embeddings. Stage 3 LRs: pose 2e-5, person embeddings/binding/projector 3e-5, backbone 1e-5; vision remains frozen. Old vocabulary rows have zero gradient and zero weight decay in stage 2. Each new stage creates a fresh optimizer; `--resume` resumes within a stage.

Variable-length examples are forwarded separately inside each local microbatch, then their losses are averaged. This is slower than a packed implementation. Multi-node/DeepSpeed execution and full-size 8B memory use have not been tested. Stage 1 is intended for the paper's single-GPU setup; distributed global-negative gathering is not implemented. Training writes a full `iris.pt`, a transferable `pose_encoder.pt`, processor files, and model/config provenance. Full checkpoints are large.

## Inference and scoring

```bash
python -m iris.infer --checkpoint outputs/stage3 --manifest data/test.jsonl \
  --output predictions.jsonl --device cuda
python -m iris.evaluate normalized_predictions.jsonl --task choice
python -m iris.evaluate normalized_counts.jsonl --task count
```

Inference manifests use the same format; answers are optional and never fed to generation. Decoding is greedy and recomputes the prefix (no KV-cache optimization). Scoring inputs contain `prediction` and `target`, already normalized to official choice labels or numerical counts. Counting returns normalized MAE and off-by-one accuracy; zero-count targets are rejected rather than silently assigning an arbitrary denominator. Benchmark-specific parsing/splits and any official evaluator must be matched separately; this utility does not claim official benchmark equivalence.

## Explicit assumptions and remaining reproducibility gaps

The manuscript does not specify the pose-transformer width/heads, convolution kernel/padding, precise part joint membership, BERT variant/pooling, contrastive projection size/temperature, 3D normalization constants, jitter/drift thresholds, or complete dataset annotations. Our choices: width 512, 8 heads, FFN 4x, pre-norm, no dropout; nonoverlapping kernel=stride=20; overlapping anatomical anchor joints; bert-base-uncased CLS and 256D contrastive projection, initial temperature .07. Values requiring data statistics are mandatory CLI arguments.

Bounding regions use the union of visible joints in each 20-frame pose window and patch-center inclusion. Empty or sub-patch regions produce zero interaction tokens, avoiding NaNs or cross-person leakage. Visibility interpolation uses intersection of boundary visibility. Jitter uses maximum consecutive local-joint displacement before EMA; EMA is `0.7*current+0.3*previous`. These are disclosed choices where the exact algorithm is absent.

Qwen3-VL uses timestamped frame blocks. Rather than invent a separate temporal scale, pose time positions are copied from actual native visual positions; spatial person/part axes use the video's global origin plus delta. Part IDs cover 0..9 after concatenating structural and interacted tokens. Text following the pose block starts beyond its maximum coordinate. These details require comparison with author code if released.

Only one video per example is supported. Pose-free examples retain the backbone path. Full-data training, end-to-end estimator integration, official benchmark execution and reported-score reproduction remain unverified. No result table in the paper should be read as a result of this repository.

Manuscript SHA-256: `ccf53d2d00f7dd76ea1fac39c4dded3ee350f76d0958163147402fa92f9e1b9a`.
