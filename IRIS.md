# Iris on Qwen3-VL — unofficial reproduction

Implementation of **Iris: Empowering Video MLLMs with High-Frequency Pose Priors via Spatiotemporal Binding**, Zhang et al., [OpenReview sfWwIjHvqo](https://openreview.net/forum?id=sfWwIjHvqo). Based on the supplied 31-page manuscript, especially §§3.2–3.6 and Appendices A/B. This is independent code, not the authors' release. The paper PDF is not redistributed.

This repository is a public fork of QwenLM/Qwen3-VL at `96588727e44c78b25ba03ea03b8e12f7e64fd0da`. Upstream files remain available; the implementation is in `iris/`. No full-data training or pretrained-weight download was performed when preparing this implementation; only synthetic optimizer-step tests were run. Numerical reproduction has **not** been established.

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

## Config-driven progressive training

The runnable recipes are [stage1.json](configs/iris/stage1.json), [stage2.json](configs/iris/stage2.json), and [stage3.json](configs/iris/stage3.json). `paper.json` is a reference summary; the trainer reads the stage recipes. This replaces the earlier `--stage/--manifest/--output` CLI.

All manifest, output, previous-stage, and DeepSpeed paths resolve relative to the recipe file. Model/BERT values are Hugging Face IDs or local paths; prefer absolute paths for local models. Copy a recipe and edit paths for your dataset. Validate the schema and inspect resolved settings without downloading weights:

```bash
python -m iris.train --config configs/iris/stage1.json --validate-only
python -m iris.train --config configs/iris/stage2.json --validate-only
python -m iris.train --config configs/iris/stage3.json --validate-only
```

With prepared manifests/pose arrays and model access, run the stages in order:

```bash
# One GPU; jointly trains BERT, pose encoder and contrastive heads.
CUDA_VISIBLE_DEVICES=0 python -m iris.train --config configs/iris/stage1.json

# Install DeepSpeed separately. Run each command on all 3 nodes with ranks 0,1,2.
# MASTER_ADDR is node 0; all nodes need the same data and checkpoint paths.
torchrun --nnodes=3 --nproc_per_node=8 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  -m iris.train --config configs/iris/stage2.json

torchrun --nnodes=3 --nproc_per_node=8 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  -m iris.train --config configs/iris/stage3.json

# Resume an interrupted stage, including optimizer, scheduler and RNG states.
python -m iris.train --config configs/iris/stage1.json \
  --resume outputs/stage1/checkpoint-100
```

### Paper settings and executable controls

| Setting | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Objective | Symmetric pose/BERT InfoNCE | Answer next-token loss | Answer next-token loss |
| Epochs | 150 | 2 | 1 |
| GPUs / per-device batch / accumulation | 1 / 512 / 1 | 24 / 12 / 1 | 24 / 12 / 1 |
| Effective batch | 512 | 288 | 288 |
| Pose LR | 1e-4 | frozen | 2e-5 |
| BERT LR | 1e-5 | absent | absent |
| Contrastive heads LR | 3e-4 | absent | absent |
| Projector/binding LR | absent | 2e-4 | 3e-5 |
| Person embedding LR | pose LR | frozen | 3e-5 |
| LLM LR | absent | frozen except new input rows at 2e-4 | 1e-5 |
| Vision encoder | absent | frozen | frozen |

AdamW, cosine scheduling, 5% warmup, weight decay .01 and gradient clipping 1 are explicit recipe fields. The recipes also expose all `PoseConfig` fields, BERT ID, projection dimension, initial temperature, maximum caption length and seed. `learning_rates` drives optimizer groups, rather than serving as documentation only. In stage 2 the embedding parameter has zero decay, protecting frozen old vocabulary rows despite AdamW's decoupled decay.

Runtime checks enforce the configured world size and effective batch before loading weights. Stage 1 rejects distributed execution and gradient accumulation because its InfoNCE negatives are computed within a single batch; accumulating small batches would not reproduce 512-way contrastive training. `drop_last=true` keeps full-size batches and avoids singleton contrastive losses. Reducing world size or batch sizes requires an explicitly changed config and is not the published recipe.

### Artifacts and stage transitions

Stage 1 writes `contrastive.pt` containing **BERT, pose encoder, both projection heads and temperature**, plus `pose_encoder.pt` for stage 2 and a saved BERT tokenizer. Stage 2 loads the stage-1 pose weights and freezes them. Stage 3 loads all of stage 2's `iris.pt` and its saved processor, then unfreezes the appropriate modules. The loader verifies the preceding stage number, pose architecture, and (for stage 3) base-model identity and special-token IDs.

Every completed stage writes `resolved_config.json`, `iris_config.json` (including the manifest SHA-256), and Trainer state. Intermediate `checkpoint-*` directories contain model, optimizer, scheduler, RNG/Trainer states and `experiment.json`. Resumption verifies the same recipe (output directory may differ) and manifest hash. Stage transitions intentionally use a fresh optimizer. The manifest hash does not hash referenced videos/arrays or remote model revisions; preserve those assets separately. Pin model revisions by using immutable local snapshots if strict artifact identity is required.

### What has actually been verified

`tests/iris/test_training.py` runs the real Hugging Face Trainer on CPU with tiny locally initialized BERT and Qwen3-VL models. It verifies BERT weight updates, full stage-1 checkpoint contents, an interrupted/resumed stage-1 run matching uninterrupted final weights, exact stage-1→2 and stage-2→3 weight transfer, frozen pose/vision weights in stage 2, and pose/LLM updates with vision frozen in stage 3. Each stage executes two synthetic optimizer steps. Stage 1 reads actual JSONL/NPY files and tokenizes captions with BERT; stage 2/3 substitute a prepared synthetic video dataset and processor so these tests require no downloaded weights or video corpus.

These smoke tests are not full training. The paper-size BERT/8B runs, 24-GPU DeepSpeed execution, real-data convergence, and published benchmark scores remain unverified. Variable-length examples are forwarded separately inside each local stage-2/3 microbatch and their losses averaged; this reference implementation is slower than a packed trainer. The author dataset/proxy QA construction remains external. BERT variant/pooling, pose width/heads, projection dimension, temperature, precision (BF16), ZeRO stage 2, seed, caption length and tail-batch handling are explicit implementation choices where the manuscript does not fully specify them.

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

## 技術的貢献と再現実装の詳細

この節は、論文の技術的貢献を現在の実装へ対応付けたレビュー用の説明です。モデル本体の基準コミットは `caf060a` です。以下では `B` をバッチ数、`P` を人物数、`D` をQwen3-VLの言語モデル隠れ次元、`H,W` を空間マージ後の映像トークングリッドとします。`Iris.forward` の姿勢付き経路は各サンプルを `B=1` で処理します。

### 1. 高頻度姿勢を使う非対称dual-stream構成（§3.3）

**狙い。** 映像のサンプリング密度を増やさずに、疎なRGBフレームでは失われやすい細かな動作を姿勢系列から取り込みます。姿勢の入力フレーム数と、LLMへ渡すトークン数を分離することがポイントです。

**実装。** [Manifest](iris/data.py) は映像を16フレームで読み込み、姿勢配列 `[160,P,24,9]` を受け取ります。[PoseEncoder](iris/modules.py) は24関節を体幹・左腕・右腕・左脚・右脚へ分け、部位ごとの関節特徴を連結して時間方向の `Conv1d` に入力します。

```text
姿勢 [B,160,P,24,9]
  → 部位ごとに [B*P,9*部位内関節数,160]
  → kernel=20, stride=20 の時間畳み込み
  → [B,8,P,5,512]
  → 時間・人物・部位の学習可能埋め込みを加算
  → 時間×人物×部位を並べた8層Transformer
  → 2層MLP（Linear → GELU → Linear）
  → 構造的姿勢トークン xp [B,8,P,5,D]
```

RGB側はQwen3-VLの時間パッチ化で16フレームから8時点となり、姿勢側の8時点と対応します。各部位は20姿勢フレームの情報を1トークンへ圧縮します。空間結合後のトークンも含めると、姿勢関連トークンは `8 × P × 10` 個で、1人なら80個、3人なら240個です。開始・終了の特殊トークン2個はこの数に含みません。

人物のpaddingにはクリップ全体のvisibilityから作るattention maskを使い、存在しない人物をTransformerのkeyから除外し、その出力をゼロにします。一部の時点だけ不可視な人物は、人物全体としては除外しません。

**実装上の選択。** 512次元・8 heads・FFN 4倍幅・pre-norm・dropoutなし、および畳み込みのkernel=strideは論文の未記載事項を補う設定です。`PARTS` では肩・腰を隣接部位と共有します。例えば左腕は `[6,8,10,12]`、左脚は `[14,16,18,20,22]` です。これは著者の部位分割を確認できたという意味ではありません。

**確認済み。** `test_actual_qwen_forward_backward_and_positions` はpose encoderとprojectorまでの勾配伝播を確認し、`test_padding_does_not_change_pose_features` は人物paddingの追加で実人物の特徴が変わらないことを確認します。動作認識精度の改善自体は未検証です。

### 2. Pose-Vision Parallel 3D RoPEによる時間結合（§3.4、Eqs. 2–3）

**狙い。** 入力列の中で映像と姿勢が離れていても、同じ時点を表すトークンに共通の時間座標を与えます。姿勢の残り2軸には画像の縦横位置の代わりに人物・部位を使います。

**実装。** [Iris.forward](iris/model.py) は次の順で `position_ids [3,1,L]` を作ります。

1. 姿勢の開始・終了・placeholderを除いた列に、Qwen3-VL本来の `get_rope_index` を適用します。
2. 各映像時点の最初のパッチから時間座標 `vt[t]` を取得します。同時点の姿勢・姿勢映像トークンへこの値をコピーします。
3. 姿勢トークンには `(vt[t], offset + 100 + p, offset + 100 + n)` を設定します。`offset` は映像トークンのネイティブ位置ID全体の最小値、`p` は人物番号です。
4. `n=0..4` を構造的姿勢、`n=5..9` を空間結合出力に割り当てます。flattenの順序は時間→人物→部位です。
5. 姿勢開始・終了トークンには先行位置の最大値より大きい位置を与え、姿勢ブロック後の質問・回答もその最大値を超える通常の連続位置に置きます。

位置IDだけを変更し、Qwen3-VL内部のRoPE演算は再実装しません。Qwen3-VLのtimestamp付きフレーム列に対し、論文の `t*Δ` を独自のフレーム番号へ置き換えるのではなく、実際の映像位置IDを共有することで統合しています。

**実装上の選択。** `δ=100` は論文の値です。一方、Qwen3-VL固有のglobal offset、連結後の10部位の番号、後続テキストの位置更新は本実装の具体化です。時間IDの一致は時間的注意を強制するmaskではなく、対応を学習するための位置情報です。

**確認済み。** `test_actual_qwen_forward_backward_and_positions` で、2時点それぞれの姿勢時間IDが対応する映像時間IDと一致し、後続テキストが先行位置の最大値を超えることを確認しています。これは小型設定の整合性検証であり、時間結合による性能向上を測るablationではありません。

### 3. 人物領域に限定した空間結合（§3.4、Eq. 4、Appendix B.2）

**狙い。** 複数人物が映る動画で、各人物の姿勢を対応する見た目・周辺の視覚特徴へ結び付けます。

**実装。** [region_mask / SpatialBinding / BindingLayer](iris/modules.py) が担当します。

| 処理 | 入力・出力と動作 |
|---|---|
| 人物bbox生成 | 各20姿勢フレームの窓で、visibleな全関節の正規化2D座標の最小・最大を取り、人物ごとの矩形を作る |
| 視覚mask | マージ後のパッチ中心がbbox内にあるか判定し、`[B,8,P,H*W]` のbool maskを作る |
| 次元縮約 | `xp` をQuery、映像特徴をKey/Valueとし、それぞれ線形層で1024次元へ写像する |
| 相互作用 | 同時点・同人物の5部位をQueryとしてmasked cross-attentionとFFNを2回適用する |
| 出力投影 | weightとbiasをゼロ初期化した線形層で `D` 次元へ戻し、`xpv [B,8,P,5,D]` を得る |
| LLMへの入力 | `cat([xp,xpv], dim=part)` により `[B,8,P,10,D]` とする |

`MultiheadAttention` の `key_padding_mask` はTrueを除外するため、領域内をTrueとするmaskを反転して渡します。注意層・FFNには内部残差接続がありますが、最終的な `xpv` を `xp` に加算する融合は行わず、論文の直接連結方式に対応させます。

bbox内にパッチ中心がない場合は、数値計算のために1つのダミーkeyを通し、出力を明示的にゼロにします。これにより全keyがmaskされたsoftmaxのNaNを防ぎます。ゼロ初期化の出力投影によって学習開始時の `xpv` はゼロですが、トークンの追加そのものがあるため、姿勢付きモデル全体がRGB-onlyモデルと同一出力になるわけではありません。

**実装上の選択。** bboxを20フレームのunionから作ること、パッチの中心点で包含判定すること、空領域の処理は独自に明示した規則です。また、このmaskはcross-attentionが直接読む視覚トークンを制限します。視覚encoder内で既に文脈が混ざる可能性があるため、人物外の情報がモデル全体から完全に排除されることは保証しません。

**確認済み。** `test_empty_binding_and_mask_isolation` は、空領域の出力と勾配が有限であること、および領域外の入力視覚特徴を変更しても同じ領域の結合出力が変わらないことを確認します。これは特徴入力レベルの検証です。

### 4. 異種姿勢表現とデータ整形（§3.2、Eq. 1、Appendix A）

**狙い。** 画像内での位置、身体の関節構造、カメラ座標における人物全体の移動を別々の特徴として持たせます。

**実装。** [descriptor](iris/pose.py) は、左右hip（Table 7の14・15番）の平均をpelvisとし、次の9次元を構築します。

```text
[u / width, v / height,
 (xyz - pelvis) / local_scale,          # 3成分
 (pelvis + translation) / global_scale, # 3成分
 visibility]
```

visibilityは、投影点が画像内にあり、その人物のsegmentation mask内にあるかで定めます。`xyz + translation` がcamera-space座標になることを入力契約とし、身体相対位置と全体移動を分けます。これはカメラ運動を明示的に推定・除去する処理ではありません。

[clean](iris/pose.py) は人物ごとに、隣接する有効フレームのない孤立フレームを除去し、両端が有効な最大3フレームの欠損を線形補間します。補間点のvisibilityには両端の積を使います。次に、連続フレームの可視関節でlocal 3D変位の最大値を評価し、指定thresholdを超える人物を除外します。残った座標の0〜7成分を `0.7*current + 0.3*previous` のEMAで平滑化し、全期間不可視の人物を除去します。最後に `sample` が160フレームへ均等サンプリングし、CLIでは先頭3人物までを保存します。

**再現範囲。** 論文のデータ構築全体のうち、上記のdescriptorと決定的な後処理を実装しています。GroundingDINOの検出、SAM3の追跡・drift検証、SAM-3D-Bodyの推定、Qwenによる学習用metadata解析・品質判定・caption整形、proxy QAの生成は未実装の外部工程です。人物の選択順は入力側の順序を引き継ぎ、検出confidenceによるtop-3の選別はここでは行いません。

**実装上の選択。** 正規化スケールとjitter thresholdは論文から確定できないため、入力必須です。推論データから値を調整せず、学習側で決定した値を固定して使います。jitter判定の定義とEMA前に判定する順序も、本実装の選択として扱います。

**確認済み。** `test_descriptor_and_short_gap` でpelvisの分離と短い欠損の補間を検証しています。推定器の精度、追跡品質、長い欠損・激しい遮蔽での性能はこのテストの対象ではありません。

### 5. 新規姿勢モダリティを導入する3段階学習（§3.5、Appendix B.3）

この学習方式は上記の貢献を機能させるための手順です。[ContrastivePose](iris/contrastive.py)、[Iris.configure_stage / optimizer_groups](iris/model.py)、[学習CLI](iris/train.py) に分けて実装しています。

| 段階 | 実装した目的関数と更新対象 |
|---|---|
| Stage 1 | pose encoder出力を有効人物・時点・部位で平均し、BERTのCLSとともに256次元へ投影・L2正規化。バッチ内のpose→text / text→poseのcross-entropyを平均する対称InfoNCE。pose encoder、BERT、両projection head、temperatureを更新 |
| Stage 2 | 回答部分のみのnext-token loss。projector・spatial binding・新規特殊トークンのinput embeddingを更新し、pose encoder・既存LLM・visionを凍結 |
| Stage 3 | 同じnext-token lossでvision以外を更新。pose-free動画QAも受け付け、RGB経路を維持 |

Stage 2ではembedding全体を学習可能にした上で、gradient hookにより既存語彙行の勾配をゼロにします。さらに、そのparameter groupのweight decayをゼロにし、勾配がゼロでもAdamWで既存行が縮む問題を避けます。姿勢placeholder自体のembeddingはforward時に姿勢特徴へ置き換わるため、特殊トークン3種類すべてに必ず非ゼロ勾配が流れるという意味ではありません。

Stage 1では人物paddingを平均の分母から除外します。stage間は `pose_encoder.pt` または `iris.pt` で重みを受け渡し、optimizerは作り直します。stage内の再開には `--resume` を使います。学習率・epoch・batch sizeは上の「Progressive training」と [paper.json](configs/iris/paper.json) にまとめています。

**外部工程・制約。** Person Grounding / Body Part DiscriminationのQAはmanifestとして準備する必要があり、このコードは自動生成しません。Stage 1のnegativeはローカルbatch内のみです。複数GPUにまたがるnegative収集や、24 GPUでの実動作は未検証です。

**確認済み。** `test_contrastive_backward` で対照学習の勾配を確認し、`test_stage2_adamw_keeps_old_embeddings_exactly_frozen` でAdamW更新後も既存語彙行が完全一致すること、Stage 2/3の凍結範囲を確認します。実データでの訓練は実行していません。

### 6. Qwen3-VLへの統合とレビュー対象

[Manifest](iris/data.py) は動画の後・質問の前へ姿勢ブロックを挿入し、[Iris.forward](iris/model.py) がplaceholderを `xp/xpv` に置き換えます。映像は既存の `get_video_features` で処理し、そのDeepStack特徴と `visual_pos_masks` をネイティブ言語モデルへ渡します。LLM本体のattentionやvision encoderを独自モデルへ差し替える構成ではありません。

姿勢なし・全不可視・人物数ゼロの入力はデータ層でRGB-onlyへ切り替えます。`test_rgb_fallback_identical` は同一backboneへ直接入力した場合と出力が一致することを確認します。`test_multi_person_and_checkpoint` は3人物とcheckpointの保存・復元を、`test_invalid_slot_count` は姿勢トークン数の不一致検出を確認します。`test_inference_manifest_never_includes_answer` は推論入力へ正解文を連結しないことを確認します。

モデルの単体テストと上記の学習統合テストは実装の整合性の根拠です。論文の各貢献による精度向上を確かめるには、同じデータ・学習条件でRGB-only、時間結合なし、空間結合なし、姿勢密度変更などのablationが別途必要です。その後のconfig対応では小型モデルの合成データで各Stageのoptimizer更新・保存・引き継ぎとStage 1の再開を検証しています。本訓練・ベンチマーク評価は行っていません。
