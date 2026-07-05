# POLY-SIM: Multimodal Speaker Identification

A PyTorch implementation of multimodal speaker identification combining face and audio modalities with optional language-adversarial training and multiple fusion strategies.

## Features

- **Multimodal Learning**: Face (ViT/ResNet) + Audio (WavLM/wav2vec2/w2v-BERT 2.0/UniSpeech)
- **Multiple Fusion Strategies**: Late fusion, cross-attention, gated blending
- **Voice Activity Detection**: Optional energy-based VAD to remove silence
- **LASE Support**: Language-Adversarial Speaker Encoding for cross-lingual robustness
- **Missing-Modality Robustness**: Learnable face masking for audio-only evaluation
- **Advanced Training**: Mixed precision, gradient checkpointing, gradient accumulation, optimizer-state CPU offload
- **TensorBoard Monitoring**: Real-time training metrics and validation curves

---

## Quick Start

### Prerequisites

```bash
pip install torch torchaudio transformers librosa scikit-learn tensorboard
```

### 1. Prepare Data (CSV mode)

Training now supports CSV input directly (recommended).

Required CSV columns:
- audio_path
- face_path
- identity

Example CSV row:
```csv
audio_path,face_path,identity
data_train/train/v1/voices/id0001/English/U3rWfLEkFvg/00000.wav,data_train/train/v1/faces/id0001/English/U3rWfLEkFvg/0010125.jpg,id0001
```

Notes:
- Paths can be relative to repository root or absolute.
- identity is used as speaker ID/class label.
- One row corresponds to one training sample.

Optional legacy mode is still available (folder scan):
```
<DATA_ROOT>/faces/<speaker_id>/<video_id>/*.jpg
<DATA_ROOT>/voices/<speaker_id>/<video_id>/*.wav
```

### 2. Configure Training (.env)

The `.env` file contains all training hyperparameters. Key settings:

```dotenv
# Data
DATASET_SOURCE=csv            # csv | folder
TRAIN_CSV=comp/v1_train_English.csv
# Multiple CSVs are supported (comma-separated), e.g.:
# TRAIN_CSV=comp/v1_train_English.csv,comp/v1_train_Urdu.csv
DATA_ROOT=data_train          # used only when DATASET_SOURCE=folder
K_FACES=10                    # frames per sample
MAX_AUDIO_SEC=30.0            # audio duration

# Audio preprocessing
USE_VAD=true                  # enable Voice Activity Detection
VAD_THRESHOLD_DB=-40.0        # VAD threshold (lower = more aggressive)

# Model architecture
FACE_ENCODER=vit_large        # vit_base | vit_large | resnet50
AUDIO_ENCODER=wavlm_large     # wavlm_base | wavlm_large | wav2vec2 | w2v_bert_2 | unispeech_sv
EMBED_DIM=384

# Fusion
FUSION_TYPE=late              # late | cross_attn | gated
CROSS_ATTN_HEADS=8

# Training
EPOCHS=100
BATCH_SIZE=32
LR=1e-4
FREEZE_ENCODERS_EPOCHS=5

# Loss weights
LAMBDA_CON=0.5                # supervised contrastive
LAMBDA_ORTH=0.1               # orthogonality
LAMBDA_LANG=0.2               # language adversarial (LASE only)

# Memory optimization
USE_AMP=true                  # mixed precision
GRAD_CHECKPOINT=true          # gradient checkpointing
GRAD_ACCUM_STEPS=1
CPU_OFFLOAD_OPTIM=true

# Checkpoints & logging
OUTPUT_DIR=checkpoints/exp4
CHECKPOINT_INTERVAL=10
VAL_SPLIT=0.2
TENSORBOARD_DIR=runs/exp4
```

### 3. Train

**CSV-based training (recommended):**

```bash
cd /path/to/code_exp4_audio_face
python scripts/train.py
```

This command reads DATASET_SOURCE and TRAIN_CSV from .env.

To run legacy folder-based training, set:
```dotenv
DATASET_SOURCE=folder
DATA_ROOT=data_train
```

This reads all config from `.env` and starts training. You'll see:
- Dataset statistics (number of samples, speakers)
- Model info (encoders, fusion type, trainable params)
- Per-epoch metrics (loss, accuracy)
- Validation results every epoch

**Monitor with TensorBoard:**

```bash
tensorboard --logdir runs/exp4
```

Open `http://localhost:6006` in your browser.

---

## Configuration Guide

### Audio Encoders: Which to Choose?

| Backbone | Size | Speed | Same-Lang | Cross-Lang | Notes |
|----------|------|-------|-----------|-----------|-------|
| **wavlm_base** | 95M | Fast | ⭐⭐ | ⭐⭐ | Good balanced baseline |
| **wavlm_large** | 317M | Slower | ⭐⭐⭐ | ⭐⭐⭐ | Recommended for most cases |
| **wav2vec2** | 95M | Fast | ⭐⭐⭐ | ⭐ | Strong English, weak cross-lingual |
| **w2v_bert_2** | 600M | Slowest | ⭐⭐⭐ | ⭐⭐⭐⭐ | **Best cross-lingual** |
| **unispeech_sv** | ~100M | Fast | ⭐⭐⭐ | ⭐⭐ | Speaker verification optimized |

**Recommendations:**

- **English only (P3/P4)**: `unispeech_sv` (fast, speaker-focused)
- **Balanced (English + limited cross-lingual)**: `wavlm_base` or `wavlm_large`
- **Best cross-lingual (P5/P6)**: `w2v_bert_2` (if compute allows)

### Fusion Strategies

| Strategy | Speed | Multi-Interaction | Use Case |
|----------|-------|-------------------|----------|
| **late** | ⭐⭐⭐ | ⭐ | Baseline, quick experiments |
| **cross_attn** | ⭐⭐ | ⭐⭐⭐ | Rich multi-modal interaction, requires more compute |
| **gated** | ⭐⭐⭐ | ⭐⭐ | Learnable modality weighting, balanced |

### LASE (Language-Adversarial Speaker Encoding)

Use when training on multiple languages (e.g., English + Urdu) to learn language-invariant speaker representations.

**Enable LASE:**

```dotenv
ENABLE_LASE=true
FUSION_TYPE=cross_attn         # or gated (required, not compatible with late)
TRAIN_CSV=comp/v1_train_English.csv,comp/v1_train_Urdu.csv
LANG_KEYWORDS=english,urdu     # keywords in wav file paths
NUM_LANGUAGES=2
LAMBDA_LANG=0.2                # weight of language loss
LANG_GRL_LAMBDA=1.0            # gradient reversal strength
```

**When NOT to use LASE:**

- Training on single language (English-only) → set `ENABLE_LASE=false`
- Language labels unreliable → VAD removes too much speech
- Only need same-language evaluation → LASE adds unnecessary complexity

---

## Training Modes & Inference

### Training (Full Multimodal)

Default behavior. Both face and audio are used. Face is randomly masked with probability `MASK_PROB` (default 0.3) to train audio-only capability.

```python
# In code: automatic during training
outputs = model(face_frames, waveforms)
embeddings, logits = outputs
```

### Validation/Evaluation (Same-Language: P3/P4)

Uses full multimodal (face + audio).

```python
model.eval()
with torch.no_grad():
    embeddings, logits = model(face_frames, waveforms)
    predictions = logits.argmax(1)
```

### Inference (Audio-Only: P4/P6)

Mask all faces to simulate missing modality:

```python
model.eval()
with torch.no_grad():
    mask_faces = torch.ones(batch_size, dtype=torch.bool, device=device)
    embeddings, logits = model(face_frames, waveforms, mask_faces=mask_faces)
    predictions = logits.argmax(1)
```

---

## Checkpoint Management

Checkpoints are saved to `OUTPUT_DIR` (default: `checkpoints/exp4/`):

- **best.pt**: Best validation accuracy checkpoint
- **last.pt**: Most recent checkpoint
- **epoch_0010.pt, epoch_0020.pt, ...**: Periodic snapshots every `CHECKPOINT_INTERVAL` epochs

### Resume Training

```dotenv
RESUME_FROM=checkpoints/exp4/epoch_0050.pt
```

Or leave empty to train from scratch.

### Use Checkpoint for Evaluation

```bash
# Set in .env
EVAL_CHECKPOINT=checkpoints/exp4/best.pt

# Challenge-style evaluation (same-lang + cross-lang)
python scripts/eval_all.py

# Local CSV scoring (adds/updates pred column)
python scripts/eval_english.py
```

For CSV scoring, set in .env:
```dotenv
EVAL_ENGLISH_CSV=comp/v1_train_English.csv
EVAL_ENGLISH_OUTPUT_CSV=checkpoints/exp4/v1_train_English_scored.csv
EVAL_ENGLISH_CHECKPOINT=checkpoints/exp4/best.pt
```

---

## Hyperparameter Tuning

### For Better Training Accuracy (P3/P4)

1. **Increase model capacity**:
   - Use `AUDIO_ENCODER=wavlm_large` or `w2v_bert_2`
   - Use `FACE_ENCODER=vit_large`

2. **Adjust loss weights**:
   - Increase `LAMBDA_CON` (0.5 → 1.0) for tighter clusters
   - Decrease `MASK_PROB` (0.3 → 0.1) for stronger face usage

3. **Training duration**:
   - Increase `EPOCHS` (100 → 200)
   - Decrease `FREEZE_ENCODERS_EPOCHS` (5 → 2) to unfreeze sooner

### For Better Cross-Lingual (P5/P6)

1. **Use cross-lingual audio**:
   - Set `AUDIO_ENCODER=w2v_bert_2` (or `wavlm_large`)

2. **If multi-language data available**:
   - Set `ENABLE_LASE=true`
   - Use `FUSION_TYPE=cross_attn`
   - Provide reliable language labels

3. **Modality robustness**:
   - Increase `MASK_PROB` (0.3 → 0.5) for audio-only robustness

### Memory Issues?

If OOM errors occur:

1. Reduce `BATCH_SIZE` (32 → 16)
2. Enable `GRAD_CHECKPOINT=true`
3. Enable `CPU_OFFLOAD_OPTIM=true`
4. Use smaller encoder: `AUDIO_ENCODER=wavlm_base`

---

## Output & Logging

### TensorBoard Metrics

Training metrics logged to `TENSORBOARD_DIR`:

- **Loss/train**: Total training loss
- **CE/train**: Cross-entropy loss
- **ConLoss/train**: Supervised contrastive loss
- **OrthLoss/train**: Orthogonality loss
- **LangLoss/train**: Language adversarial loss (LASE only)
- **Accuracy/train**: Training accuracy
- **Validation metrics**: Loss, accuracy (if `VAL_SPLIT > 0`)

### Startup Banner

At training start, the script prints:

```
════════════════════════════════════════════════════════════════
 POLY-SIM  |  Multimodal Speaker Identification
════════════════════════════════════════════════════════════════
  Resume from    : scratch
  Ckpt interval  : every 10 epochs
  Face encoder   : vit_large
  Audio encoder  : wavlm_large
  Fusion type    : late
  Embed dim      : 384
  LASE           : off
  Epochs         : 100  |  Batch size: 32
  Trainable parameters : 145,234,567
  ...
════════════════════════════════════════════════════════════════
```

---

## Troubleshooting

### VAD Removes Too Much Speech

Increase `VAD_THRESHOLD_DB` (less aggressive):
```dotenv
VAD_THRESHOLD_DB=-30.0  # was -40.0
```

Or disable entirely:
```dotenv
USE_VAD=false
```

### Out of Memory (OOM)

1. Reduce batch size: `BATCH_SIZE=16`
2. Enable gradient checkpointing: `GRAD_CHECKPOINT=true`
3. Use smaller audio encoder: `AUDIO_ENCODER=wav2vec2`

### Validation accuracy not improving

1. Check `VAL_SPLIT > 0` is set
2. Increase training time: `EPOCHS=200`
3. Check data is properly balanced

### Language labels not inferred (LASE mode)

When using LASE, check that:
1. `LANG_KEYWORDS=english,urdu` matches your path structure
2. Wav file paths contain keywords (case-insensitive): `.../english/.../sample.wav`
3. Check console output for "Language labels" line showing inferred counts

---

## Citation & References

Built with:
- **PyTorch**: Deep learning framework
- **HuggingFace Transformers**: Pretrained encoders
- **librosa**: Audio processing and VAD

For more details on the model architecture, see `iteration.md`.

---

## License

[Your license here]
