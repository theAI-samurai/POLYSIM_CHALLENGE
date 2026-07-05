# Iteration 1: Current Multimodal Speaker Identification Implementation

## 1) Scope and Objective

Iteration 1 is the baseline multimodal speaker identification system currently implemented in this repository.

Goal:
- Predict speaker identity from synchronized face frames and voice waveform.
- Learn a shared embedding space that is discriminative for identity classification and robust under missing-face conditions.

Task type:
- Closed-set speaker classification (softmax over known training speakers).
- Multimodal representation learning (face + audio).

---

## 2) High-Level Algorithm

The training pipeline performs the following steps for each mini-batch:

1. Sample one audio clip and K face frames for each sample.
2. Encode face and audio with pretrained transformer/CNN backbones.
3. Project both modalities into the same embedding dimension.
4. Fuse modality embeddings by concatenation + MLP head.
5. Compute three losses on fused features/logits:
   - Cross-Entropy classification loss
   - Supervised contrastive loss
   - Inter-class orthogonality regularization
6. Backpropagate with optional mixed precision + gradient accumulation.
7. Update with AdamW, cosine annealing LR schedule, and optional optimizer-state CPU offload.

Total loss in Iteration 1:

L_total = L_ce + lambda_con * L_supcon + lambda_orth * L_orth

---

## 3) Data Pipeline and Sampling

### 3.1 Dataset structure assumptions

Data root is expected to contain:
- faces/<speaker_id>/<video_id>/*.jpg
- voices/<speaker_id>/<video_id>/*.wav

Each sample is formed as:
- Face tensor: [K, 3, H, W]
- Waveform: [T]
- Label: integer speaker index

### 3.2 Audio preprocessing

- Load waveform with torchaudio.
- Resample to target sample rate if needed (default 16 kHz).
- Convert to mono by channel averaging.
- Duration handling:
  - If shorter than max duration: zero-pad.
  - If longer than max duration: random temporal crop.

This creates fixed-length audio input of:
- T = MAX_AUDIO_SEC * 16000

### 3.3 Face preprocessing

Training transform:
- Resize to 224x224
- Random horizontal flip
- Color jitter
- Normalize using ImageNet mean/std

Validation transform:
- Resize + normalize only (deterministic)

Frame sampling strategy:
- K frames are sampled with replacement from the available frame pool for the same video.

### 3.4 Class balancing

A WeightedRandomSampler is used.
- Per-sample weight = 1 / (#samples of that sample's speaker)
- This approximates speaker-balanced sampling under class imbalance.

---

## 4) Model Architecture (Iteration 1)

## 4.1 Face encoder

Selectable backbones:
- vit_base: google/vit-base-patch16-224, feature dim 768
- vit_large: google/vit-large-patch16-224, feature dim 1024
- resnet50: microsoft/resnet-50, pooled feature dim 2048

Feature extraction:
- ViT: use CLS token from last_hidden_state[:, 0]
- ResNet: use pooler output and flatten

Projection head:
- Linear(feat_dim -> embed_dim)
- LayerNorm(embed_dim)

Output shape:
- Per frame: [B, embed_dim]

Multi-frame aggregation:
- Input face frames are [B, K, 3, H, W]
- Flatten to [B*K, 3, H, W], encode each frame
- Reshape to [B, K, D], then average over K
- Final face embedding: [B, D]

## 4.2 Audio encoder

Selectable backbones:
- wavlm_base: microsoft/wavlm-base, hidden size 768
- wavlm_large: microsoft/wavlm-large, hidden size 1024
- wav2vec2: facebook/wav2vec2-base, hidden size 768
- unispeech_sv: microsoft/unispeech-sat-base-plus-sv, hidden size 768

Feature extraction:
- Use backbone last_hidden_state [B, L, D]
- Mean-pool over time dimension L -> [B, D]

Projection head:
- Linear(feat_dim -> embed_dim)
- LayerNorm(embed_dim)

Final audio embedding:
- [B, D]

## 4.3 Fusion block

Fusion type:
- Feature-level late fusion with embedding concatenation.

Operation:
- Concatenate [f_emb || a_emb] -> [B, 2D]
- Pass through fusion MLP:
  - Linear(2D -> D)
  - GELU
  - LayerNorm

Output:
- Fused embedding z in R^(B x D)

## 4.4 Classification head

- Linear(D -> num_speakers)
- Produces logits for cross-entropy training.

## 4.5 Missing-modality robustness

A learnable face mask token is used.

During training:
- For each sample, with probability mask_prob, replace face embedding with mask token.
- This is modality dropout on the face branch.

At inference:
- Caller can pass mask_faces=all_true for audio-only operation.

This directly supports protocols where face is unavailable or unreliable.

---

## 5) Loss Functions and Optimization Objective

## 5.1 Cross-Entropy loss

L_ce = CE(logits, labels)

Purpose:
- Directly optimizes closed-set speaker classification accuracy.

## 5.2 Supervised Contrastive loss (SupCon)

Implementation details:
- Normalize fused embeddings.
- Compute pairwise similarity matrix divided by temperature.
- For each anchor, positives are same-label samples excluding itself.
- Denominator includes all non-self samples.
- Anchors with zero positives are ignored from mean reduction.

Purpose:
- Tighten intra-speaker clusters.
- Increase inter-speaker separation in the shared embedding space.

## 5.3 Orthogonality loss

Implementation details:
- Normalize fused embeddings.
- Compute cosine similarity matrix.
- For pairs with different labels, penalize squared cosine similarity.

L_orth approx mean_{i,j: y_i != y_j}(cos(z_i, z_j)^2)

Purpose:
- Encourage near-orthogonal directions for different speakers.
- Reduce class overlap in embedding geometry.

## 5.4 Full multi-objective training

L_total = L_ce + lambda_con * L_supcon + lambda_orth * L_orth

Default weights in current config parser:
- lambda_con = 0.5
- lambda_orth = 0.1

---

## 6) Training Strategy (Iteration 1)

## 6.1 Optimizer and scheduler

- Optimizer: AdamW
- Weight decay: configurable
- LR scheduler: CosineAnnealingLR

Warmup-style freezing:
- Backbones can be frozen for first N epochs (FREEZE_ENCODERS_EPOCHS).
- Only train fusion/projection/classifier during freeze phase.
- After unfreezing, optimizer is recreated with lower LR (0.1 * base LR).

## 6.2 Stabilization and memory controls

- Gradient clipping: global norm max 1.0
- Optional AMP: torch.amp.GradScaler("cuda")
- Optional gradient checkpointing on HF backbones
- Optional gradient accumulation via GRAD_ACCUM_STEPS
- Optional optimizer-state CPU offload (move Adam moment buffers CPU<->GPU)

## 6.3 Validation split

- Optional random train/val split via VAL_SPLIT.
- Validation transform is deterministic.
- Best checkpoint selection uses validation accuracy when validation exists.
- Otherwise best is based on training accuracy.

## 6.4 Logging and checkpoints

- Batch and epoch metrics logged to TensorBoard when available.
- Saved checkpoints:
  - best.pt
  - last.pt
  - epoch_xxxx.pt every CHECKPOINT_INTERVAL epochs

Checkpoint payload includes:
- model state
- optimizer state
- scheduler state
- epoch
- best_acc
- parsed config snapshot

---

## 7) Tensor Shapes Through the Network

Given batch size B, K face frames, waveform length T, embedding dim D:

1. Face input: [B, K, 3, 224, 224]
2. Face frame encoding: [B*K, D]
3. K-frame mean pooling: [B, D]
4. Audio input: [B, T]
5. Audio sequence states: [B, L, D_backbone]
6. Audio pooled + projection: [B, D]
7. Concatenation: [B, 2D]
8. Fusion MLP output: [B, D]
9. Classifier logits: [B, num_speakers]

Losses are computed from:
- logits for L_ce
- fused embedding for L_supcon and L_orth

---

## 8) Why This Architecture Is Reasonable

Iteration 1 combines three complementary principles:

- Strong pretrained unimodal encoders:
  - Vision transformer/CNN for identity cues in faces.
  - Self-supervised speech transformers for speaker cues in voice.

- Shared-space metric shaping:
  - SupCon + orthogonality improve embedding geometry beyond plain CE.

- Robustness to partial modality failure:
  - Face masking trains the model to retain discriminative audio-only capability.

This makes the model usable for fully multimodal, partially-missing, and audio-only settings with one training recipe.

---

## 9) Current Iteration Label

This document defines the current codebase baseline as:

- Iteration 1

Future files can follow the same format for Iteration 2, Iteration 3, etc., to track architectural and algorithmic changes over time.

---

## 10) Iteration 2 Update: Fusion Switching + Optional LASE + VAD + w2v-BERT 2.0

This section documents the newly implemented changes in the current codebase.

### 10.1 Voice Activity Detection (VAD)

Audio preprocessing now includes optional VAD to trim silence and remove non-speech segments before fixed-length crop/pad.

Implementation details:
- Energy-based detection using STFT frame energy.
- Frame energies are converted to dB and thresholded relative to the clip maximum.
- Speech frames are smoothed with a small median filter.
- Only detected speech samples are kept.
- Fallback behavior: if VAD detects no speech, original waveform is retained.

Configuration:
- USE_VAD: true | false
- VAD_THRESHOLD_DB: threshold in dB below max energy (default -40.0)

Practical guidance:
- Lower threshold (for example -50) trims more aggressively.
- Higher threshold (for example -30) is safer if speech is being cut.

### 10.2 Audio encoder update: w2v-BERT 2.0

Audio backbone options now include:
- wavlm_base
- wavlm_large
- wav2vec2
- w2v_bert_2
- unispeech_sv

New backbone:
- w2v_bert_2 maps to facebook/w2v-bert-2.0 via Wav2Vec2BertModel.
- It is high-capacity and generally strongest for cross-lingual robustness.

### 10.3 New fusion model options

Training can now switch fusion architecture via environment variable FUSION_TYPE:
- late: existing concatenation-based late fusion (FusionModel)
- cross_attn: bidirectional cross-attention fusion (CrossAttentionFusionModel)
- gated: learnable gated blending fusion (GatedFusionModel)

Cross-attention fusion details:
- Face tokens: K frame embeddings [B, K, D]
- Audio tokens: temporal speech hidden states projected to [B, L, D]
- Two cross-attention directions:
  - face query over audio keys/values
  - audio query over face keys/values
- Mean pooling on both attended token streams, then concatenation and fusion MLP

Gated fusion details:
- Face embedding f and audio embedding a in R^D
- Gate g = sigmoid(MLP([f || a])) in R^D
- Fused vector z = g * f + (1 - g) * a
- Post-fusion normalization/projection before speaker classifier

### 10.4 LASE integration

Optional Language-Adversarial Speaker Encoding is integrated for cross_attn and gated models:
- Gradient Reversal Layer (GRL)
- Auxiliary language classifier head on fused embedding
- Extra loss term:

L_total = L_ce + lambda_con * L_supcon + lambda_orth * L_orth + lambda_lang * L_lang

where L_lang is cross-entropy on predicted language logits.

Implementation behavior:
- ENABLE_LASE=false:
  - model returns (fused, speaker_logits)
  - language loss path is disabled
- ENABLE_LASE=true:
  - model returns (fused, speaker_logits, language_logits)
  - training/evaluation expects language labels in each batch
  - FUSION_TYPE must be cross_attn or gated

Language label source in current implementation:
- Labels are inferred from wav path text by matching keywords in LANG_KEYWORDS.
- If no keyword matches, sample is assigned language index 0 and counted as unknown.

### 10.5 New .env variables

Added configuration keys:
- FUSION_TYPE: late | cross_attn | gated
- CROSS_ATTN_HEADS: integer heads for cross-attention model
- ENABLE_LASE: true | false
- LANG_KEYWORDS: comma-separated language markers in path text
- NUM_LANGUAGES: language class count (read directly from .env)
- LAMBDA_LANG: weight of adversarial language loss
- LANG_GRL_LAMBDA: GRL coefficient for adversarial signal strength
- USE_VAD: true | false
- VAD_THRESHOLD_DB: VAD energy threshold in dB

### 10.6 When to use LASE

Use LASE when:
- Training data contains multiple languages (for example English and Urdu).
- Target evaluation includes cross-language generalization and language-invariant speaker identity.
- You can provide reliable language labels (explicitly or via trustworthy path conventions).

Do not use LASE when:
- Training is monolingual (for example English-only training).
- Language labels are unreliable or mostly inferred as unknown.
- You only need same-language evaluation and maximum in-language performance.

Practical recommendation for your current setup (train English only, test English+Urdu):
- Set ENABLE_LASE=false
- Use strong audio backbone (for example w2v_bert_2 or wavlm_large)
- Try FUSION_TYPE=cross_attn or gated for better multimodal interaction without adversarial branch
