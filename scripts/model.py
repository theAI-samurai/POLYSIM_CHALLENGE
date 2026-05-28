"""
model.py — Encoders and FusionModel for multimodal speaker identification.
All backbones loaded from HuggingFace Hub via the `transformers` library.

Face encoder   (--face-encoder):
  vit_base     google/vit-base-patch16-224   [CLS] token  feat_dim=768
               → general-purpose ViT, good face features
  vit_large    google/vit-large-patch16-224  [CLS] token  feat_dim=1024
               → higher capacity, better for large datasets
  resnet50     microsoft/resnet-50           pooled        feat_dim=2048
               → lightweight, fast baseline

Audio encoder  (--audio-encoder):
  wavlm_base   microsoft/wavlm-base          mean-pool    feat_dim=768   95M params
               → balanced accuracy, good starting point
  wavlm_large  microsoft/wavlm-large         mean-pool    feat_dim=1024  317M params
               → best cross-lingual (P5/P6), recommended
  wav2vec2     facebook/wav2vec2-base         mean-pool    feat_dim=768   95M params
               → strong English baseline, lower cross-lingual than WavLM
  w2v_bert_2   facebook/w2v-bert-2.0         mean-pool    feat_dim=768   600M params
               → SOTA cross-lingual (wav2vec 2.0 + BERT), best for P5/P6
  unispeech_sv microsoft/unispeech-sat-base-plus-sv  mean-pool  feat_dim=768
               → pretrained for speaker verification, good for P3/P4

Recommended combinations
─────────────────────────
  P3/P4 (English):    vit_base  + unispeech_sv      (fast, speaker-focused)
  P3–P6 (all):        vit_base  + wavlm_base        (good balance)
  Best overall:       vit_large + w2v_bert_2        (highest P5/P6 expected)
  Alternative best:   vit_large + wavlm_large       (lighter, still excellent)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import (
    AutoFeatureExtractor,
    ResNetModel,
    ViTModel,
    WavLMModel,
    Wav2Vec2Model,
    UniSpeechSatModel,
    Wav2Vec2BertModel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Face Encoder
# ─────────────────────────────────────────────────────────────────────────────

class FaceEncoder(nn.Module):
    """
    Encodes a face image to a fixed-size embedding using a HuggingFace ViT or ResNet.

    Args:
        backbone:   'vit_base', 'vit_large', or 'resnet50'
        embed_dim:  Output embedding dimension.
        pretrained: Load pretrained weights from HuggingFace Hub.
    """

    _HF_IDS = {
        "vit_base":  "google/vit-base-patch16-224",
        "vit_large": "google/vit-large-patch16-224",
        "resnet50":  "microsoft/resnet-50",
    }

    def __init__(self,backbone: str = "vit_base",embed_dim: int = 256,pretrained: bool = True,) -> None:
        super().__init__()
        self.backbone_name = backbone

        if backbone not in self._HF_IDS:
            raise ValueError(f"Unknown face backbone: {backbone!r}. Choose from {list(self._HF_IDS)}.")

        model_id = self._HF_IDS[backbone]

        if backbone == "resnet50":
            if pretrained:
                self.backbone = ResNetModel.from_pretrained(model_id)
            else:
                cfg = ResNetModel.config_class.from_pretrained(model_id)
                self.backbone = ResNetModel(cfg)
            feat_dim = 2048
        else:
            # vit_base or vit_large
            if pretrained:
                self.backbone = ViTModel.from_pretrained(model_id)
            else:
                cfg = ViTModel.config_class.from_pretrained(model_id)
                self.backbone = ViTModel(cfg)
            feat_dim = self.backbone.config.hidden_size   # 768 or 1024

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]  pre-processed pixel values
        Returns:
            [B, embed_dim]
        """
        if self.backbone_name == "resnet50":
            # ResNetModel returns pooler_output [B, feat_dim, 1, 1]
            out = self.backbone(x).pooler_output          # [B, 2048, 1, 1]
            emb = out.flatten(1)                          # [B, 2048]
        else:
            # ViTModel returns last_hidden_state; index [CLS] token
            emb = self.backbone(x).last_hidden_state[:, 0]  # [B, D]

        return self.proj(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Encoder
# ─────────────────────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """
    Encodes a raw mono waveform (16 kHz) to a fixed-size embedding
    using a HuggingFace speech model.

    Args:
        backbone:   'wavlm_base', 'wavlm_large', 'wav2vec2', or 'unispeech_sv'
        embed_dim:  Output embedding dimension.
        pretrained: Load pretrained weights from HuggingFace Hub.
    """

    _HF_IDS = {
        "wavlm_base":    "microsoft/wavlm-base",
        "wavlm_large":   "microsoft/wavlm-large",
        "wav2vec2":      "facebook/wav2vec2-base",
        "w2v_bert_2":    "facebook/w2v-bert-2.0",
        "unispeech_sv":  "microsoft/unispeech-sat-base-plus-sv",
    }

    def __init__(self, backbone: str = "wavlm_base", embed_dim: int = 256, pretrained: bool = True,) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.feature_extractor = None

        if backbone not in self._HF_IDS:
            raise ValueError(
                f"Unknown audio backbone: {backbone!r}. "
                f"Choose from {list(self._HF_IDS)}."
            )

        model_id = self._HF_IDS[backbone]

        if backbone.startswith("wavlm"):
            if pretrained:
                self.backbone = WavLMModel.from_pretrained(model_id)
            else:
                cfg = WavLMModel.config_class.from_pretrained(model_id)
                self.backbone = WavLMModel(cfg)
        elif backbone == "wav2vec2":
            if pretrained:
                self.backbone = Wav2Vec2Model.from_pretrained(model_id)
            else:
                cfg = Wav2Vec2Model.config_class.from_pretrained(model_id)
                self.backbone = Wav2Vec2Model(cfg)
        elif backbone == "w2v_bert_2":
            if pretrained:
                self.backbone = Wav2Vec2BertModel.from_pretrained(model_id)
            else:
                cfg = Wav2Vec2BertModel.config_class.from_pretrained(model_id)
                self.backbone = Wav2Vec2BertModel(cfg)
            # Wav2Vec2BertModel expects log-mel style input_features, not raw waveform.
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        else:  # unispeech_sv
            if pretrained:
                self.backbone = UniSpeechSatModel.from_pretrained(model_id)
            else:
                cfg = UniSpeechSatModel.config_class.from_pretrained(model_id)
                self.backbone = UniSpeechSatModel(cfg)

        feat_dim = self.backbone.config.hidden_size   # 768 (base) or 1024 (large)

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: [B, T]  — mono, 16 kHz, already padded/cropped
        Returns:
            [B, embed_dim]
        """
        out = self.encode_sequence(waveforms)
        emb = out.mean(dim=1)                              # [B, D]
        return self.proj(emb)

    def encode_sequence(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return frame-level backbone states [B, L, D_backbone]."""
        if self.backbone_name == "w2v_bert_2":
            if self.feature_extractor is None:
                raise RuntimeError("Feature extractor not initialized for w2v_bert_2")
            device = waveforms.device
            wave_np = waveforms.detach().cpu().numpy()
            feats = self.feature_extractor(
                [w for w in wave_np],
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
            )
            input_features = feats["input_features"].to(device)
            attention_mask = feats.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            return self.backbone(
                input_features=input_features,
                attention_mask=attention_mask,
            ).last_hidden_state

        # WavLM/Wav2Vec2/UniSpeech consume raw waveform [B, T].
        return self.backbone(waveforms).last_hidden_state


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Model
# ─────────────────────────────────────────────────────────────────────────────

class FusionModel(nn.Module):
    """
    Combines a FaceEncoder and AudioEncoder with:
      - K-frame average pooling for robust face representations
      - Learnable [MASK] token to simulate missing face at test time
      - MLP fusion head projecting to a shared embedding space
      - Softmax classification head over num_speakers

    Modality dropout:
      During training, the face embedding is replaced by the learned MASK
      token with probability `mask_prob`. This teaches the audio branch to
      identify speakers independently — directly targeting P4/P6 protocols.

    Args:
        face_encoder:  FaceEncoder instance
        audio_encoder: AudioEncoder instance
        num_speakers:  Number of output classes
        embed_dim:     Shared embedding dimension (must match both encoders)
        mask_prob:     Probability of zeroing the face branch during training
    """

    def __init__(self,face_encoder: FaceEncoder,audio_encoder: AudioEncoder,num_speakers: int,embed_dim: int = 256,mask_prob: float = 0.3,) -> None:
        super().__init__()
        self.face_encoder = face_encoder
        self.audio_encoder = audio_encoder
        self.mask_prob = mask_prob
        self.embed_dim = embed_dim

        # Learnable token that replaces the face embedding when it is masked
        self.face_mask_token = nn.Parameter(torch.zeros(embed_dim))

        # Fusion: [face_emb ‖ audio_emb] → shared space
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Speaker classification head
        self.classifier = nn.Linear(embed_dim, num_speakers)

    # ------------------------------------------------------------------
    def enable_gradient_checkpointing(self) -> None:
        """Recompute activations during backward pass instead of caching them.

        Reduces GPU activation memory ~50% at ~25% extra compute cost.
        Delegates to HuggingFace's built-in gradient_checkpointing_enable().
        """
        self.face_encoder.backbone.gradient_checkpointing_enable()
        self.audio_encoder.backbone.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    def _encode_faces(self, face_frames: torch.Tensor) -> torch.Tensor:
        """Average K face frames into one embedding per sample.

        Args:
            face_frames: [B, K, 3, H, W]
        Returns:
            [B, embed_dim]
        """
        B, K, C, H, W = face_frames.shape
        flat = face_frames.view(B * K, C, H, W)
        embs = self.face_encoder(flat)        # [B*K, D]
        return embs.view(B, K, -1).mean(dim=1)  # [B, D]

    # ------------------------------------------------------------------
    def forward(
        self,
        face_frames: torch.Tensor,
        waveforms: torch.Tensor,
        mask_faces: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            face_frames: [B, K, 3, H, W]
            waveforms:   [B, T]
            mask_faces:  bool Tensor [B] — True means use MASK token for that sample.
                         If None and model.training, sampled automatically at mask_prob.
                         Pass all-True at inference for audio-only evaluation (P4/P6).

        Returns:
            embeddings: [B, embed_dim]  — fused representation (use for metric losses)
            logits:     [B, num_speakers]
        """
        B = waveforms.shape[0]
        device = waveforms.device

        # Audio branch
        a_emb = self.audio_encoder(waveforms)   # [B, D]

        # Face branch
        f_emb = self._encode_faces(face_frames)  # [B, D]

        # Modality dropout
        if mask_faces is None and self.training:
            mask_faces = torch.rand(B, device=device) < self.mask_prob

        if mask_faces is not None:
            mask_token = self.face_mask_token.unsqueeze(0).expand(B, -1)  # [B, D]
            f_emb = torch.where(mask_faces.unsqueeze(1), mask_token, f_emb)

        # Fuse and classify
        fused = self.fusion(torch.cat([f_emb, a_emb], dim=1))  # [B, D]
        logits = self.classifier(fused)                         # [B, num_speakers]

        return fused, logits


class _GradientReversalFn(Function):
    """Identity in forward pass, gradient sign flip in backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_adv * grad_output, None


class GradientReversal(nn.Module):
    """Gradient reversal layer used for adversarial domain/language invariance."""

    def __init__(self, lambda_adv: float = 1.0) -> None:
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(self, x: torch.Tensor, lambda_adv: float | None = None) -> torch.Tensor:
        coef = self.lambda_adv if lambda_adv is None else float(lambda_adv)
        return _GradientReversalFn.apply(x, coef)


class LanguageAdversarialHead(nn.Module):
    """Auxiliary language classifier for LASE-style adversarial learning."""

    def __init__(self, embed_dim: int, num_languages: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if num_languages < 2:
            raise ValueError("num_languages must be >= 2 for language-adversarial training.")

        self.grl = GradientReversal(lambda_adv=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_languages),
        )

    def forward(self, x: torch.Tensor, lambda_adv: float = 1.0) -> torch.Tensor:
        return self.classifier(self.grl(x, lambda_adv=lambda_adv))


class CrossAttentionFusionModel(nn.Module):
    """Cross-attention fusion model with optional language-adversarial branch.

    Returns:
      - (fused, logits) when num_languages is None
      - (fused, logits, lang_logits) when num_languages is set
    """

    def __init__(
        self,
        face_encoder: FaceEncoder,
        audio_encoder: AudioEncoder,
        num_speakers: int,
        embed_dim: int = 256,
        mask_prob: float = 0.3,
        num_heads: int = 8,
        num_languages: int | None = None,
    ) -> None:
        super().__init__()
        self.face_encoder = face_encoder
        self.audio_encoder = audio_encoder
        self.embed_dim = embed_dim
        self.mask_prob = mask_prob

        self.face_mask_token = nn.Parameter(torch.zeros(embed_dim))

        self.audio_token_proj = nn.Sequential(
            nn.Linear(self.audio_encoder.backbone.config.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.face_queries_audio = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.audio_queries_face = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_speakers)

        self.lang_head = (
            LanguageAdversarialHead(embed_dim=embed_dim, num_languages=num_languages)
            if num_languages is not None
            else None
        )

    def enable_gradient_checkpointing(self) -> None:
        self.face_encoder.backbone.gradient_checkpointing_enable()
        self.audio_encoder.backbone.gradient_checkpointing_enable()

    def _encode_face_tokens(self, face_frames: torch.Tensor) -> torch.Tensor:
        bsz, k, c, h, w = face_frames.shape
        flat = face_frames.view(bsz * k, c, h, w)
        tokens = self.face_encoder(flat)
        return tokens.view(bsz, k, -1)

    def _encode_audio_tokens(self, waveforms: torch.Tensor) -> torch.Tensor:
        audio_states = self.audio_encoder.encode_sequence(waveforms)
        return self.audio_token_proj(audio_states)

    def forward(
        self,
        face_frames: torch.Tensor,
        waveforms: torch.Tensor,
        mask_faces: torch.Tensor | None = None,
        lambda_adv: float = 1.0,
    ):
        bsz = waveforms.shape[0]
        device = waveforms.device

        face_tokens = self._encode_face_tokens(face_frames)
        audio_tokens = self._encode_audio_tokens(waveforms)

        if mask_faces is None and self.training:
            mask_faces = torch.rand(bsz, device=device) < self.mask_prob
        if mask_faces is not None:
            mask_token = self.face_mask_token.view(1, 1, -1).expand(bsz, face_tokens.size(1), -1)
            face_tokens = torch.where(mask_faces.view(bsz, 1, 1), mask_token, face_tokens)

        face_ctx, _ = self.face_queries_audio(
            query=face_tokens,
            key=audio_tokens,
            value=audio_tokens,
            need_weights=False,
        )
        audio_ctx, _ = self.audio_queries_face(
            query=audio_tokens,
            key=face_tokens,
            value=face_tokens,
            need_weights=False,
        )

        face_vec = face_ctx.mean(dim=1)
        audio_vec = audio_ctx.mean(dim=1)

        fused = self.fusion(torch.cat([face_vec, audio_vec], dim=1))
        logits = self.classifier(fused)

        if self.lang_head is None:
            return fused, logits

        lang_logits = self.lang_head(fused, lambda_adv=lambda_adv)
        return fused, logits, lang_logits


class GatedFusionModel(nn.Module):
    """Gated feature fusion model with optional language-adversarial branch.

    Returns:
      - (fused, logits) when num_languages is None
      - (fused, logits, lang_logits) when num_languages is set
    """

    def __init__(
        self,
        face_encoder: FaceEncoder,
        audio_encoder: AudioEncoder,
        num_speakers: int,
        embed_dim: int = 256,
        mask_prob: float = 0.3,
        num_languages: int | None = None,
    ) -> None:
        super().__init__()
        self.face_encoder = face_encoder
        self.audio_encoder = audio_encoder
        self.embed_dim = embed_dim
        self.mask_prob = mask_prob

        self.face_mask_token = nn.Parameter(torch.zeros(embed_dim))

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

        self.post_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        self.classifier = nn.Linear(embed_dim, num_speakers)

        self.lang_head = (
            LanguageAdversarialHead(embed_dim=embed_dim, num_languages=num_languages)
            if num_languages is not None
            else None
        )

    def enable_gradient_checkpointing(self) -> None:
        self.face_encoder.backbone.gradient_checkpointing_enable()
        self.audio_encoder.backbone.gradient_checkpointing_enable()

    def _encode_faces(self, face_frames: torch.Tensor) -> torch.Tensor:
        bsz, k, c, h, w = face_frames.shape
        flat = face_frames.view(bsz * k, c, h, w)
        embs = self.face_encoder(flat)
        return embs.view(bsz, k, -1).mean(dim=1)

    def forward(
        self,
        face_frames: torch.Tensor,
        waveforms: torch.Tensor,
        mask_faces: torch.Tensor | None = None,
        lambda_adv: float = 1.0,
    ):
        bsz = waveforms.shape[0]
        device = waveforms.device

        f_emb = self._encode_faces(face_frames)
        a_emb = self.audio_encoder(waveforms)

        if mask_faces is None and self.training:
            mask_faces = torch.rand(bsz, device=device) < self.mask_prob
        if mask_faces is not None:
            mask_token = self.face_mask_token.unsqueeze(0).expand(bsz, -1)
            f_emb = torch.where(mask_faces.unsqueeze(1), mask_token, f_emb)

        gate = self.gate(torch.cat([f_emb, a_emb], dim=1))
        fused = gate * f_emb + (1.0 - gate) * a_emb
        fused = self.post_fusion(fused)
        logits = self.classifier(fused)

        if self.lang_head is None:
            return fused, logits

        lang_logits = self.lang_head(fused, lambda_adv=lambda_adv)
        return fused, logits, lang_logits
