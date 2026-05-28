"""eval_english.py — local English training-set evaluation.

This script scores the English training CSV using both face and voice paths
from the CSV itself, writes a `pred` column, and saves the result back to CSV.

It is intended for local test evaluation, not challenge submission packaging.

Environment variables (read from project-root .env):
  EVAL_ENGLISH_CSV           input CSV to score (default: data_train/comp/v1_train_English.csv)
  EVAL_ENGLISH_OUTPUT_CSV    output CSV path; empty means overwrite input in place
  EVAL_ENGLISH_CHECKPOINT    checkpoint to load (default: checkpoints/exp1/best.pt)
    EVAL_ENGLISH_BATCH_SIZE    eval batch size (default: 64)
    EVAL_ENGLISH_K_FACES       eval face frames per sample (default: 1)
    EVAL_ENGLISH_MAX_AUDIO_SEC eval audio length in sec (default: MAX_AUDIO_SEC)
    EVAL_ENGLISH_USE_AMP       use mixed-precision inference on CUDA (default: true)
  DEVICE                     optional override; otherwise cuda if available
  K_FACES                    optional override; default 4
  MAX_AUDIO_SEC              optional override; default 6.0
  FACE_ENCODER / AUDIO_ENCODER / EMBED_DIM / MASK_PROB
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

sys.path.insert(0, str(Path(__file__).parent))

from model import (
    AudioEncoder,
    CrossAttentionFusionModel,
    FaceEncoder,
    FusionModel,
    GatedFusionModel,
)


TARGET_SR: int = 16_000


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip()
    return env


def _get(env: dict[str, str], key: str, default: str) -> str:
    return env.get(key, os.environ.get(key, default))


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def make_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_and_pad_audio(wav_path: Path, max_samples: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(wav_path)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    waveform = waveform.mean(0)
    current = waveform.shape[0]
    if current < max_samples:
        waveform = F.pad(waveform, (0, max_samples - current))
    elif current > max_samples:
        start = (current - max_samples) // 2
        waveform = waveform[start : start + max_samples]
    return waveform


def load_face_frames(img_path: Path, transform: transforms.Compose, k: int) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    frame = transform(img)
    return frame.unsqueeze(0).expand(k, -1, -1, -1)


def build_model(cfg: dict, num_speakers: int) -> FusionModel | CrossAttentionFusionModel | GatedFusionModel:
    embed_dim = int(cfg.get("embed_dim", 256))
    face_enc = FaceEncoder(
        backbone=cfg.get("face_encoder", "vit_base"),
        embed_dim=embed_dim,
        pretrained=False,
    )
    audio_enc = AudioEncoder(
        backbone=cfg.get("audio_encoder", "wavlm_base"),
        embed_dim=embed_dim,
        pretrained=False,
    )
    fusion_type = str(cfg.get("fusion_type", "late")).strip().lower()
    mask_prob = float(cfg.get("mask_prob", 0.3))

    if fusion_type == "late":
        return FusionModel(
            face_encoder=face_enc,
            audio_encoder=audio_enc,
            num_speakers=num_speakers,
            embed_dim=embed_dim,
            mask_prob=mask_prob,
        )

    if fusion_type == "cross_attn":
        return CrossAttentionFusionModel(
            face_encoder=face_enc,
            audio_encoder=audio_enc,
            num_speakers=num_speakers,
            embed_dim=embed_dim,
            mask_prob=mask_prob,
            num_heads=int(cfg.get("cross_attn_heads", 8)),
            num_languages=(
                int(cfg.get("num_languages", 2))
                if bool(cfg.get("enable_lase", False))
                else None
            ),
        )

    if fusion_type == "gated":
        return GatedFusionModel(
            face_encoder=face_enc,
            audio_encoder=audio_enc,
            num_speakers=num_speakers,
            embed_dim=embed_dim,
            mask_prob=mask_prob,
            num_languages=(
                int(cfg.get("num_languages", 2))
                if bool(cfg.get("enable_lase", False))
                else None
            ),
        )

    raise ValueError(
        f"Unknown fusion_type={fusion_type!r}. Choose from late, cross_attn, gated."
    )


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        required = ["audio_path", "face_path", "identity"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns {missing}: {csv_path}")
        return list(reader)


def main() -> None:
    env = _load_env()
    repo_root = Path(__file__).parent.parent

    def _abs(rel: str) -> Path:
        return _resolve_path(repo_root, rel)

    csv_cfg = _get(env, "EVAL_ENGLISH_CSV", " ").strip()
    out_cfg = _get(env, "EVAL_ENGLISH_OUTPUT_CSV", "").strip()
    ckpt_cfg = _get(env, "EVAL_ENGLISH_CHECKPOINT", "").strip()

    csv_path = _abs(csv_cfg)
    out_path = _abs(out_cfg) if out_cfg else csv_path
    ckpt_path = _abs(ckpt_cfg)

    max_audio_sec = float(_get(env, "EVAL_ENGLISH_MAX_AUDIO_SEC", _get(env, "MAX_AUDIO_SEC", "6.0")))
    k_faces = int(_get(env, "EVAL_ENGLISH_K_FACES", "1"))
    batch_size = int(_get(env, "EVAL_ENGLISH_BATCH_SIZE", "64"))
    device_name = _get(env, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    use_amp = (_get(env, "EVAL_ENGLISH_USE_AMP", "true").lower() == "true") and device.type == "cuda"

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    rows = _load_rows(csv_path)
    speaker_ids = sorted({row["identity"].strip() for row in rows if row.get("identity")})
    if not speaker_ids:
        raise ValueError(f"No identity values found in {csv_path}")

    identity_to_idx = {identity: idx for idx, identity in enumerate(speaker_ids)}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg: dict = ckpt.get("cfg", {})
    resolved_cfg: dict[str, object] = dict(ckpt_cfg)
    resolved_cfg.setdefault("face_encoder", _get(env, "FACE_ENCODER", "vit_base"))
    resolved_cfg.setdefault("audio_encoder", _get(env, "AUDIO_ENCODER", "wavlm_base"))
    resolved_cfg.setdefault("embed_dim", int(_get(env, "EMBED_DIM", "256")))
    resolved_cfg.setdefault("mask_prob", float(_get(env, "MASK_PROB", "0.3")))

    model = build_model(resolved_cfg, len(speaker_ids))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    face_tf = make_eval_transform()
    max_samples = int(max_audio_sec * TARGET_SR)

    print("\n" + "═" * 72)
    print(" POLY-SIM  |  English training-set evaluation")
    print("═" * 72)
    print(f"  CSV input   : {csv_path}")
    print(f"  CSV output  : {out_path}")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Speakers    : {len(speaker_ids)}")
    print(f"  Batch size  : {batch_size}")
    print(f"  K faces     : {k_faces}")
    print(f"  Audio sec   : {max_audio_sec}")
    print(f"  AMP         : {use_amp}")
    print(f"  Device      : {device}")
    print("═" * 72 + "\n")

    total = 0
    correct = 0
    pred_hist: Counter[int] = Counter()
    out_rows: list[dict[str, str]] = []

    def batcher(seq, size):
        for pos in range(0, len(seq), size):
            yield seq[pos:pos+size]

    total_batches = (len(rows) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_rows in tqdm(
            batcher(rows, batch_size),
            total=total_batches,
            desc="Scoring English train CSV",
            unit="batch",
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            audio_paths = [
                _resolve_path(repo_root, row["audio_path"].strip()) for row in batch_rows
            ]
            face_paths = [
                _resolve_path(repo_root, row["face_path"].strip()) for row in batch_rows
            ]

            for audio_path, face_path in zip(audio_paths, face_paths):
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                if not face_path.exists():
                    raise FileNotFoundError(f"Face file not found: {face_path}")

            face_tensors = [load_face_frames(face_path, face_tf, k_faces) for face_path in face_paths]
            waveforms = [load_and_pad_audio(audio_path, max_samples) for audio_path in audio_paths]

            face_batch = torch.stack(face_tensors).to(device, non_blocking=True)
            wave_batch = torch.stack(waveforms).to(device, non_blocking=True)
            mask_none = torch.zeros(len(batch_rows), dtype=torch.bool, device=device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(face_batch, wave_batch, mask_faces=mask_none)
            if not isinstance(outputs, tuple) or len(outputs) < 2:
                raise RuntimeError("Model forward did not return logits in expected tuple output")
            logits = outputs[1]
            preds = logits.argmax(dim=1).cpu().numpy()

            for row, pred_idx in zip(batch_rows, preds):
                pred_hist[int(pred_idx)] += 1
                gt_label = row.get("label", "").strip()
                if gt_label:
                    gt_idx = int(float(gt_label))
                    correct += int(pred_idx == gt_idx)
                    total += 1
                out_row = dict(row)
                out_row["pred"] = str(pred_idx)
                out_rows.append(out_row)

    fieldnames = list(rows[0].keys())
    if "pred" not in fieldnames:
        fieldnames.append("pred")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    if total > 0:
        acc = correct / total * 100.0
        print(f"  Local accuracy : {acc:.2f}% ({correct}/{total})")
    print(f"  Unique preds   : {len(pred_hist)}")
    print(f"  Saved CSV      : {out_path}")
    print("═" * 72 + "\n")

    if SummaryWriter is not None:
        tb_dir = _abs(_get(env, "EVAL_ENGLISH_TENSORBOARD_DIR", "runs/eval_english"))
        writer = SummaryWriter(log_dir=str(tb_dir))
        try:
            writer.add_scalar("EvalEnglish/num_rows", len(rows), 0)
            writer.add_scalar("EvalEnglish/num_speakers", len(speaker_ids), 0)
            if total > 0:
                writer.add_scalar("EvalEnglish/accuracy", acc, 0)
        finally:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    main()