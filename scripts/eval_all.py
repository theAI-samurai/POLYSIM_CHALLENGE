"""
eval_all.py — Unified evaluation for POLY-SIM protocols P3, P4, P5, P6.

What this script does
---------------------
Runs all four challenge protocols in one execution using a single checkpoint:

  P3: Face + Audio, English -> English
  P4: Audio only,   English -> English
  P5: Face + Audio, English -> Urdu
  P6: Audio only,   English -> Urdu

Output files
------------
1) Monolingual CSV (challenge format): key,p3,p4
2) Cross-lingual CSV (challenge format): key,p5,p6

Optionally, local labeled metrics can be logged to TensorBoard if labels CSVs
are provided for the same-language and/or cross-language splits.

Usage:
    python scripts/eval_all.py

Env variables (all optional)
----------------------------
General:
  EVAL_ALL_CHECKPOINT      fallback: EVAL_CHECKPOINT, then checkpoints/best.pt
  DATA_ROOT                fallback: Data
  DATA_TEST_ROOT           fallback: Data_Test
  OUTPUT_DIR               fallback: checkpoints
  DEVICE                   fallback: cuda if available else cpu
  K_FACES                  fallback: 4
  MAX_AUDIO_SEC            fallback: 6.0

Language folders:
  EVAL_LANG_SAME           fallback: English
  EVAL_LANG_CROSS          fallback: Urdu

CSV outputs:
  EVAL_OUTPUT_EN_EN        fallback: submission_v1_test_English_English.csv
  EVAL_OUTPUT_EN_UR        fallback: submission_v1_test_English_Urdu.csv

CSV inputs:
    EVAL_ALL_SAME_CSV        fallback: empty (use folder layout)
    EVAL_ALL_CROSS_CSV       fallback: empty (use folder layout)
    EVAL_ALL_BATCH_SIZE      fallback: 16

TensorBoard:
  EVAL_ALL_TENSORBOARD_DIR fallback: runs/eval_all

Optional local GT labels:
  EVAL_LABELS_SAME_CSV     fallback: EVAL_LABELS_CSV
  EVAL_LABELS_CROSS_CSV    fallback: empty
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

# Allow running from repo root: python scripts/eval_all.py
sys.path.insert(0, str(Path(__file__).parent))

from model import (
    AudioEncoder,
    CrossAttentionFusionModel,
    FaceEncoder,
    FusionModel,
    GatedFusionModel,
)


TARGET_SR: int = 16_000
_TORCHAUDIO = None


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def _get(env: dict[str, str], key: str, default: str) -> str:
    return env.get(key, os.environ.get(key, default))


def _resolve_csv_output(repo_root: Path, output_dir: Path, csv_cfg: str) -> Path:
    """Resolve CSV path with the same policy as eval_p3.py.

    - Absolute path: use as-is
    - Relative path with directory: resolve from repo root
    - Filename only: write under OUTPUT_DIR
    """
    path = Path(csv_cfg)
    if path.is_absolute():
        out = path
    elif path.parent != Path("."):
        out = repo_root / path
    else:
        out = output_dir / path
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_eval_rows_from_csv(csv_path: Path, repo_root: Path) -> list[tuple[str, Path, Path]]:
    """Load eval samples from a CSV containing key plus audio/face path columns."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Eval CSV has no header row: {csv_path}")

        audio_col = None
        face_col = None
        for candidate in ("audio_path", "voices", "voice_path", "wav_path"):
            if candidate in reader.fieldnames:
                audio_col = candidate
                break
        for candidate in ("face_path", "faces", "image_path", "img_path"):
            if candidate in reader.fieldnames:
                face_col = candidate
                break

        if "key" not in reader.fieldnames or audio_col is None or face_col is None:
            raise ValueError(
                "Eval CSV must contain 'key' and one audio path column plus one face path column. "
                f"Found columns: {reader.fieldnames}"
            )

        rows: list[tuple[str, Path, Path]] = []
        for row in reader:
            key = (row.get("key") or "").strip()
            audio_path = (row.get(audio_col) or "").strip()
            face_path = (row.get(face_col) or "").strip()
            if not key or not audio_path or not face_path:
                continue
            rows.append((key, _resolve_path(repo_root, face_path), _resolve_path(repo_root, audio_path)))
    return rows


def _load_speakers_from_train_csv(train_csv: Path) -> list[str]:
    """Load unique speaker IDs from a training CSV.

    Expected column priority:
      - identity (preferred)
      - speaker_id
      - speaker
    """
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Training CSV has no header row: {train_csv}")

        id_col = None
        for candidate in ("identity", "speaker_id", "speaker"):
            if candidate in reader.fieldnames:
                id_col = candidate
                break

        if id_col is None:
            raise ValueError(
                "Training CSV must contain one of ['identity', 'speaker_id', 'speaker']. "
                f"Found columns: {reader.fieldnames}"
            )

        speakers = {
            (row.get(id_col) or "").strip()
            for row in reader
            if (row.get(id_col) or "").strip()
        }

    if not speakers:
        raise ValueError(f"No speaker IDs found in training CSV: {train_csv}")

    return sorted(speakers)


def _chunks(seq: list[tuple[str, Path, Path]], size: int) -> list[list[tuple[str, Path, Path]]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _summarize_eval_source(
    title: str,
    source: str,
    rows: list[tuple[str, Path, Path]],
) -> None:
    """Print a compact summary of where evaluation samples are read from."""
    print(f"  {title} source      : {source}")
    print(f"  {title} samples     : {len(rows)}")
    if not rows:
        return

    # Show first resolved sample path for quick verification.
    key0, face0, audio0 = rows[0]
    print(f"  {title} first key   : {key0}")
    print(f"  {title} first face  : {face0}  [exists={face0.exists()}]")
    print(f"  {title} first audio : {audio0}  [exists={audio0.exists()}]")

    # Quick health signal for path resolution quality.
    missing_face = sum(1 for _, f, _ in rows if not f.exists())
    missing_audio = sum(1 for _, _, a in rows if not a.exists())
    print(
        f"  {title} missing     : faces={missing_face}  "
        f"audio={missing_audio}"
    )


def _load_labels_csv(path: Path, protocol_label_col: str) -> dict[str, int]:
    """Load optional local labels for metrics.

    Required columns:
      - key
      - one of: label, speaker, speaker_id, target, <protocol_label_col>
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header: {path}")

        label_col = None
        for candidate in ("label", "speaker", "speaker_id", "target", protocol_label_col):
            if candidate in reader.fieldnames:
                label_col = candidate
                break

        if "key" not in reader.fieldnames or label_col is None:
            raise ValueError(
                "Labels CSV must contain 'key' and one of "
                f"['label', 'speaker', 'speaker_id', 'target', '{protocol_label_col}']"
            )

        labels: dict[str, int] = {}
        for row in reader:
            key = (row.get("key") or "").strip()
            val = (row.get(label_col) or "").strip()
            if not key or not val:
                continue
            labels[key] = int(val)
    return labels


def make_eval_transform() -> transforms.Compose:
    """Deterministic transform used for all protocols at evaluation time."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_and_pad_audio(wav_path: Path, max_samples: int) -> torch.Tensor:
    """Load waveform, resample to 16 kHz, convert to mono, pad/crop to fixed size."""
    global _TORCHAUDIO
    if _TORCHAUDIO is None:
        import torchaudio as _ta  # Delayed import for clearer startup diagnostics.
        _TORCHAUDIO = _ta

    torchaudio = _TORCHAUDIO
    waveform, sr = torchaudio.load(wav_path)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    waveform = waveform.mean(0)
    t = waveform.shape[0]
    if t < max_samples:
        waveform = F.pad(waveform, (0, max_samples - t))
    elif t > max_samples:
        start = (t - max_samples) // 2
        waveform = waveform[start: start + max_samples]
    return waveform


def load_face_frames(img_path: Path, transform: transforms.Compose, k: int) -> torch.Tensor:
    """Load one representative face image and tile it to K frames."""
    img = Image.open(img_path).convert("RGB")
    frame = transform(img)
    return frame.unsqueeze(0).expand(k, -1, -1, -1)


def build_model(cfg: dict, num_speakers: int) -> FusionModel | CrossAttentionFusionModel | GatedFusionModel:
    """Rebuild model architecture from checkpoint config."""
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


def _evaluate_pair(
    *,
    model: FusionModel,
    device: torch.device,
    samples: list[tuple[str, Path, Path]],
    face_tf: transforms.Compose,
    k_faces: int,
    max_samples: int,
    idx_to_num: dict[int, int],
    num_to_idx: dict[int, int],
    proto_full: str,
    proto_audio: str,
    labels: dict[str, int] | None,
    batch_size: int,
) -> tuple[list[dict[str, int | str]], dict[str, float]]:
    """Evaluate one language subset for a pair of protocols (full + audio-only)."""

    if not samples:
        raise ValueError("No evaluation samples provided.")

    print(f"  Samples ({proto_full}/{proto_audio}): {len(samples)}")

    rows: list[dict[str, int | str]] = []
    full_ce_sum = audio_ce_sum = 0.0
    full_correct = audio_correct = 0
    gt_count = 0

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode():
        with autocast_ctx:
            for batch in tqdm(_chunks(samples, max(1, batch_size)), desc=f"Evaluating {proto_full}/{proto_audio}", unit="batch"):
                keys = [key for key, _, _ in batch]
                face_tensors = [load_face_frames(face_path, face_tf, k_faces) for _, face_path, _ in batch]
                waveforms = [load_and_pad_audio(audio_path, max_samples) for _, _, audio_path in batch]

                face_batch = torch.stack(face_tensors).to(device)
                wave_batch = torch.stack(waveforms).to(device)

                # Full modality protocol (P3 or P5)
                mask_none = torch.zeros(len(batch), dtype=torch.bool, device=device)
                _, logits_full = model(face_batch, wave_batch, mask_faces=mask_none)
                probs_full = torch.softmax(logits_full, dim=1)
                pred_full_idxs = probs_full.argmax(dim=1).tolist()

                # Audio-only protocol (P4 or P6)
                mask_all = torch.ones(len(batch), dtype=torch.bool, device=device)
                _, logits_audio = model(face_batch, wave_batch, mask_faces=mask_all)
                probs_audio = torch.softmax(logits_audio, dim=1)
                pred_audio_idxs = probs_audio.argmax(dim=1).tolist()

                for row_idx, key in enumerate(keys):
                    pred_full_idx = int(pred_full_idxs[row_idx])
                    pred_audio_idx = int(pred_audio_idxs[row_idx])

                    rows.append(
                        {
                            "key": key,
                            proto_full: idx_to_num[pred_full_idx],
                            proto_audio: idx_to_num[pred_audio_idx],
                        }
                    )

                    if labels is not None and key in labels:
                        gt_num = int(labels[key])
                        if gt_num in num_to_idx:
                            gt_idx = num_to_idx[gt_num]
                            gt_tensor = torch.tensor([gt_idx], dtype=torch.long, device=device)
                            full_ce_sum += F.cross_entropy(
                                logits_full[row_idx : row_idx + 1], gt_tensor, reduction="sum"
                            ).item()
                            audio_ce_sum += F.cross_entropy(
                                logits_audio[row_idx : row_idx + 1], gt_tensor, reduction="sum"
                            ).item()
                            full_correct += int(pred_full_idx == gt_idx)
                            audio_correct += int(pred_audio_idx == gt_idx)
                            gt_count += 1

    metrics: dict[str, float] = {
        f"{proto_full}_unique": float(len({int(r[proto_full]) for r in rows})),
        f"{proto_audio}_unique": float(len({int(r[proto_audio]) for r in rows})),
        "count": float(len(rows)),
        "gt_count": float(gt_count),
    }

    if gt_count > 0:
        metrics[f"{proto_full}_acc"] = full_correct / gt_count * 100.0
        metrics[f"{proto_audio}_acc"] = audio_correct / gt_count * 100.0
        metrics[f"{proto_full}_loss"] = full_ce_sum / gt_count
        metrics[f"{proto_audio}_loss"] = audio_ce_sum / gt_count

    return rows, metrics


def main() -> None:
    """Entry point for evaluating all four challenge protocols in one run."""
    env = _load_env()
    repo_root = Path(__file__).parent.parent

    def _abs(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    ckpt_path = _abs(_get(env, "EVAL_ALL_CHECKPOINT", _get(env, "EVAL_CHECKPOINT", "checkpoints/best.pt")))
    data_train_root = _abs(_get(env, "DATA_ROOT", "Data"))
    train_csv_cfg = _get(env, "EVAL_ALL_TRAIN_CSV", _get(env, "TRAIN_CSV", "data_train/comp/v1_train_English.csv")).strip()
    train_csv_path = _resolve_path(repo_root, train_csv_cfg)
    data_test_root = _abs(_get(env, "DATA_TEST_ROOT", ""))
    same_csv_cfg = _get(env, "EVAL_ALL_SAME_CSV", "data_train/comp/v1_test_English.csv").strip()
    cross_csv_cfg = _get(env, "EVAL_ALL_CROSS_CSV", "data_train/comp/v1_test_Urdu.csv").strip()
    output_dir = _abs(_get(env, "OUTPUT_DIR", "checkpoints"))

    lang_same = _get(env, "EVAL_LANG_SAME", "English").strip()
    lang_cross = _get(env, "EVAL_LANG_CROSS", "Urdu").strip()

    eval_lang       = _get(env, "EVAL_LANG", "English").strip()
    eval_set       = _get(env, "EVAL_SET", "dev").strip()

    csv_en_en = _get(env, "EVAL_OUTPUT_EN_EN", f"submission_v1_{eval_set}_{lang_same}_{lang_same}.csv").strip() 
    csv_en_ur = _get(env, "EVAL_OUTPUT_EN_UR", f"submission_v1_{eval_set}_{lang_same}_{lang_cross}.csv").strip()

    max_audio_sec = float(_get(env, "MAX_AUDIO_SEC", "6.0"))
    eval_batch_size = int(_get(env, "EVAL_ALL_BATCH_SIZE", "16"))
    tb_dir = _abs(_get(env, "EVAL_ALL_TENSORBOARD_DIR", "runs/eval_all"))

    labels_same_csv = _get(env, "EVAL_LABELS_SAME_CSV", _get(env, "EVAL_LABELS_CSV", "")).strip()
    labels_cross_csv = _get(env, "EVAL_LABELS_CROSS_CSV", "").strip()

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(_get(env, "DEVICE", auto_device))

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 72)
    print(" POLY-SIM  |  Unified Evaluation  |  P3 P4 P5 P6")
    print("═" * 72)
    print(f"  Checkpoint      : {ckpt_path}")
    print(f"  Train data      : {data_train_root}")
    print(f"  Train CSV       : {train_csv_path}")
    print(f"  Test data       : {data_test_root}")
    print(f"  Same language   : {lang_same}")
    print(f"  Cross language  : {lang_cross}")
    print(f"  Device          : {device}")
    print(f"  Output EN-EN    : {csv_en_en}")
    print(f"  Output EN-UR    : {csv_en_ur}")
    print(f"  TensorBoard dir : {tb_dir}")
    print("═" * 72 + "\n")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Set EVAL_ALL_CHECKPOINT/EVAL_CHECKPOINT in .env"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg: dict = ckpt.get("cfg", {})

    # Use checkpoint architecture first, fall back to env for older ckpts.
    resolved_cfg: dict[str, object] = dict(ckpt_cfg)
    resolved_cfg.setdefault("face_encoder", _get(env, "FACE_ENCODER", "vit_base"))
    resolved_cfg.setdefault("audio_encoder", _get(env, "AUDIO_ENCODER", "wavlm_base"))
    resolved_cfg.setdefault("embed_dim", int(_get(env, "EMBED_DIM", "256")))
    resolved_cfg.setdefault("mask_prob", float(_get(env, "MASK_PROB", "0.3")))

    face_encoder_name = str(resolved_cfg["face_encoder"])
    audio_encoder_name = str(resolved_cfg["audio_encoder"])

    print(
        f"  Loaded epoch {ckpt.get('epoch', '?')} "
        f"(best train acc {ckpt.get('best_acc', 0.0):.2f}%)"
    )
    print(f"  Face/Image encoder: {face_encoder_name}")
    print(f"  Audio encoder     : {audio_encoder_name}")

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")

    speakers: list[str] = _load_speakers_from_train_csv(train_csv_path)
    num_speakers = len(speakers)
    # Challenge labels are class indices, so emit 0..N-1 directly.
    idx_to_num: dict[int, int] = {i: i for i in range(num_speakers)}
    num_to_idx: dict[int, int] = {i: i for i in range(num_speakers)}
    print(f"  Speakers (train) : {num_speakers} (from training CSV)\n")

    model = build_model(resolved_cfg, num_speakers)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    face_tf = make_eval_transform()
    max_samples = int(max_audio_sec * TARGET_SR)
    k_faces = int(resolved_cfg.get("k_faces", _get(env, "K_FACES", "4")))

    same_samples: list[tuple[str, Path, Path]]
    cross_samples: list[tuple[str, Path, Path]]
    same_source: str
    cross_source: str
    if same_csv_cfg:
        same_csv_path = _resolve_path(repo_root, same_csv_cfg)
        if not same_csv_path.exists():
            raise FileNotFoundError(f"EVAL_ALL_SAME_CSV not found: {same_csv_path}")
        same_samples = _load_eval_rows_from_csv(same_csv_path, repo_root)
        same_source = str(same_csv_path)
        print(f"  Same-language CSV : {same_csv_path} ({len(same_samples)} rows)")
    else:
        same_samples = [
            (p.stem, p, data_test_root / "voices" / lang_same / f"{p.stem}.wav")
            for p in sorted((data_test_root / "faces" / lang_same).glob("*.jpg"))
        ]
        same_source = f"folder:{data_test_root / 'faces' / lang_same}"

    if cross_csv_cfg:
        cross_csv_path = _resolve_path(repo_root, cross_csv_cfg)
        if not cross_csv_path.exists():
            raise FileNotFoundError(f"EVAL_ALL_CROSS_CSV not found: {cross_csv_path}")
        cross_samples = _load_eval_rows_from_csv(cross_csv_path, repo_root)
        cross_source = str(cross_csv_path)
        print(f"  Cross-language CSV: {cross_csv_path} ({len(cross_samples)} rows)")
    else:
        cross_samples = [
            (p.stem, p, data_test_root / "voices" / lang_cross / f"{p.stem}.wav")
            for p in sorted((data_test_root / "faces" / lang_cross).glob("*.jpg"))
        ]
        cross_source = f"folder:{data_test_root / 'faces' / lang_cross}"

    print("\n  Input summary")
    _summarize_eval_source("Same-lang", same_source, same_samples)
    _summarize_eval_source("Cross-lang", cross_source, cross_samples)
    print()

    labels_same: dict[str, int] | None = None
    if labels_same_csv:
        p = Path(labels_same_csv)
        p = p if p.is_absolute() else repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"EVAL_LABELS_SAME_CSV not found: {p}")
        labels_same = _load_labels_csv(p, "p3")
        print(f"  Loaded same-language labels  : {len(labels_same)} from {p}")

    labels_cross: dict[str, int] | None = None
    if labels_cross_csv:
        p = Path(labels_cross_csv)
        p = p if p.is_absolute() else repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"EVAL_LABELS_CROSS_CSV not found: {p}")
        labels_cross = _load_labels_csv(p, "p5")
        print(f"  Loaded cross-language labels : {len(labels_cross)} from {p}")

    # P3/P4 on same-language subset
    rows_34, metrics_34 = _evaluate_pair(
        model=model,
        device=device,
        samples=same_samples,
        face_tf=face_tf,
        k_faces=k_faces,
        max_samples=max_samples,
        idx_to_num=idx_to_num,
        num_to_idx=num_to_idx,
        proto_full="p3",
        proto_audio="p4",
        labels=labels_same,
        batch_size=eval_batch_size,
    )

    # P5/P6 on cross-language subset
    rows_56, metrics_56 = _evaluate_pair(
        model=model,
        device=device,
        samples=cross_samples,
        face_tf=face_tf,
        k_faces=k_faces,
        max_samples=max_samples,
        idx_to_num=idx_to_num,
        num_to_idx=num_to_idx,
        proto_full="p5",
        proto_audio="p6",
        labels=labels_cross,
        batch_size=eval_batch_size,
    )

    out_34 = _resolve_csv_output(repo_root, output_dir, csv_en_en)
    out_56 = _resolve_csv_output(repo_root, output_dir, csv_en_ur)

    with open(out_34, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "p3", "p4"])
        writer.writeheader()
        writer.writerows(rows_34)

    with open(out_56, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "p5", "p6"])
        writer.writeheader()
        writer.writerows(rows_56)

    print("\n" + "═" * 72)
    print("  Unified evaluation complete")
    print(f"  P3/P4 samples        : {int(metrics_34['count'])}")
    print(f"  P5/P6 samples        : {int(metrics_56['count'])}")
    print(f"  P3/P4 CSV            : {out_34}")
    print(f"  P5/P6 CSV            : {out_56}")

    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_dir))
        try:
            step = int(ckpt.get("epoch", 0))
            for k, v in metrics_34.items():
                writer.add_scalar(f"EvalAll/{k}", v, step)
            for k, v in metrics_56.items():
                writer.add_scalar(f"EvalAll/{k}", v, step)

            # Overall mean accuracy if both label files are present.
            if "p3_acc" in metrics_34 and "p4_acc" in metrics_34 and "p5_acc" in metrics_56 and "p6_acc" in metrics_56:
                overall = (metrics_34["p3_acc"] + metrics_34["p4_acc"] + metrics_56["p5_acc"] + metrics_56["p6_acc"]) / 4.0
                writer.add_scalar("EvalAll/overall_mean_acc", overall, step)
                print(f"  Overall mean acc     : {overall:.2f}%")

            print(f"  TensorBoard saved    : {tb_dir}")
        finally:
            writer.flush()
            writer.close()
    else:
        print("  TensorBoard          : not available (pip install tensorboard)")

    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
