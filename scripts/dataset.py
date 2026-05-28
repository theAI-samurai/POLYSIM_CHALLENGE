"""
dataset.py — MAV-Celeb English dataset for multimodal speaker identification.

Data layout expected:
    <data_root>/
        faces/
            <speaker_id>/
                <video_folder>/
                    *.jpg          (frames already at 224×224)
        voices/
            <speaker_id>/
                <video_folder>/
                    *.wav          (mono, 16 kHz)

Each __getitem__ returns:
    face_frames  : Tensor [K, 3, H, W]   — K frames sampled from the same video
    waveform     : Tensor [T]            — cropped/padded mono waveform at 16 kHz
    label        : int                   — speaker index (0 … num_speakers-1)
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ─────────────────────────────────────────────────────────────────────────────
# Voice Activity Detection (VAD)
# ─────────────────────────────────────────────────────────────────────────────

def apply_vad(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    threshold_db: float = -40.0,
    frame_length_ms: int = 20,
) -> torch.Tensor:
    """Apply energy-based Voice Activity Detection to remove silence.
    
    Args:
        waveform:        Mono audio tensor [T]
        sample_rate:     Sample rate in Hz (default 16kHz)
        threshold_db:    Energy threshold in dB below max (default -40dB)
        frame_length_ms: Frame length for energy computation (default 20ms)
    
    Returns:
        waveform:        VAD-processed waveform (silence trimmed)
    """
    if not HAS_LIBROSA:
        # VAD disabled if librosa not available
        return waveform
    
    # Convert to numpy
    wav_np = waveform.cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform
    
    # Compute STFT for energy-based VAD
    S = librosa.stft(wav_np, n_fft=2048, hop_length=sample_rate // 50)  # 20ms hop
    energy = np.abs(S).mean(axis=0)
    energy_db = 20 * np.log10(np.maximum(energy, 1e-10))
    
    # Threshold: max energy - threshold_db
    max_energy_db = np.max(energy_db)
    threshold = max_energy_db + threshold_db
    
    # Find speech frames (above threshold)
    speech_frames = energy_db > threshold
    
    # Smooth tiny frame-level fluctuations in a version-safe way.
    # Use a short moving-average vote instead of librosa.util.frame, whose
    # signature differs across librosa versions.
    speech_f = speech_frames.astype(np.float32)
    kernel = np.ones(3, dtype=np.float32) / 3.0
    speech_f = np.convolve(speech_f, kernel, mode="same")
    speech_frames = speech_f >= 0.5
    
    # Convert frame indices back to sample indices
    hop_length = sample_rate // 50  # 20ms hop
    speech_samples = np.concatenate([
        np.arange(i * hop_length, (i + 1) * hop_length)
        for i in range(len(speech_frames)) if speech_frames[i]
    ])
    speech_samples = speech_samples[speech_samples < len(wav_np)]
    
    if len(speech_samples) == 0:
        # No speech detected, return original (avoid empty audio)
        return waveform
    
    # Extract speech segments
    trimmed = wav_np[speech_samples]
    
    return torch.from_numpy(trimmed).float()


class CSVMAVCelebDataset(Dataset):
    """CSV-based MAV-Celeb dataset.

    Expected CSV columns:
      - audio_path
      - face_path
      - identity
    """

    TARGET_SR: int = 16_000

    def __init__(
        self,
        csv_path: str | Path,
        repo_root: str | Path,
        k_faces: int = 4,
        max_audio_sec: float = 6.0,
        sample_rate: int = TARGET_SR,
        face_transform=None,
        use_vad: bool = True,
        vad_threshold_db: float = -40.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.repo_root = Path(repo_root)
        self.data_root = self.repo_root
        self.k_faces = k_faces
        self.max_audio_samples = int(max_audio_sec * sample_rate)
        self.sample_rate = sample_rate
        self.face_transform = face_transform
        self.use_vad = use_vad
        self.vad_threshold_db = vad_threshold_db

        if not self.csv_path.is_absolute():
            self.csv_path = self.repo_root / self.csv_path
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.samples: list[tuple[str, Path, list[Path]]] = []
        speaker_counts: dict[str, int] = {}

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"audio_path", "face_path", "identity"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"CSV must contain columns {sorted(required)}. Found: {reader.fieldnames}"
                )

            for row_idx, row in enumerate(reader, start=2):
                speaker = (row.get("identity") or "").strip()
                audio_raw = (row.get("audio_path") or "").strip()
                face_raw = (row.get("face_path") or "").strip()

                if not speaker or not audio_raw or not face_raw:
                    continue

                wav_path = Path(audio_raw)
                face_path = Path(face_raw)
                if not wav_path.is_absolute():
                    wav_path = self.repo_root / wav_path
                if not face_path.is_absolute():
                    face_path = self.repo_root / face_path

                if not wav_path.exists():
                    raise FileNotFoundError(f"Row {row_idx}: audio file not found: {wav_path}")
                if not face_path.exists():
                    raise FileNotFoundError(f"Row {row_idx}: face file not found: {face_path}")

                # face_pool is kept as a list to match existing dataset contract.
                self.samples.append((speaker, wav_path, [face_path]))
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        self.speakers = sorted(speaker_counts.keys())
        self.speaker_to_idx = {s: i for i, s in enumerate(self.speakers)}
        self._speaker_counts = [speaker_counts[s] for s in self.speakers]

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    def get_sample_weights(self) -> list[float]:
        weights: list[float] = []
        for speaker, _, _ in self.samples:
            idx = self.speaker_to_idx[speaker]
            n = self._speaker_counts[idx]
            weights.append(1.0 / max(n, 1))
        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        speaker, wav_path, face_pool = self.samples[idx]
        label = self.speaker_to_idx[speaker]

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(0)

        if self.use_vad and HAS_LIBROSA:
            waveform = apply_vad(
                waveform,
                sample_rate=self.sample_rate,
                threshold_db=self.vad_threshold_db,
            )

        T = waveform.shape[0]
        if T < self.max_audio_samples:
            waveform = F.pad(waveform, (0, self.max_audio_samples - T))
        else:
            start = random.randint(0, T - self.max_audio_samples)
            waveform = waveform[start : start + self.max_audio_samples]

        chosen_paths = random.choices(face_pool, k=self.k_faces)
        frames: list[torch.Tensor] = []
        for fp in chosen_paths:
            img = Image.open(fp).convert("RGB")
            if self.face_transform is not None:
                img = self.face_transform(img)
            else:
                img = torch.tensor(
                    list(img.getdata()),
                    dtype=torch.float32,
                ).reshape(224, 224, 3).permute(2, 0, 1) / 255.0
            frames.append(img)

        face_tensor = torch.stack(frames)
        return face_tensor, waveform, label


class MAVCelebDataset(Dataset):
    """English-only MAV-Celeb dataset.

    Args:
        data_root:       Path to the prepared Data/ directory.
        k_faces:         Number of face frames to sample per audio clip.
        max_audio_sec:   Maximum audio duration in seconds (clips are cropped/padded).
        sample_rate:     Target sample rate. Source files are already 16 kHz.
        face_transform:  torchvision transform applied to each PIL face image.
    """

    TARGET_SR: int = 16_000

    def __init__(
        self,
        data_root: str | Path,
        k_faces: int = 4,
        max_audio_sec: float = 6.0,
        sample_rate: int = TARGET_SR,
        face_transform=None,
        use_vad: bool = True,
        vad_threshold_db: float = -40.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.k_faces = k_faces
        self.max_audio_samples = int(max_audio_sec * sample_rate)
        self.sample_rate = sample_rate
        self.face_transform = face_transform
        self.use_vad = use_vad
        self.vad_threshold_db = vad_threshold_db

        faces_root = self.data_root / "faces"
        voices_root = self.data_root / "voices"

        if not faces_root.is_dir():
            raise FileNotFoundError(f"Faces directory not found: {faces_root}")
        if not voices_root.is_dir():
            raise FileNotFoundError(f"Voices directory not found: {voices_root}")

        # Sorted speaker list — deterministic label assignment
        self.speakers: list[str] = sorted(
            p.name for p in faces_root.iterdir() if p.is_dir()
        )
        self.speaker_to_idx: dict[str, int] = {
            s: i for i, s in enumerate(self.speakers)
        }

        # samples[i] = (speaker_id, wav_path, face_pool)
        # face_pool is a list of jpg Paths for the same video
        self.samples: list[tuple[str, Path, list[Path]]] = []

        # per-speaker sample counts (for WeightedRandomSampler)
        self._speaker_counts: list[int] = []

        for speaker in self.speakers:
            count_before = len(self.samples)
            voice_spk = voices_root / speaker
            face_spk = faces_root / speaker

            if not voice_spk.is_dir():
                self._speaker_counts.append(0)
                continue

            for video_dir in sorted(v for v in voice_spk.iterdir() if v.is_dir()):
                face_video_dir = face_spk / video_dir.name
                if not face_video_dir.is_dir():
                    continue

                face_pool = sorted(face_video_dir.glob("*.jpg"))
                if not face_pool:
                    continue

                for wav in sorted(video_dir.glob("*.wav")):
                    self.samples.append((speaker, wav, face_pool))

            self._speaker_counts.append(len(self.samples) - count_before)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    def get_sample_weights(self) -> list[float]:
        """Per-sample weights for WeightedRandomSampler (equalises speakers)."""
        weights: list[float] = []
        for speaker, _, _ in self.samples:
            idx = self.speaker_to_idx[speaker]
            n = self._speaker_counts[idx]
            weights.append(1.0 / max(n, 1))
        return weights

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        speaker, wav_path, face_pool = self.samples[idx]
        label = self.speaker_to_idx[speaker]

        # ---- Audio -------------------------------------------------------
        waveform, sr = torchaudio.load(wav_path)           # [C, T]
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(0)                        # mono [T]

        # Apply Voice Activity Detection (VAD) to remove silence
        if self.use_vad and HAS_LIBROSA:
            waveform = apply_vad(
                waveform,
                sample_rate=self.sample_rate,
                threshold_db=self.vad_threshold_db,
            )

        T = waveform.shape[0]
        if T < self.max_audio_samples:
            # Zero-pad short clips
            waveform = F.pad(waveform, (0, self.max_audio_samples - T))
        else:
            # Random crop
            start = random.randint(0, T - self.max_audio_samples)
            waveform = waveform[start : start + self.max_audio_samples]

        # ---- Faces -------------------------------------------------------
        chosen_paths = random.choices(face_pool, k=self.k_faces)
        frames: list[torch.Tensor] = []
        for fp in chosen_paths:
            img = Image.open(fp).convert("RGB")
            if self.face_transform is not None:
                img = self.face_transform(img)
            else:
                img = torch.tensor(
                    list(img.getdata()),
                    dtype=torch.float32,
                ).reshape(224, 224, 3).permute(2, 0, 1) / 255.0
            frames.append(img)

        face_tensor = torch.stack(frames)  # [K, 3, H, W]

        return face_tensor, waveform, label
