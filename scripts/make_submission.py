"""
make_submission.py — Package challenge CSV files into a submission ZIP.

What this script does
----------------------
After running eval_all.py (which writes the two challenge-format CSVs), this
script:

  1. Locates both CSV files using the same .env paths as eval_all.py.
  2. Validates each file has the required header and non-empty predictions.
  3. Creates a ZIP archive (submission.zip by default) with both files at the
     top level — not inside a subdirectory — matching the challenge spec:

        zip submission.zip *.csv   (do NOT zip the folder itself)

  4. Prints a pre-flight checklist so you can verify the archive before upload.

Challenge file naming
---------------------
  submission_v1_<phase>_English_English.csv   columns: key, p3, p4
  submission_v1_<phase>_English_Urdu.csv      columns: key, p5, p6

  <phase> is either "val" (progress phase) or "test" (evaluation phase).

Usage
-----
  # Default: packages both CSVs into submission.zip
  python scripts/make_submission.py

  # Override output ZIP name
  SUBMISSION_ZIP=my_team_submission.zip python scripts/make_submission.py

  # Progress-phase submission (val split)
  EVAL_PHASE=val python scripts/make_submission.py

Environment variables (read from .env at repo root)
----------------------------------------------------
  EVAL_ALL_CHECKPOINT        (used to resolve OUTPUT_DIR fallback)
  OUTPUT_DIR                 fallback: checkpoints
  EVAL_OUTPUT_EN_EN          fallback: submission_v1_test_English_English.csv
  EVAL_OUTPUT_EN_UR          fallback: submission_v1_test_English_Urdu.csv
  EVAL_PHASE                 override phase token in filenames (val|test)
  SUBMISSION_ZIP             output ZIP path (default: submission.zip)
"""

from __future__ import annotations

import csv
import os
import sys
import zipfile
from pathlib import Path


# ─── Config ──────────────────────────────────────────────────────────────────

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


def _resolve_csv_path(repo_root: Path, output_dir: Path, cfg: str) -> Path:
    """Mirror the path-resolution logic used in eval_all.py."""
    p = Path(cfg)
    if p.is_absolute():
        return p
    if p.parent != Path("."):
        return repo_root / p
    return output_dir / p


# ─── Validation ──────────────────────────────────────────────────────────────

_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "en_en": ["key", "p3", "p4"],
    "en_ur": ["key", "p5", "p6"],
}


def _validate_csv(path: Path, required_cols: list[str]) -> tuple[bool, str, int]:
    """
    Check that path exists, has the required columns, and contains rows.

    Returns (ok, message, row_count).
    """
    if not path.exists():
        return False, f"File not found: {path}", 0

    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return False, "CSV has no header row", 0

            missing = [c for c in required_cols if c not in reader.fieldnames]
            if missing:
                return False, f"Missing columns: {missing}", 0

            extra = [c for c in reader.fieldnames if c not in required_cols]
            warning = f" (extra columns ignored: {extra})" if extra else ""

            rows = list(reader)
            if not rows:
                return False, "CSV has no data rows", 0

            # Spot-check: prediction columns must be integers.
            pred_cols = [c for c in required_cols if c != "key"]
            for i, row in enumerate(rows):
                for col in pred_cols:
                    val = (row.get(col) or "").strip()
                    if not val:
                        return False, f"Row {i+1}: empty value in column '{col}'", 0
                    try:
                        int(val)
                    except ValueError:
                        return False, f"Row {i+1}: non-integer in column '{col}': {val!r}", 0

        return True, f"OK{warning}", len(rows)

    except Exception as exc:  # noqa: BLE001
        return False, f"Read error: {exc}", 0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    env = _load_env()
    repo_root = Path(__file__).parent.parent

    def _abs(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    output_dir = _abs(_get(env, "OUTPUT_DIR", "checkpoints"))

    # CSV paths (mirrors eval_all.py resolution)
    csv_en_en_cfg = _get(
        env, "EVAL_OUTPUT_EN_EN","submission_v1_dev_English_English.csv",).strip()

    csv_en_ur_cfg = _get(env, "EVAL_OUTPUT_EN_UR", "submission_v1_dev_English_Urdu.csv",).strip()

    csv_en_en = _resolve_csv_path(repo_root, output_dir, csv_en_en_cfg)
    csv_en_ur = _resolve_csv_path(repo_root, output_dir, csv_en_ur_cfg)

    # Allow overriding phase in filenames (val / test)
    phase_override = _get(env, "EVAL_PHASE", "").strip()
    if phase_override:
        def _rephase(p: Path, phase: str) -> Path:
            """Replace the phase token (val|test) inside the stem."""
            for token in ("_val_", "_test_"):
                if token in p.name:
                    return p.with_name(p.name.replace(token, f"_{phase}_"))
            return p
        csv_en_en = _rephase(csv_en_en, phase_override)
        csv_en_ur = _rephase(csv_en_ur, phase_override)

    # ZIP output path
    zip_cfg = _get(env, "SUBMISSION_ZIP", "submission.zip").strip() or "submission.zip"
    zip_path = _abs(zip_cfg)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print(" POLY-SIM  |  Submission Packager")
    print("═" * 72)
    print(f"  EN-EN CSV : {csv_en_en}")
    print(f"  EN-UR CSV : {csv_en_ur}")
    print(f"  ZIP output: {zip_path}")
    print()

    # ── Validate both CSVs ───────────────────────────────────────────────────
    checks = [
        (csv_en_en, _REQUIRED_COLUMNS["en_en"], "EN-EN (P3/P4)"),
        (csv_en_ur, _REQUIRED_COLUMNS["en_ur"], "EN-UR (P5/P6)"),
    ]

    all_ok = True
    validated: list[tuple[Path, str]] = []  # (path, arcname)

    for path, req_cols, label in checks:
        ok, msg, count = _validate_csv(path, req_cols)
        status = "✓" if ok else "✗"
        print(f"  {status} {label:20s}  rows={count:<6d}  {msg}")
        if not ok:
            all_ok = False
        else:
            validated.append((path, path.name))

    if not all_ok:
        print()
        print("  ERROR: Fix the issues above before packaging.")
        print("  Tip: run  python scripts/eval_all.py  to generate the CSVs.")
        print("═" * 72 + "\n")
        sys.exit(1)

    # ── Build ZIP ─────────────────────────────────────────────────────────────
    # Challenge spec: files must sit at the TOP LEVEL of the archive.
    # Equivalent to: cd <dir>; zip submission.zip *.csv
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in validated:
            zf.write(path, arcname=arcname)

    # ── Summary ───────────────────────────────────────────────────────────────
    zip_size_kb = zip_path.stat().st_size / 1024
    print()
    print(f"  ZIP created: {zip_path}")
    print(f"  ZIP size   : {zip_size_kb:.1f} KB")
    print()
    print("  Archive contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"    {info.filename}  ({info.file_size / 1024:.1f} KB)")
    print()
    print("  Pre-flight checklist:")
    print("    ✓ Both CSV files present")
    print("    ✓ Required columns verified (key,p3,p4 and key,p5,p6)")
    print("    ✓ All prediction values are valid integers")
    print("    ✓ Files are at ZIP root (no subdirectory)")
    print()
    print("  Upload submission.zip to CodaBench:")
    print("    https://www.codabench.org/competitions/11283")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
