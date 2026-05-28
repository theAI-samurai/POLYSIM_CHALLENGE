# POLY-SIM 2026 Challenge Guide

## What this challenge is about
POLY-SIM 2026 (Polyglot Speaker Identification with Missing Modality) focuses on robust speaker identification when:
- training uses paired face + voice data in English,
- testing may have missing face modality (audio-only),
- testing may involve a different language (Urdu).

The goal is to build a model that remains accurate under both modality missingness and cross-lingual shift.

## Core challenge to solve
Train with multimodal English data and predict speaker identity under multiple evaluation settings, especially hard settings where face is unavailable and language changes.

## Dataset context
- Dataset base: MAV-Celeb (English-Urdu subset).
- Training examples are organized by modality, identity, language, and sample.
- Official challenge also uses progress/dev and evaluation/test CSVs.

Stats reported in the challenge document:
- English: videos 262 / 70 / 70, samples 4039 / 1290 / 1521 (train/val/test)
- Urdu: videos 415 / 70 / 70, samples 9304 / 1779 / 1623 (train/val/test)

## Protocols (evaluation setups)
The challenge reports four protocols:
- P3: In-language multimodal (train/test same language, both modalities available)
- P4: Missing-modality (test with audio only, face missing)
- P5: Cross-lingual multimodal (train on one language, test on another, both modalities)
- P6: Cross-lingual missing-modality (cross-lingual test with missing face)

## Metric
Primary metric is P-accuracy.

Reported scores per protocol:
- P3Acc
- P4Acc
- P5Acc
- P6Acc

Overall score:

$$
\text{Overall Score} = \frac{\text{Acc}(P3) + \text{Acc}(P4) + \text{Acc}(P5) + \text{Acc}(P6)}{4}
$$

## Submission format
Submissions are made on CodaBench and must be uploaded as a ZIP containing CSV files.

Rules:
- Put CSV files directly inside the ZIP (do not zip the parent folder).
- Create ZIP from within the CSV directory with: `zip submission.zip *.csv`
- One CSV per language pair.

File naming:
- submission_v1_<phase>_English_English.csv
- submission_v1_<phase>_English_Urdu.csv

Where `<phase>` is:
- val (progress/dev)
- test (evaluation)

CSV schema by setup:
- Monolingual file (lang1 == lang2): columns `key,p3,p4`
- Cross-lingual file (lang1 != lang2): columns `key,p5,p6`

`key` is the unique test-pair ID.
`p3/p4/p5/p6` are predicted identity indices among P candidates.

Example rows (from challenge doc style):
- key,p3,p4
- t5M7dziYVY,1,0
- RmUYdg2luC,50,0

## Submission limits
- Progress phase: maximum 150 total submissions, maximum 15 per day.
- Evaluation phase: maximum 15 total submissions.

## Rules you must satisfy
- You must submit a system description.
- You must submit a link to a working codebase (for example GitHub).
- Missing system description or missing code link can disqualify a team.

## Timeline (as announced in challenge plan)
- Registration: 27 Mar 2026 - 10 May 2026
- Progress phase: 27 Mar 2026 - 15 May 2026
- Evaluation phase: 16 May 2026 - 23 May 2026
- Challenge results: 25 May 2026
- Final paper: 8 Jun 2026

## Official links
- Challenge repository: https://github.com/msaadsaeed/polysim
- CodaBench competition: https://www.codabench.org/competitions/11283
- Registration form: https://forms.gle/EwmVBiph2QsZ2QRB9

## Important practical notes for this workspace
- This workspace currently contains training data and training CSV files under `data_train/`.
- If you only have training split locally, use it for model development and internal validation setup.
- Official benchmark scores require generating prediction CSVs for the challenge-provided progress/evaluation keys and uploading them to CodaBench.

## Recommended solution workflow
- Build strong unimodal audio baseline first (critical for missing-face protocols).
- Add multimodal training objectives to improve representation quality and identity discrimination.
- Add robustness strategies for cross-lingual shift (normalization, domain adaptation, language-robust embeddings).
- Validate output CSV generation early to avoid submission-format failures.
- Track protocol-wise performance (P3 to P6), not only aggregate score.
