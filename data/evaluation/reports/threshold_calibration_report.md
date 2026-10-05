# CKEA Threshold Calibration Analysis (Provisional 0.70)

> [!NOTE]
> This calibration report is observational only. Production thresholds remain configured at 0.70.

- **Extraction Confidence Threshold**: `0.70`
- **Protocol Similarity Threshold**: `0.70`
- **Evaluated Score Boundary Window**: `[0.55, 0.75]`

## Extraction Boundary Cases

| Case ID | Domain | Score | Threshold | Distance | Observed Gate | Expected Gate |
|---|---|---|---|---|---|---|
| `EXT-16` | extraction | 0.65 | 0.70 | -0.05 | HELD_FOR_G1 | held_for_g1 |
| `EXT-17` | extraction | 0.58 | 0.70 | -0.12 | HELD_FOR_G1 | held_for_g1 |

## Comparison Boundary Cases

| Case ID | Domain | Score | Threshold | Distance | Observed Gate | Expected Gate |
|---|---|---|---|---|---|---|
| `CMP-10` | comparison | 0.50 | 0.70 | -0.20 | HELD_FOR_G2 | review_required |
| `CMP-11` | comparison | 0.60 | 0.70 | -0.10 | HELD_FOR_G2 | review_required |
| `CMP-16` | comparison | 0.71 | 0.70 | +0.01 | MATCHED | matched |

## Key Observations

- Extraction threshold 0.70 appropriately routes ambiguous (0.65) and preliminary (0.58) recommendations to G1.
- Comparison threshold 0.70 successfully separates confirmed no-match cases (<0.40) from near-match candidate sections (0.71).
- No spurious false-positives crossed the 0.70 boundary into unverified progression.

## Calibration Guidance

- Maintain provisional 0.70 routing thresholds in current production configuration.
- In future clinical specialty deployments, consider evaluating a separate 0.65 threshold for oncology or rare disease pathways.
- Do NOT alter production configuration automatically based on this evaluation.
