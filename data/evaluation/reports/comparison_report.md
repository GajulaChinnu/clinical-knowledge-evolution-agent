# Evaluation Report: Comparison

- **Total Cases**: 16
- **Passed**: 11
- **Expected Holds**: 5
- **Failed**: 0
- **Errors**: 0

## Domain Metrics

- **total_scenarios**: `16`
- **matched_count**: `11`
- **no_match_count**: `2`
- **g2_holds**: `3`
- **g3_holds**: `2`
- **difference_type_accuracy**: `1.0000`
- **precision**: `0.7500`
- **recall**: `0.7500`
- **f1_score**: `0.7500`
- **confidence_mean**: `0.8225`

## Detailed Case Results

| Case ID | Outcome | Expected | Actual | Provenance | Notes |
|---|---|---|---|---|---|
| `CMP-01` | `PASS` | no_gap (none) | no_gap (none) | Valid | Comparison confirmed: result='no_gap', diff='none' |
| `CMP-02` | `PASS` | no_gap (no_material_difference) | no_gap (no_material_difference) | Valid | Comparison confirmed: result='no_gap', diff='no_material_difference' |
| `CMP-03` | `PASS` | gap (dosage_change) | gap (dosage_change) | Valid | Comparison confirmed: result='gap', diff='dosage_change' |
| `CMP-04` | `PASS` | gap (frequency_change) | gap (frequency_change) | Valid | Comparison confirmed: result='gap', diff='frequency_change' |
| `CMP-05` | `PASS` | gap (population_restriction) | gap (population_restriction) | Valid | Comparison confirmed: result='gap', diff='population_restriction' |
| `CMP-06` | `PASS` | gap (contraindication) | gap (contraindication) | Valid | Comparison confirmed: result='gap', diff='contraindication' |
| `CMP-07` | `PASS` | gap (other_supported_change) | gap (other_supported_change) | Valid | Comparison confirmed: result='gap', diff='other_supported_change' |
| `CMP-08` | `PASS` | no_gap (none) | no_gap (none) | Valid | Comparison confirmed: result='no_gap', diff='none' |
| `CMP-09` | `PASS` | gap (dosage_change) | gap (dosage_change) | Valid | Comparison confirmed: result='gap', diff='dosage_change' |
| `CMP-10` | `HELD_EXPECTED` | ambiguous (none) | ambiguous (none) | Valid | Comparison confirmed: result='ambiguous', diff='none' |
| `CMP-11` | `HELD_EXPECTED` | no_gap (dosage_change) | no_gap (dosage_change) | Valid | Comparison confirmed: result='no_gap', diff='dosage_change' |
| `CMP-12` | `HELD_EXPECTED` | ambiguous (none) | ambiguous (none) | Valid | Comparison confirmed: result='ambiguous', diff='none' |
| `CMP-13` | `HELD_EXPECTED` | no_match (no_match) | no_match (no_match) | Valid | Comparison confirmed: result='no_match', diff='no_match' |
| `CMP-14` | `HELD_EXPECTED` | no_match (no_match) | no_match (no_match) | Valid | Comparison confirmed: result='no_match', diff='no_match' |
| `CMP-15` | `PASS` | gap (other_supported_change) | gap (other_supported_change) | Valid | Comparison confirmed: result='gap', diff='other_supported_change' |
| `CMP-16` | `PASS` | gap (threshold_change) | gap (threshold_change) | Valid | Comparison confirmed: result='gap', diff='threshold_change' |
