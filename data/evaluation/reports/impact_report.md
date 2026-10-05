# Evaluation Report: Impact

- **Total Cases**: 6
- **Passed**: 6
- **Expected Holds**: 0
- **Failed**: 0
- **Errors**: 0

## Domain Metrics

- **rule_version**: `1.0`
- **immutability_enforced**: `True`

## Detailed Case Results

| Case ID | Outcome | Expected | Actual | Provenance | Notes |
|---|---|---|---|---|---|
| `IMP-01` | `PASS` | Tier=ImpactTier.CRITICAL, SLA_hours=48 | Tier=ImpactTier.CRITICAL, SLA_hours=48 | Valid | Score=14, Tier=Critical, SLA=48h |
| `IMP-02` | `PASS` | Tier=ImpactTier.HIGH, SLA_hours=168 | Tier=ImpactTier.HIGH, SLA_hours=168 | Valid | Score=9, Tier=High, SLA=168h |
| `IMP-03` | `PASS` | Tier=ImpactTier.STANDARD, SLA_hours=720 | Tier=ImpactTier.STANDARD, SLA_hours=720 | Valid | Score=6, Tier=Standard, SLA=720h |
| `IMP-04` | `PASS` | Tier=ImpactTier.LOW, SLA_hours=2160 | Tier=ImpactTier.LOW, SLA_hours=2160 | Valid | Score=3, Tier=Low, SLA=2160h |
| `IMP-05` | `PASS` | Tier=None, SLA_hours=None | Tier=None, SLA_hours=None | Valid | Incomplete score correctly un-tiered: complete=False |
| `IMP-06` | `PASS` | TierDowngradeBlockedError | Blocked | Valid | Automatic tier downgrade was successfully blocked |
