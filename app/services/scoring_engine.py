"""Deterministic impact scoring engine executing versioned YAML rules.

Calculates Clinical Urgency (1-5), Evidence Strength (1-5), and Pathway Breadth (1-5),
determining total score (3-15), routing tier, and SLA without any LLM calls.
"""

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from app.schemas.impact import DimensionScore, ImpactStatus, ImpactTier, ScoringResult
from app.services.config_service import AppConfig, load_config

logger = logging.getLogger("ckea.services.scoring_engine")


class ScoringError(Exception):
    """Base exception for scoring engine errors."""
    pass


class InvalidRuleConfigError(ScoringError):
    """Raised when scoring YAML configuration is invalid or missing required sections."""
    pass


class TierDowngradeBlockedError(ScoringError):
    """Raised when an attempt is made to automatically downgrade an assigned impact tier."""
    pass


class UnresolvedComparisonError(ScoringError):
    """Raised when an unresolved or invalid comparison finding is passed for impact scoring."""
    pass


TIER_SEVERITY_ORDER: Dict[str, int] = {
    ImpactTier.CRITICAL.value: 4,
    ImpactTier.HIGH.value: 3,
    ImpactTier.STANDARD.value: 2,
    ImpactTier.LOW.value: 1,
}


class ScoringEngine:
    """Deterministic, auditable scoring engine driven by versioned YAML rules."""

    def __init__(
        self,
        yaml_path: Optional[str | Path] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or load_config()
        self.yaml_path = Path(yaml_path) if yaml_path else self.config.scoring_rules_path
        self.version = "1.0"
        self.dimensions: Dict[str, Dict[str, Any]] = {}
        self.tiers: Dict[str, Dict[str, Any]] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        """Load and parse the YAML rule file."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Scoring rules file not found: {self.yaml_path}")

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise InvalidRuleConfigError(f"Failed to parse scoring YAML '{self.yaml_path}': {e}") from e

        if not isinstance(data, dict):
            raise InvalidRuleConfigError("Scoring rules YAML root must be a dictionary.")

        self.version = str(data.get("version", "1.0")).strip()
        dimensions = data.get("dimensions")
        if not isinstance(dimensions, dict):
            raise InvalidRuleConfigError("Missing or invalid 'dimensions' section in scoring YAML.")

        for dim_key in ("clinical_urgency", "evidence_strength", "pathway_breadth"):
            if dim_key not in dimensions or not isinstance(dimensions[dim_key], dict):
                raise InvalidRuleConfigError(f"Missing required dimension '{dim_key}' in scoring YAML.")
            rules = dimensions[dim_key].get("rules")
            if not isinstance(rules, list) or len(rules) == 0:
                raise InvalidRuleConfigError(f"Dimension '{dim_key}' must have non-empty 'rules' list.")

        self.dimensions = dimensions
        self.tiers = data.get("tiers", {})

    def resolve_dimension(
        self,
        dimension_name: str,
        input_value: Optional[Any],
    ) -> Optional[DimensionScore]:
        """Match an input value against rules for a specific dimension.

        Args:
            dimension_name: One of 'clinical_urgency', 'evidence_strength', 'pathway_breadth'.
            input_value: Categorical input or integer/string score.

        Returns:
            DimensionScore if matched; None if missing or unmapped.
        """
        if input_value is None:
            return None

        # Clean string representation
        norm_val = str(input_value).strip().lower()
        if not norm_val:
            return None

        dim_config = self.dimensions.get(dimension_name)
        if not dim_config:
            raise InvalidRuleConfigError(f"Unknown dimension '{dimension_name}'")

        rules: List[Dict[str, Any]] = dim_config.get("rules", [])
        for rule in rules:
            rule_id = rule.get("id")
            score = rule.get("score")
            basis = rule.get("basis")
            criteria: List[str] = rule.get("criteria", [])

            # Check direct numeric score match (e.g. input_value=4 or input_value="4")
            if norm_val == str(score):
                return DimensionScore(
                    dimension=dimension_name,
                    score=score,
                    rule_id=rule_id,
                    rule_version=self.version,
                    basis=basis,
                )

            # Check criteria matches (case-insensitive substring/equality)
            for c in criteria:
                norm_c = str(c).strip().lower()
                if norm_val == norm_c:
                    return DimensionScore(
                        dimension=dimension_name,
                        score=score,
                        rule_id=rule_id,
                        rule_version=self.version,
                        basis=basis,
                    )

        # No match found in YAML
        logger.debug("Dimension '%s' input '%s' did not match any YAML rule.", dimension_name, input_value)
        return None

    def score_urgency(self, value: Optional[Any]) -> Optional[DimensionScore]:
        """Resolve Clinical Urgency (1-5)."""
        return self.resolve_dimension("clinical_urgency", value)

    def score_evidence(self, value: Optional[Any]) -> Optional[DimensionScore]:
        """Resolve Evidence Strength (1-5)."""
        return self.resolve_dimension("evidence_strength", value)

    def score_breadth(self, value: Optional[Any]) -> Optional[DimensionScore]:
        """Resolve Pathway Breadth (1-5)."""
        return self.resolve_dimension("pathway_breadth", value)

    def calculate_tier_and_sla(
        self,
        total_score: int,
        base_time: Optional[datetime] = None,
    ) -> Tuple[ImpactTier, int, datetime, str]:
        """Determine routing tier, SLA hours, deadline, and target committee from total score.

        Boundaries (AGENTS.md):
        12–15 → Critical → 48 hours
        8–11  → High → 7 days (168 hours)
        4–7   → Standard → 30 days (720 hours)
        3     → Low → 90 days (2160 hours)
        """
        now = base_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        if 12 <= total_score <= 15:
            tier = ImpactTier.CRITICAL
            sla_hours = 48
            target = "Rapid Governance Committee"
        elif 8 <= total_score <= 11:
            tier = ImpactTier.HIGH
            sla_hours = 168
            target = "Clinical Governance Board"
        elif 4 <= total_score <= 7:
            tier = ImpactTier.STANDARD
            sla_hours = 720
            target = "Protocol Review Committee"
        elif total_score == 3:
            tier = ImpactTier.LOW
            sla_hours = 2160
            target = "Routine Maintenance Review"
        else:
            raise ValueError(f"Total score {total_score} outside valid range [3, 15].")

        deadline = now + timedelta(hours=sla_hours)
        return tier, sla_hours, deadline, target

    def calculate_impact(
        self,
        urgency_input: Optional[Any] = None,
        evidence_input: Optional[Any] = None,
        breadth_input: Optional[Any] = None,
        base_time: Optional[datetime] = None,
    ) -> ScoringResult:
        """Deterministically calculate multidimensional impact and routing.

        If any dimension cannot be resolved, status is set to INCOMPLETE and
        no total score, tier, or SLA is assigned.
        """
        urgency_score = self.score_urgency(urgency_input)
        evidence_score = self.score_evidence(evidence_input)
        breadth_score = self.score_breadth(breadth_input)

        dimension_scores: Dict[str, DimensionScore] = {}
        rule_ids: Dict[str, str] = {}

        if urgency_score:
            dimension_scores["clinical_urgency"] = urgency_score
            rule_ids["clinical_urgency"] = urgency_score.rule_id
        if evidence_score:
            dimension_scores["evidence_strength"] = evidence_score
            rule_ids["evidence_strength"] = evidence_score.rule_id
        if breadth_score:
            dimension_scores["pathway_breadth"] = breadth_score
            rule_ids["pathway_breadth"] = breadth_score.rule_id

        # Check completeness
        is_complete = bool(urgency_score and evidence_score and breadth_score)

        if not is_complete:
            reasons = []
            if urgency_score is None:
                if urgency_input is None or not str(urgency_input).strip():
                    reasons.append("Missing clinical urgency classification")
                else:
                    reasons.append(f"Unsupported clinical urgency value '{urgency_input}'")
            if evidence_score is None:
                if evidence_input is None or not str(evidence_input).strip():
                    reasons.append("Missing evidence grade")
                else:
                    reasons.append(f"Unsupported evidence grade '{evidence_input}'")
            if breadth_score is None:
                if breadth_input is None or not str(breadth_input).strip():
                    reasons.append("Missing pathway breadth classification")
                else:
                    reasons.append(f"Unsupported pathway breadth value '{breadth_input}'")

            return ScoringResult(
                clinical_urgency=urgency_score.score if urgency_score else None,
                evidence_strength=evidence_score.score if evidence_score else None,
                pathway_breadth=breadth_score.score if breadth_score else None,
                rule_ids=rule_ids,
                scoring_yaml_version=self.version,
                total_score=None,
                tier=None,
                routing_target=None,
                sla_deadline=None,
                sla_hours=None,
                urgency_basis=urgency_score.basis if urgency_score else None,
                evidence_basis=evidence_score.basis if evidence_score else None,
                breadth_basis=breadth_score.basis if breadth_score else None,
                status=ImpactStatus.INCOMPLETE,
                incomplete_reason="; ".join(reasons),
                dimension_scores=dimension_scores,
            )

        # All 3 dimensions resolved
        assert urgency_score is not None
        assert evidence_score is not None
        assert breadth_score is not None

        total = urgency_score.score + evidence_score.score + breadth_score.score
        tier, sla_hours, deadline, target = self.calculate_tier_and_sla(total, base_time=base_time)

        return ScoringResult(
            clinical_urgency=urgency_score.score,
            evidence_strength=evidence_score.score,
            pathway_breadth=breadth_score.score,
            rule_ids=rule_ids,
            scoring_yaml_version=self.version,
            total_score=total,
            tier=tier,
            routing_target=target,
            sla_deadline=deadline,
            sla_hours=sla_hours,
            urgency_basis=urgency_score.basis,
            evidence_basis=evidence_score.basis,
            breadth_basis=breadth_score.basis,
            status=ImpactStatus.CALCULATED,
            incomplete_reason=None,
            dimension_scores=dimension_scores,
        )
