"""Deterministic relevance filtering, evidence grading (source quality, novelty) and ranking.

Scores only ORDER findings and the governance queue. They never change an impact tier or skip
a human gate. Every score records the rule id and its written basis (config/ranking.yaml).
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import yaml

DEFAULT_RANKING_PATH = Path("./config/ranking.yaml")


class RankingConfigError(ValueError):
    """Raised when ranking.yaml is missing or malformed."""


@dataclass(frozen=True)
class Rule:
    id: str
    score: int
    basis: str


@dataclass(frozen=True)
class RankingRules:
    version: str
    relevance: Dict[str, Rule]
    urgency: Dict[str, Rule]
    evidence_level_cap: Dict[str, int]
    default_tier: int
    novelty_multipliers: Dict[str, float]
    duplicate_similarity: float
    weights: Dict[str, float]
    filters: Dict[str, str]

    def relevance_rule(self, rule_id: str) -> Rule:
        return self.relevance[rule_id]

    def urgency_rule(self, rule_id: str) -> Rule:
        return self.urgency[rule_id]


def _rules(section: dict, name: str) -> Dict[str, Rule]:
    out = {}
    for rule_id, spec in (section or {}).items():
        try:
            score = int(spec["score"])
        except (KeyError, TypeError, ValueError) as e:
            raise RankingConfigError(f"{name}.{rule_id}: integer 'score' required") from e
        if not 1 <= score <= 5:
            raise RankingConfigError(f"{name}.{rule_id}: score must be 1-5")
        out[rule_id] = Rule(rule_id, score, str(spec.get("basis") or ""))
    if not out:
        raise RankingConfigError(f"ranking.yaml defines no {name} rules")
    return out


def load_ranking_rules(path: Union[str, Path] = DEFAULT_RANKING_PATH) -> RankingRules:
    path = Path(path)
    if not path.exists():
        raise RankingConfigError(f"Ranking rules not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise RankingConfigError(f"ranking.yaml is not valid YAML: {e}") from e
    weights = {k: float(v) for k, v in (data.get("priority_weights") or {}).items()}
    if set(weights) != {"urgency", "relevance", "source_quality"} or abs(sum(weights.values()) - 1.0) > 1e-6:
        raise RankingConfigError("priority_weights must define urgency, relevance and source_quality summing to 1.0")
    novelty = data.get("novelty") or {}
    multipliers = {k: float(v) for k, v in (novelty.get("multipliers") or {}).items()}
    if set(multipliers) != {"new", "revision_of_known", "duplicate_of_existing"}:
        raise RankingConfigError("novelty.multipliers must define new, revision_of_known, duplicate_of_existing")
    sq = data.get("source_quality") or {}
    return RankingRules(
        version=str(data.get("version") or "1.0"),
        relevance=_rules(data.get("relevance"), "relevance"),
        urgency=_rules(data.get("urgency"), "urgency"),
        evidence_level_cap={k.upper(): int(v) for k, v in (sq.get("evidence_level_cap") or {}).items()},
        default_tier=int(sq.get("default_tier") or 2),
        novelty_multipliers=multipliers,
        duplicate_similarity=float(novelty.get("duplicate_similarity") or 0.92),
        weights=weights,
        filters={k: str((v or {}).get("basis") or "") for k, v in (data.get("filters") or {}).items()},
    )


@lru_cache(maxsize=4)
def get_ranking_rules(path: str = str(DEFAULT_RANKING_PATH)) -> RankingRules:
    return load_ranking_rules(path)


# ------------------------------------------------------------------------------ scores
def source_quality(rules: RankingRules, quality_tier: Optional[int], evidence_level: Optional[str]) -> Tuple[int, str]:
    tier = int(quality_tier) if quality_tier else rules.default_tier
    cap = rules.evidence_level_cap.get((evidence_level or "").upper(), 5)
    score = max(1, min(tier, cap))
    basis = f"watchlist quality tier {tier}" + (f", evidence level {evidence_level.upper()} (cap {cap})" if evidence_level else "")
    return score, basis


def priority(rules: RankingRules, urgency: int, relevance: int, quality: int, novelty: str) -> float:
    w = rules.weights
    base = (w["urgency"] * urgency + w["relevance"] * relevance + w["source_quality"] * quality) / 5.0 * 100.0
    return round(base * rules.novelty_multipliers.get(novelty, 1.0), 1)


# Query-mode urgency by finding kind/relation
def finding_urgency_rule(kind: str, relation: str, change_category: Optional[str]) -> str:
    if kind == "contraindication":
        if relation == "applies":
            return "URG-01"
        if relation == "check_applicability":
            return "URG-02"
        return "URG-05"
    if kind == "safety_warning":
        return "URG-02"
    if kind == "withdrawal":
        return "URG-02"
    if kind == "evidence":
        return "URG-06"
    if change_category in ("dose_change", "threshold_change"):
        return "URG-03"
    if relation == "differs_from_plan":
        return "URG-04"
    return "URG-05"


# Background (surveillance) urgency by change category
CATEGORY_URGENCY_RULE = {
    "contraindication_added": "URG-01",
    "safety_warning": "URG-01",
    "withdrawn": "URG-02",
    "dose_change": "URG-03",
    "threshold_change": "URG-03",
    "new_recommendation": "URG-04",
    "revised_recommendation": "URG-04",
    "contraindication_removed": "URG-05",
    "new_evidence": "URG-06",
    "no_practice_change": "URG-06",
}
