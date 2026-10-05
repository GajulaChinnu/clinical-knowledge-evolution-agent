# Clinical Change Brief: {{ brief_id }}

**Status:** {{ status|default("draft")|upper }}  
**Generated:** {{ generated_at }}  
**Schema Version:** v{{ schema_version }}  
**Source Document:** {{ source_metadata.source_identifier }} (v{{ source_metadata.source_version }})  
**Assigned Governance Tier:** {% if impact_assessment.is_complete and impact_assessment.tier %}**{{ impact_assessment.tier }}** (Score: {{ impact_assessment.total_score }}/15){% else %}**INCOMPLETE**{% endif %}  
**Target Committee:** {{ impact_assessment.routing_target|default("Manual Governance Resolution Required") }}  
**SLA Deadline:** {{ impact_assessment.sla_deadline|default("Not Assigned") }}  

---

## 1. What Changed

> "{{ what_changed.recommendation_text }}"

- **Source Identifier:** {{ source_metadata.source_identifier }}
- **Location:** Page {{ what_changed.page|default("N/A") }}, Section: {{ what_changed.section|default("N/A") }}
- **Recommendation Type:** {{ what_changed.recommendation_type }}
- **Target Population:** {{ what_changed.target_population }}
- **Intervention:** {{ what_changed.intervention }}
- **Evidence Grade:** {{ what_changed.evidence_grade|default("Not specified") }}

---

## 2. What Our Protocol Currently Says

{% if current_protocol.is_match %}
> "{{ current_protocol.exact_protocol_text }}"

- **Protocol ID:** {{ current_protocol.protocol_id }}
- **Protocol Version:** v{{ current_protocol.protocol_version }}
- **Protocol Section ID:** {{ current_protocol.section_id|default("N/A") }}
- **Section Heading:** {{ current_protocol.section_heading|default("N/A") }}
{% else %}
> [!WARNING]
> **No Matching Protocol Section Identified**  
> {{ current_protocol.no_match_statement|default("No matching protocol section was identified in the current institutional protocol library.") }}

- **Protocol Status:** Unindexed / No Match (Requires G3 Governance Review)
{% endif %}

---

## 3. Specific Difference

**{{ specific_difference.specific_difference }}**

- **Difference Type:** `{{ specific_difference.difference_type }}`
- **Comparison Result:** {{ specific_difference.comparison_result }}
- **Comparison Confidence:** {% if specific_difference.comparison_confidence is not none %}{{ "%.0f"|format(specific_difference.comparison_confidence * 100) }}%{% else %}N/A{% endif %}

---

## 4. Impact Assessment

{% if impact_assessment.is_complete %}
| Dimension | Rule ID | Score | Written Score Basis |
| :--- | :--- | :---: | :--- |
| **Clinical Urgency** | `{{ impact_assessment.rule_ids.get('clinical_urgency', 'URGENCY') }}` | **{{ impact_assessment.clinical_urgency }} / 5** | {{ impact_assessment.urgency_basis }} |
| **Evidence Strength** | `{{ impact_assessment.rule_ids.get('evidence_strength', 'EVIDENCE') }}` | **{{ impact_assessment.evidence_strength }} / 5** | {{ impact_assessment.evidence_basis }} |
| **Pathway Breadth** | `{{ impact_assessment.rule_ids.get('pathway_breadth', 'BREADTH') }}` | **{{ impact_assessment.pathway_breadth }} / 5** | {{ impact_assessment.breadth_basis }} |

- **Total Score:** **{{ impact_assessment.total_score }} / 15**
- **Assigned Tier:** **{{ impact_assessment.tier }}**
- **Routing Target:** {{ impact_assessment.routing_target }}
- **SLA Deadline:** {{ impact_assessment.sla_deadline }}
- **Scoring Rules Version:** v{{ impact_assessment.scoring_yaml_version }}
{% else %}
> [!WARNING]
> **Impact Assessment Incomplete**  
> {{ impact_assessment.incomplete_reason|default("Missing or unmapped dimension values prevent deterministic scoring. No tier or SLA has been assigned.") }}
{% endif %}

---

## 5. Affected Workflows

{% if affected_workflows.is_available and affected_workflows.affected_workflows %}
{% if affected_workflows.workflow_summary %}
{{ affected_workflows.workflow_summary }}

{% endif %}
{% for wf in affected_workflows.affected_workflows %}
- {{ wf }}
{% endfor %}
{% elif affected_workflows.is_available and affected_workflows.workflow_summary %}
{{ affected_workflows.workflow_summary }}
{% else %}
*{{ affected_workflows.unavailability_reason|default("Workflow information unavailable in source metadata.") }}*
{% endif %}

---

## 6. Proposed Review Actions

> [!NOTE]
> **Governance Notice:** These are proposed review actions for clinical governance. They are NOT decisions. The system does not automatically approve or reject protocol modifications.

{% for action in proposed_actions %}
{{ action.step }}. **{{ action.action }}** — {{ action.description }}
{% endfor %}

---

## 7. Source Excerpt

> {{ source_excerpt.source_excerpt }}

*Verified source quotation from **{{ source_excerpt.source_identifier }}** (Page {{ source_excerpt.page|default("N/A") }}, Section: {{ source_excerpt.section|default("N/A") }}).*

---
*CKEA Clinical Knowledge Evolution Agent — Deterministic Briefing Engine (Impact ID: {{ impact_record_id }}, Gap ID: {{ gap_record_id }})*
