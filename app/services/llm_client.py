"""Shared LLM client for Groq API operations with credential protection and token observability."""

import logging
import time
from typing import Any, Dict, Optional
from openai import OpenAI

from app.schemas.comparison import ComparisonResponse
from app.schemas.extraction import ExtractionResponse
from app.services.config_service import AppConfig, MissingAPIKeyError, load_config

logger = logging.getLogger("ckea.services.llm_client")

EXTRACTION_SYSTEM_PROMPT = """You are an expert clinical guideline extraction assistant.
Analyze the provided clinical guideline section text and extract actionable clinical recommendations.
Return a structured JSON object strictly conforming to the schema.

Extraction Rules:
1. 'verbatim_text' must be the exact word-for-word sentence(s) directly from the section text. Do NOT edit, summarize, or paraphrase.
2. 'source_excerpt' must be the exact verbatim paragraph or immediate context surrounding the recommendation.
3. 'recommendation_type' must describe the nature of the recommendation (e.g., treatment, diagnostic, monitoring, dosing, lifestyle).
4. 'target_population' specifies the clinical patient group or condition.
5. 'intervention' specifies the specific drug, action, test, or strategy recommended.
6. 'evidence_grade' must be the grade explicitly cited (e.g. 'Grade A', 'Level 1a') or null if none is stated in the text. Do NOT invent a grade.
7. 'page' must be the page number provided in the prompt.
8. 'section' must be the section heading provided in the prompt.
9. 'confidence' must be a confidence score between 0.0 and 1.0 reflecting extraction certainty.
"""

COMPARISON_SYSTEM_PROMPT = """You are an expert clinical protocol comparison assistant.
Compare an extracted clinical guideline recommendation against a retrieved institutional protocol section.
Determine if there is a clinical gap, conflict, dosage change, threshold change, scope change, or no material difference.
Return a structured JSON object strictly conforming to the schema.

Comparison Rules:
1. 'comparison_result' must be 'gap' if there is a clinical difference/conflict/addition, 'no_gap' if the recommendation aligns with the protocol without material difference, 'ambiguous' if unclear, or 'no_match' if the section is clinically unrelated.
2. 'matched_protocol_section' must be the section heading or title provided in the candidate context.
3. 'protocol_id' must match the protocol ID provided in the prompt.
4. 'protocol_version' must match the protocol version provided in the prompt.
5. 'section_id' must match the section ID provided in the prompt.
6. 'exact_protocol_text' must be the exact verbatim excerpt directly quoted from the candidate protocol section text. Do NOT edit or invent text.
7. 'specific_difference' must be a concise, objective summary of the textual difference between the recommendation and protocol.
8. 'difference_type' must be one of: 'threshold_change', 'population_expansion', 'population_restriction', 'intervention_change', 'contraindication', 'monitoring_change', 'frequency_change', 'dosage_change', 'conflict', 'new_recommendation', 'scope_expansion', 'no_material_difference', 'no_match', 'other_supported_change', 'none'.
9. If 'comparison_result' is 'no_gap', 'difference_type' must be 'none' or 'no_material_difference'.
10. 'confidence' must be a score between 0.0 and 1.0 reflecting comparison certainty.
11. 'rationale' must objectively explain the clinical alignment or difference based strictly on the provided texts.
"""


class SharedLLMClient:
    """Shared client for executing structured inferences against Groq models."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        base_url: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.config = config or load_config()
        self.base_url = base_url or self.config.groq_base_url
        self.model = self.config.groq_model
        self._client = client
        self.request_count: int = 0
        self.last_usage: Optional[Dict[str, Any]] = None

    @property
    def client(self) -> OpenAI:
        """Lazily initialize the underlying OpenAI client configured for Groq."""
        if self._client is None:
            if not self.config.has_api_key:
                raise MissingAPIKeyError(
                    "Cannot execute LLM inference: GROQ_API_KEY is not configured."
                )
            self._client = OpenAI(
                api_key=self.config.get_api_key(),
                base_url=self.base_url,
            )
        return self._client

    def extract_recommendations(
        self,
        section_heading: str,
        section_text: str,
        page_number: int,
        document_identifier: str = "guideline",
    ) -> ExtractionResponse:
        """Call Groq to extract structured recommendations from a candidate section.

        Args:
            section_heading: Heading of the candidate section.
            section_text: Ground truth text of the candidate section.
            page_number: Page number where section appears.
            document_identifier: Ground truth identifier of the source document.

        Returns:
            ExtractionResponse containing discrete ExtractedRecommendation objects.
        """
        user_prompt = (
            f"Document: {document_identifier}\n"
            f"Page: {page_number}\n"
            f"Section: {section_heading}\n\n"
            f"--- SECTION TEXT START ---\n"
            f"{section_text}\n"
            f"--- SECTION TEXT END ---\n\n"
            "Extract all discrete clinical recommendations present in the section above."
        )

        start_time = time.perf_counter()
        try:
            # Use Groq structured JSON schema output supported by openai/gpt-oss-20b
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ExtractionResponse",
                        "strict": True,
                        "schema": ExtractionResponse.model_json_schema(),
                    },
                },
                temperature=0.0,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            self.request_count += 1

            # Observability: record usage metadata safely
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else None
            total_tokens = usage.total_tokens if usage else None

            self.last_usage = {
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "request_count": self.request_count,
                "latency_ms": latency_ms,
            }

            logger.info(
                "Groq extraction completed: model=%s, prompt_tokens=%s, completion_tokens=%s, latency_ms=%.1f",
                self.model,
                prompt_tokens,
                completion_tokens,
                latency_ms,
            )

            # Application-side Pydantic validation (never trust provider output blindly)
            raw_content = completion.choices[0].message.content
            if not raw_content:
                raise ValueError("Model returned empty or null content.")

            response = ExtractionResponse.model_validate_json(raw_content)
            return response

        except Exception as e:
            logger.error("Groq extraction failed: %s: %s", type(e).__name__, str(e))
            raise

    def compare_recommendation_to_protocol(
        self,
        recommendation_text: str,
        target_population: str,
        intervention: str,
        candidate_section_heading: str,
        candidate_section_text: str,
        candidate_protocol_id: str,
        candidate_protocol_version: str,
        candidate_section_id: str,
        evidence_grade: Optional[str] = None,
    ) -> ComparisonResponse:
        """Call Groq to compare an extracted recommendation against a candidate protocol section.

        Args:
            recommendation_text: Verbatim clinical recommendation text.
            target_population: Targeted patient demographic or clinical group.
            intervention: Recommended clinical intervention, drug, or action.
            candidate_section_heading: Heading of the candidate protocol section.
            candidate_section_text: Full text of the candidate protocol section.
            candidate_protocol_id: Protocol ID (e.g. PROT-DM-001).
            candidate_protocol_version: Immutable protocol version (e.g. v1.0).
            candidate_section_id: Candidate section identifier (e.g. SEC-3).
            evidence_grade: Optional clinical evidence grade.

        Returns:
            ComparisonResponse containing validated comparison conclusion.
        """
        user_prompt = (
            f"--- EXTRACTED RECOMMENDATION ---\n"
            f"Recommendation: {recommendation_text}\n"
            f"Target Population: {target_population}\n"
            f"Intervention: {intervention}\n"
            f"Evidence Grade: {evidence_grade or 'None stated'}\n\n"
            f"--- CANDIDATE PROTOCOL SECTION ---\n"
            f"Protocol ID: {candidate_protocol_id}\n"
            f"Protocol Version: {candidate_protocol_version}\n"
            f"Section ID: {candidate_section_id}\n"
            f"Section Heading: {candidate_section_heading}\n"
            f"Section Text:\n{candidate_section_text}\n\n"
            "Compare the extracted recommendation to the candidate protocol section and output structured comparison JSON."
        )

        start_time = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ComparisonResponse",
                        "strict": True,
                        "schema": ComparisonResponse.model_json_schema(),
                    },
                },
                temperature=0.0,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            self.request_count += 1

            usage = completion.usage
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else None
            total_tokens = usage.total_tokens if usage else None

            self.last_usage = {
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "request_count": self.request_count,
                "latency_ms": latency_ms,
            }

            logger.info(
                "Groq comparison completed: model=%s, prompt_tokens=%s, completion_tokens=%s, latency_ms=%.1f",
                self.model,
                prompt_tokens,
                completion_tokens,
                latency_ms,
            )

            raw_content = completion.choices[0].message.content
            if not raw_content:
                raise ValueError("Model returned empty or null comparison content.")

            response = ComparisonResponse.model_validate_json(raw_content)
            return response

        except Exception as e:
            logger.error("Groq comparison failed: %s: %s", type(e).__name__, str(e))
            raise

    def __repr__(self) -> str:
        return f"SharedLLMClient(model='{self.model}', base_url='{self.base_url}')"
