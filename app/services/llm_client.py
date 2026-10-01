"""Shared LLM client for Groq API operations with credential protection and token observability."""

import logging
import time
from typing import Any, Dict, Optional
from openai import OpenAI

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

    def __repr__(self) -> str:
        return f"SharedLLMClient(model='{self.model}', base_url='{self.base_url}')"
