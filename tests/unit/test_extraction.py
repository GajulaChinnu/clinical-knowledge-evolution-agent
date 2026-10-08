"""Unit tests for CKEA Extraction Agent, section detection, verification, and persistence."""

from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock
from pydantic import ValidationError
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.extraction_agent import ExtractionAgent
from app.models.database import get_engine, get_session_factory, init_db
from app.models.entities import ChangeRecord, IngestedDocument
from app.schemas.changes import ChangeStatus
from app.schemas.documents import DocumentStatus
from app.schemas.extraction import ExtractedRecommendation, ExtractionResponse
from app.services.config_service import AppConfig, MissingAPIKeyError
from app.services.llm_client import SharedLLMClient
from app.services.pdf_parser import (
    DocumentSection,
    PageText,
    detect_document_sections,
    extract_page_texts,
    filter_candidate_sections,
)
from app.services.source_verifier import verify_recommendation_provenance


def make_pdf_bytes(lines: List[str]) -> bytes:
    """Generate a valid single-page PDF with extractable lines of text."""
    stream_lines = ["BT", "/F1 12 Tf", "14 TL", "50 750 Td"]
    for line in lines:
        clean = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"({clean}) '")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("latin1", "replace")
    stream_len = len(stream)

    pdf = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {stream_len} >> stream
{stream.decode('latin1')}
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000320 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
390
%%EOF""".encode("latin1")
    return pdf


@pytest.fixture
def extraction_env(tmp_path: Path):
    """Set up temporary database, config, and PDF files for extraction tests."""
    db_path = tmp_path / "test_extraction.db"
    engine = get_engine(db_url=f"sqlite:///{db_path}")
    init_db(engine=engine)
    session_factory = get_session_factory(engine=engine)

    config = AppConfig(
        database_url=f"sqlite:///{db_path}",
        extraction_confidence_threshold=0.70,
        groq_model="openai/gpt-oss-20b",
    )

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "config": config,
        "tmp_path": tmp_path,
    }

    engine.dispose()


def test_section_extraction():
    """Test 1: Verify deterministic section detection identifies headings and page numbers."""
    pages = [
        PageText(
            page_number=1,
            text=(
                "CLINICAL GUIDELINE OVERVIEW\n"
                "This document covers management of chronic conditions.\n\n"
                "1. Clinical Recommendations\n"
                "Adult patients with T2D should be offered Metformin.\n"
                "Target HbA1c is recommended to be < 7.0%.\n"
            ),
        )
    ]

    sections = detect_document_sections(pages)
    assert len(sections) >= 2

    headings = [s.section_heading for s in sections]
    assert "CLINICAL GUIDELINE OVERVIEW" in headings
    assert "1. Clinical Recommendations" in headings

    rec_sec = next(s for s in sections if s.section_heading == "1. Clinical Recommendations")
    assert rec_sec.page_number == 1
    assert "Metformin" in rec_sec.text


def test_candidate_section_selection():
    """Test 2: Verify candidate selection filters down to sections with recommendation terms."""
    sections = [
        DocumentSection(
            page_number=1,
            section_heading="Background and Epidemiology",
            text="Diabetes is a metabolic disease with high global prevalence.",
        ),
        DocumentSection(
            page_number=2,
            section_heading="Treatment Strategy",
            text="Metformin should be initiated at diagnosis unless contraindicated.",
        ),
    ]

    candidates = filter_candidate_sections(sections)
    assert len(candidates) == 1
    assert candidates[0].section_heading == "Treatment Strategy"
    assert candidates[0].candidate is True


def test_valid_structured_extraction(extraction_env):
    """Test 3: Verify valid extraction creates a ChangeRecord in 'extracted' status."""
    tmp_path = extraction_env["tmp_path"]
    session_factory = extraction_env["session_factory"]
    config = extraction_env["config"]

    # 1. Create source PDF
    pdf_path = tmp_path / "valid_guideline.pdf"
    content_lines = [
        "1. Pharmacological Management",
        "Adults with type 2 diabetes should be started on metformin immediately upon diagnosis.",
    ]
    pdf_path.write_bytes(make_pdf_bytes(content_lines))

    # 2. Persist IngestedDocument in DB
    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="ADA-T2D",
            source_path=str(pdf_path),
            sha256_hash="e" * 64,
            source_version="1.0",
            pipeline_version="1.0",
            status=DocumentStatus.PARSED.value,
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    # 3. Mock LLM client
    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.extract_recommendations.return_value = ExtractionResponse(
        recommendations=[
            ExtractedRecommendation(
                verbatim_text="Adults with type 2 diabetes should be started on metformin immediately upon diagnosis.",
                recommendation_type="treatment",
                target_population="Adults with type 2 diabetes",
                intervention="Metformin initiation",
                evidence_grade="Grade A",
                page=1,
                section="1. Pharmacological Management",
                source_excerpt="Adults with type 2 diabetes should be started on metformin immediately upon diagnosis.",
                confidence=0.95,
            )
        ]
    )

    agent = ExtractionAgent(
        session_factory=session_factory,
        llm_client=mock_llm,
        config=config,
    )

    changes = agent.process_document(doc_id)

    assert len(changes) == 1
    change = changes[0]
    assert change.status == ChangeStatus.EXTRACTED.value
    assert change.confidence == 0.95
    assert change.evidence_grade == "Grade A"
    assert "metformin" in change.verbatim_text.lower()

    # Document should transition to COMPLETE
    with session_factory() as session:
        updated_doc = session.get(IngestedDocument, doc_id)
        assert updated_doc.status == DocumentStatus.COMPLETE.value


def test_missing_required_field_in_extraction():
    """Test 4: Verify Pydantic validation rejects recommendations missing required fields."""
    with pytest.raises(ValidationError):
        ExtractedRecommendation(
            verbatim_text="",  # Invalid: empty string
            recommendation_type="treatment",
            target_population="Adults",
            intervention="Drug X",
            page=1,
            section="Section 1",
            source_excerpt="Drug X is advised.",
            confidence=0.85,
        )


def test_invalid_structured_output_handling(extraction_env):
    """Test 5: Verify agent handles LLM extraction exceptions gracefully without crashing."""
    tmp_path = extraction_env["tmp_path"]
    session_factory = extraction_env["session_factory"]
    config = extraction_env["config"]

    pdf_path = tmp_path / "failing_doc.pdf"
    pdf_path.write_bytes(make_pdf_bytes(["1. Therapy Guidelines", "Patients should exercise daily."]))

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="FAIL-DOC",
            source_path=str(pdf_path),
            sha256_hash="f" * 64,
            status=DocumentStatus.PARSED.value,
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.extract_recommendations.side_effect = RuntimeError("Groq API rate limit exceeded")

    agent = ExtractionAgent(
        session_factory=session_factory,
        llm_client=mock_llm,
        config=config,
    )

    with pytest.raises(RuntimeError, match="Groq API rate limit"):
        agent.process_document(doc_id)


def test_source_quotation_verification_success():
    """Test 6: Verify source quotation verification succeeds when text strictly exists."""
    section_text = "Section 2. Dosing\nPatients should take 500mg daily with evening meals."
    verbatim = "Patients should take 500mg daily with evening meals."
    excerpt = "Section 2. Dosing\nPatients should take 500mg daily with evening meals."

    res = verify_recommendation_provenance(
        verbatim_text=verbatim,
        source_excerpt=excerpt,
        source_section_text=section_text,
        page=1,
        expected_page=1,
    )
    assert res.is_verified is True
    assert res.reason is None


def test_source_quotation_verification_failure_routes_to_g1(extraction_env):
    """Test 7: Verify hallucinated or altered quote fails verification and routes to held_for_G1."""
    tmp_path = extraction_env["tmp_path"]
    session_factory = extraction_env["session_factory"]
    config = extraction_env["config"]

    pdf_path = tmp_path / "hallucinated.pdf"
    pdf_path.write_bytes(
        make_pdf_bytes(["1. Recommendations", "Clinicians should consider diet modification first."])
    )

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="DIET-GUIDE",
            source_path=str(pdf_path),
            sha256_hash="1" * 64,
            status=DocumentStatus.PARSED.value,
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    # Model returns hallucinated/paraphrased quotation not matching the source
    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.extract_recommendations.return_value = ExtractionResponse(
        recommendations=[
            ExtractedRecommendation(
                verbatim_text="Clinicians must mandate keto diets immediately.",  # Hallucinated
                recommendation_type="lifestyle",
                target_population="All patients",
                intervention="Keto diet",
                page=1,
                section="1. Recommendations",
                source_excerpt="Clinicians must mandate keto diets immediately.",
                confidence=0.90,
            )
        ]
    )

    agent = ExtractionAgent(session_factory=session_factory, llm_client=mock_llm, config=config)
    changes = agent.process_document(doc_id)

    assert len(changes) == 1
    assert changes[0].status == ChangeStatus.HELD_FOR_G1.value

    # Document should transition to HELD
    with session_factory() as session:
        updated_doc = session.get(IngestedDocument, doc_id)
        assert updated_doc.status == DocumentStatus.HELD.value


def test_confidence_routing_thresholds(extraction_env):
    """Test 8 & 9: Verify confidence >= 0.70 is extracted, and < 0.70 is held_for_G1."""
    tmp_path = extraction_env["tmp_path"]
    session_factory = extraction_env["session_factory"]
    config = extraction_env["config"]

    pdf_path = tmp_path / "conf_test.pdf"
    pdf_path.write_bytes(
        make_pdf_bytes([
            "1. Dosing Guidelines",
            "High confidence recommendation: prescribe statins for cardiovascular risk.",
            "Low confidence recommendation: consider aspirin for low risk patients.",
        ])
    )

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="CONF-DOC",
            source_path=str(pdf_path),
            sha256_hash="2" * 64,
            status=DocumentStatus.PARSED.value,
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.extract_recommendations.return_value = ExtractionResponse(
        recommendations=[
            ExtractedRecommendation(
                verbatim_text="High confidence recommendation: prescribe statins for cardiovascular risk.",
                recommendation_type="treatment",
                target_population="Patients at cardiovascular risk",
                intervention="Statins",
                page=1,
                section="1. Dosing Guidelines",
                source_excerpt="High confidence recommendation: prescribe statins for cardiovascular risk.",
                confidence=0.85,  # >= 0.70 -> extracted
            ),
            ExtractedRecommendation(
                verbatim_text="Low confidence recommendation: consider aspirin for low risk patients.",
                recommendation_type="treatment",
                target_population="Low risk patients",
                intervention="Aspirin",
                page=1,
                section="1. Dosing Guidelines",
                source_excerpt="Low confidence recommendation: consider aspirin for low risk patients.",
                confidence=0.55,  # < 0.70 -> held_for_G1
            ),
        ]
    )

    agent = ExtractionAgent(session_factory=session_factory, llm_client=mock_llm, config=config)
    changes = agent.process_document(doc_id)

    assert len(changes) == 2
    c_high = next(c for c in changes if c.confidence == 0.85)
    c_low = next(c for c in changes if c.confidence == 0.55)

    assert c_high.status == ChangeStatus.EXTRACTED.value
    assert c_low.status == ChangeStatus.HELD_FOR_G1.value


def test_multiple_recommendations_and_parent_relationship(extraction_env):
    """Test 10, 11, 12, 13: Multiple recommendations, None evidence grade, and version persistence."""
    tmp_path = extraction_env["tmp_path"]
    session_factory = extraction_env["session_factory"]
    config = extraction_env["config"]

    pdf_path = tmp_path / "multi_rec.pdf"
    pdf_path.write_bytes(
        make_pdf_bytes([
            "1. Multi Therapy",
            "First recommendation: perform annual retinopathy screening.",
            "Second recommendation: monitor urinary albumin excretion.",
        ])
    )

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="MULTI-REC-DOC",
            source_path=str(pdf_path),
            sha256_hash="3" * 64,
            status=DocumentStatus.PARSED.value,
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.extract_recommendations.return_value = ExtractionResponse(
        recommendations=[
            ExtractedRecommendation(
                verbatim_text="First recommendation: perform annual retinopathy screening.",
                recommendation_type="monitoring",
                target_population="T2D patients",
                intervention="Retinopathy screening",
                evidence_grade=None,  # Missing evidence grade supported
                page=1,
                section="1. Multi Therapy",
                source_excerpt="First recommendation: perform annual retinopathy screening.",
                confidence=0.92,
            ),
            ExtractedRecommendation(
                verbatim_text="Second recommendation: monitor urinary albumin excretion.",
                recommendation_type="monitoring",
                target_population="T2D patients",
                intervention="Albumin excretion test",
                evidence_grade="Grade B",
                page=1,
                section="1. Multi Therapy",
                source_excerpt="Second recommendation: monitor urinary albumin excretion.",
                confidence=0.88,
            ),
        ]
    )

    agent = ExtractionAgent(
        session_factory=session_factory,
        llm_client=mock_llm,
        config=config,
        prompt_version="2.1.0",
    )
    changes = agent.process_document(doc_id)

    assert len(changes) == 2

    # Verify parent relationship
    with session_factory() as session:
        fetched_doc = session.get(IngestedDocument, doc_id)
        assert len(fetched_doc.change_records) == 2
        for ch in fetched_doc.change_records:
            assert ch.ingested_document_id == doc_id
            assert ch.extraction_model_version == "openai/gpt-oss-20b"
            assert ch.extraction_prompt_version == "2.1.0"

    rec1 = next(c for c in changes if c.evidence_grade is None)
    rec2 = next(c for c in changes if c.evidence_grade == "Grade B")
    assert rec1.evidence_grade is None
    assert rec2.evidence_grade == "Grade B"


def test_llm_receives_only_candidate_section_text_not_full_pdf(extraction_env):
    """Test 14 (TOKEN OPTIMIZATION): Verify LLM receives only candidate section, not full document."""
    tmp_path = extraction_env["tmp_path"]
    session_factory = extraction_env["session_factory"]
    config = extraction_env["config"]

    # PDF with 3 sections: Background (huge), Recommendations (candidate), Appendix (huge)
    background_text = "Background Information: " + ("history " * 40)
    rec_text = "Treatment: Patients should receive insulin therapy if oral agents fail."
    appendix_text = "Appendix Information: " + ("appendix " * 40)

    pdf_path = tmp_path / "token_opt.pdf"
    pdf_path.write_bytes(
        make_pdf_bytes([
            "Section 1. Background",
            background_text,
            "Section 2. Clinical Treatment",
            rec_text,
            "Section 3. Appendix",
            appendix_text,
        ])
    )

    with session_factory() as session:
        doc = IngestedDocument(
            source_identifier="TOKEN-OPT-DOC",
            source_path=str(pdf_path),
            sha256_hash="4" * 64,
            status=DocumentStatus.PARSED.value,
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    mock_llm = MagicMock(spec=SharedLLMClient)
    mock_llm.extract_recommendations.return_value = ExtractionResponse(
        recommendations=[
            ExtractedRecommendation(
                verbatim_text=rec_text,
                recommendation_type="treatment",
                target_population="T2D patients",
                intervention="Insulin therapy",
                page=1,
                section="Section 2. Clinical Treatment",
                source_excerpt=rec_text,
                confidence=0.95,
            )
        ]
    )

    agent = ExtractionAgent(session_factory=session_factory, llm_client=mock_llm, config=config)
    agent.process_document(doc_id)

    # Verify mock call arguments
    assert mock_llm.extract_recommendations.call_count == 1
    call_kwargs = mock_llm.extract_recommendations.call_args.kwargs

    sent_heading = call_kwargs["section_heading"]
    sent_text = call_kwargs["section_text"]

    # Candidate section was sent
    assert "Section 2. Clinical Treatment" in sent_heading
    assert "insulin therapy" in sent_text

    # Entire PDF text (Background / Appendix) was NOT sent to LLM
    assert "Background Information:" not in sent_text
    assert "Appendix Information:" not in sent_text
    assert "history history" not in sent_text
    assert "appendix appendix" not in sent_text


def test_shared_llm_client_missing_key():
    """Verify SharedLLMClient raises MissingAPIKeyError when GROQ_API_KEY is missing."""
    config = AppConfig(groq_api_key=None)
    client = SharedLLMClient(config=config)
    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY is not configured"):
        _ = client.client


def test_shared_llm_client_extract_recommendations_and_observability():
    """Verify SharedLLMClient invokes Groq API with json_schema and records observability metadata."""
    import json
    from openai import OpenAI

    mock_openai = MagicMock(spec=OpenAI)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    sample_json = json.dumps({
        "recommendations": [
            {
                "verbatim_text": "Patients should take 500mg daily.",
                "recommendation_type": "treatment",
                "target_population": "Adults with T2D",
                "intervention": "Metformin",
                "evidence_grade": "Grade A",
                "page": 1,
                "section": "Dosing",
                "source_excerpt": "Dosing: Patients should take 500mg daily.",
                "confidence": 0.95,
            }
        ]
    })
    mock_choice.message.content = sample_json
    mock_completion.choices = [mock_choice]
    mock_completion.usage = MagicMock(prompt_tokens=150, completion_tokens=80, total_tokens=230)
    mock_openai.chat.completions.create.return_value = mock_completion

    config = AppConfig(groq_api_key="mock-key-12345", groq_model="openai/gpt-oss-20b")
    client = SharedLLMClient(config=config, client=mock_openai)

    response = client.extract_recommendations(
        section_heading="Dosing",
        section_text="Dosing: Patients should take 500mg daily.",
        page_number=1,
        document_identifier="test-doc",
    )

    assert isinstance(response, ExtractionResponse)
    assert len(response.recommendations) == 1
    assert response.recommendations[0].verbatim_text == "Patients should take 500mg daily."

    # Verify structured output arguments passed to client
    mock_openai.chat.completions.create.assert_called_once()
    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-oss-20b"
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["temperature"] == 0.0

    schema = kwargs["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert "$defs" in schema
    assert "ExtractedRecommendation" in schema["$defs"]
    assert schema["$defs"]["ExtractedRecommendation"]["additionalProperties"] is False

    # Verify token usage observability
    assert client.request_count == 1
    assert client.last_usage is not None
    assert client.last_usage["model"] == "openai/gpt-oss-20b"
    assert client.last_usage["prompt_tokens"] == 150
    assert client.last_usage["completion_tokens"] == 80
    assert client.last_usage["total_tokens"] == 230
    assert client.last_usage["latency_ms"] >= 0.0


def test_extraction_response_schema_groq_strict_compliance():
    """Verify ExtractionResponse JSON schema conforms to Groq strict structured outputs rules:
    1. Root object has additionalProperties: False
    2. Every object under $defs has additionalProperties: False
    3. Every object has a 'required' list exactly covering all of its properties
    4. Conceptual optional fields (evidence_grade) are represented as nullable and required
    5. No default values remain inside property definitions that might bypass required
    """
    schema = ExtractionResponse.model_json_schema()

    # 1. Root object verification
    assert schema.get("type") == "object"
    assert schema.get("additionalProperties") is False
    assert "properties" in schema
    assert "recommendations" in schema["properties"]
    assert "required" in schema
    assert "recommendations" in schema["required"]
    assert set(schema["required"]) == set(schema["properties"].keys())

    # 2. $defs verification
    assert "$defs" in schema
    assert "ExtractedRecommendation" in schema["$defs"]

    rec_schema = schema["$defs"]["ExtractedRecommendation"]
    assert rec_schema.get("type") == "object"
    assert rec_schema.get("additionalProperties") is False

    # 3. Every property in ExtractedRecommendation must be in 'required'
    rec_props = rec_schema.get("properties", {})
    rec_required = rec_schema.get("required", [])
    expected_props = {
        "verbatim_text",
        "recommendation_type",
        "target_population",
        "intervention",
        "evidence_grade",
        "page",
        "section",
        "source_excerpt",
        "confidence",
    }
    assert set(rec_props.keys()) == expected_props
    assert set(rec_required) == expected_props

    # 4. evidence_grade is nullable (anyOf string or null) and required
    ev_schema = rec_props["evidence_grade"]
    assert "anyOf" in ev_schema
    types = [t.get("type") for t in ev_schema["anyOf"] if isinstance(t, dict)]
    assert "string" in types
    assert "null" in types
    assert "default" not in ev_schema

    # 5. Check all nested objects recursively if any
    for def_name, def_obj in schema["$defs"].items():
        if def_obj.get("type") == "object":
            assert def_obj.get("additionalProperties") is False
            assert "required" in def_obj
            assert set(def_obj["required"]) == set(def_obj.get("properties", {}).keys())


def test_extracted_recommendation_forbids_extra_fields():
    """Verify ExtractedRecommendation rejects arbitrary unknown fields due to extra='forbid'."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExtractedRecommendation(
            verbatim_text="Patients should take metformin.",
            recommendation_type="treatment",
            target_population="Adults",
            intervention="Metformin",
            evidence_grade="Grade A",
            page=1,
            section="Section 1",
            source_excerpt="Excerpt",
            confidence=0.9,
            unexpected_field="disallowed",
        )


def test_extraction_nullable_evidence_grade_behavior():
    """Verify ExtractedRecommendation semantics allow evidence_grade=None or explicit grade."""
    rec_none = ExtractedRecommendation(
        verbatim_text="Patients should exercise.",
        recommendation_type="lifestyle",
        target_population="Adults",
        intervention="Exercise",
        evidence_grade=None,
        page=2,
        section="Lifestyle",
        source_excerpt="Patients should exercise daily.",
        confidence=0.85,
    )
    assert rec_none.evidence_grade is None

    rec_omitted = ExtractedRecommendation(
        verbatim_text="Patients should exercise.",
        recommendation_type="lifestyle",
        target_population="Adults",
        intervention="Exercise",
        page=2,
        section="Lifestyle",
        source_excerpt="Patients should exercise daily.",
        confidence=0.85,
    )
    assert rec_omitted.evidence_grade is None

    rec_grade = ExtractedRecommendation(
        verbatim_text="Patients should exercise.",
        recommendation_type="lifestyle",
        target_population="Adults",
        intervention="Exercise",
        evidence_grade="Grade B",
        page=2,
        section="Lifestyle",
        source_excerpt="Patients should exercise daily.",
        confidence=0.85,
    )
    assert rec_grade.evidence_grade == "Grade B"


