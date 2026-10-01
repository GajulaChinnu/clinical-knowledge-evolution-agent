"""Unit tests for protocol indexing, ChromaDB persistent storage, and semantic retrieval."""

import json
from pathlib import Path
import pytest

from app.schemas.protocol import (
    CandidateProtocolSection,
    ProtocolDocument,
    ProtocolIndexingSummary,
    ProtocolSection,
)
from app.services.config_service import AppConfig
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.protocol_index import (
    EmptyProtocolError,
    ProtocolError,
    ProtocolIndexService,
    ProtocolParseError,
    parse_protocol_file,
)


@pytest.fixture
def temp_protocol_index(tmp_path: Path):
    """Provide a ProtocolIndexService isolated to a temporary ChromaDB directory."""
    chroma_dir = tmp_path / "temp_chroma"
    config = AppConfig(
        chroma_dir=chroma_dir,
        protocol_dir=tmp_path / "protocols",
    )
    service = ProtocolIndexService(chroma_dir=chroma_dir, config=config)
    return {
        "service": service,
        "chroma_dir": chroma_dir,
        "protocol_dir": tmp_path / "protocols",
        "tmp_path": tmp_path,
    }


def test_protocol_discovery(temp_protocol_index):
    """Test 1: Verify protocol file discovery in configured directory."""
    protocol_dir = temp_protocol_index["protocol_dir"]
    protocol_dir.mkdir(parents=True, exist_ok=True)

    # Valid protocol files
    (protocol_dir / "p1.json").write_text(
        json.dumps({
            "protocol_id": "P1",
            "protocol_version": "v1.0",
            "sections": [{"section_id": "S1", "section_heading": "H1", "section_text": "T1"}]
        }),
        encoding="utf-8"
    )
    (protocol_dir / "p2.md").write_text(
        "# Protocol: P2\nVersion: v1.0\n## S1: H1\nT1",
        encoding="utf-8"
    )

    # Ignored files (hidden or unsupported extension)
    (protocol_dir / ".hidden.json").write_text("{}", encoding="utf-8")
    (protocol_dir / "unsupported.pdf").write_bytes(b"%PDF-1.4 dummy")

    service = temp_protocol_index["service"]
    summary = service.index_directory(protocol_dir=protocol_dir)

    assert summary.protocols_discovered == 2
    assert summary.sections_indexed == 2
    assert summary.protocols_failed == 0


def test_protocol_parsing_json(tmp_path: Path):
    """Test 2: Verify local deterministic parsing of structured JSON protocols."""
    file_path = tmp_path / "test_protocol.json"
    content = {
        "protocol_id": "PROT-CARDIO-01",
        "protocol_version": "v2.1",
        "title": "Hypertension Protocol",
        "sections": [
            {
                "section_id": "SEC-HTN-1",
                "section_heading": "Blood Pressure Targets",
                "section_text": "Target systolic BP < 130 mmHg for high-risk adults."
            },
            {
                "section_id": "SEC-HTN-2",
                "section_heading": "First-line Agents",
                "section_text": "Thiazide diuretics, CCBs, or ACE inhibitors/ARBs."
            }
        ]
    }
    file_path.write_text(json.dumps(content), encoding="utf-8")

    doc = parse_protocol_file(file_path)

    assert doc.protocol_id == "PROT-CARDIO-01"
    assert doc.protocol_version == "v2.1"
    assert doc.title == "Hypertension Protocol"
    assert len(doc.sections) == 2
    assert doc.sections[0].section_id == "SEC-HTN-1"
    assert doc.sections[0].section_heading == "Blood Pressure Targets"
    assert doc.sections[1].section_id == "SEC-HTN-2"


def test_section_detection_markdown(tmp_path: Path):
    """Test 3: Verify local deterministic section detection from Markdown protocols."""
    file_path = tmp_path / "PROT-RESP_v1.0.md"
    md_content = """# Asthma Management Guideline
Protocol_ID: PROT-RESP
Protocol_Version: v1.0
Title: Asthma Protocol

## Section 1: Inhaled Corticosteroids
Low-dose ICS should be initiated as controller therapy for persistent asthma.

## Section 2: Reliever Therapy
Short-acting beta2-agonists (SABA) should be prescribed for acute symptom relief.
"""
    file_path.write_text(md_content, encoding="utf-8")

    doc = parse_protocol_file(file_path)

    assert doc.protocol_id == "PROT-RESP"
    assert doc.protocol_version == "v1.0"
    assert len(doc.sections) == 2
    assert doc.sections[0].section_id == "1"
    assert doc.sections[0].section_heading == "Inhaled Corticosteroids"
    assert "Low-dose ICS" in doc.sections[0].section_text
    assert doc.sections[1].section_id == "2"
    assert doc.sections[1].section_heading == "Reliever Therapy"


def test_embedding_generation():
    """Test 4: Verify local all-MiniLM-L6-v2 embedding generation."""
    embedding_service = get_embedding_service()

    # Single embedding
    vector = embedding_service.embed_text("Target HbA1c < 7.0% for type 2 diabetes.")
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)

    # Empty text produces deterministic zero vector
    empty_vec = embedding_service.embed_text("")
    assert len(empty_vec) == 384
    assert sum(abs(x) for x in empty_vec) == 0.0

    # Batch embedding
    texts = ["First recommendation.", "Second recommendation."]
    vectors = embedding_service.embed_batch(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_chromadb_initialization(temp_protocol_index):
    """Test 5: Verify ChromaDB local embedded persistent client and collection initialization."""
    service = temp_protocol_index["service"]
    chroma_dir = temp_protocol_index["chroma_dir"]

    assert chroma_dir.exists()
    assert service.collection is not None
    assert service.collection.name == "protocol_sections"
    assert service.collection.count() == 0


def test_section_indexing(temp_protocol_index):
    """Test 6: Verify section indexing persists into ChromaDB."""
    service = temp_protocol_index["service"]

    doc = ProtocolDocument(
        protocol_id="PROT-DM",
        protocol_version="v1.0",
        sections=[
            ProtocolSection(
                protocol_id="PROT-DM",
                protocol_version="v1.0",
                section_id="SEC-1",
                section_heading="Target HbA1c",
                section_text="Target HbA1c < 7.0% for non-pregnant adults.",
            ),
            ProtocolSection(
                protocol_id="PROT-DM",
                protocol_version="v1.0",
                section_id="SEC-2",
                section_heading="Metformin Dosing",
                section_text="Start metformin 500mg daily with evening meals.",
            ),
        ]
    )

    indexed_count = service.index_protocol(doc)
    assert indexed_count == 2
    assert service.collection.count() == 2


def test_metadata_correctness(temp_protocol_index):
    """Test 7: Verify metadata attached to ChromaDB items is complete and accurate."""
    service = temp_protocol_index["service"]

    doc = ProtocolDocument(
        protocol_id="PROT-RENAL",
        protocol_version="v3.0",
        sections=[
            ProtocolSection(
                protocol_id="PROT-RENAL",
                protocol_version="v3.0",
                section_id="SEC-CKD-4",
                section_heading="Metformin Renal Cutoff",
                section_text="Discontinue metformin when eGFR < 30 mL/min/1.73m2.",
            )
        ]
    )
    service.index_protocol(doc)

    expected_id = "PROT-RENAL__v3.0__SEC-CKD-4"
    result = service.collection.get(ids=[expected_id], include=["metadatas", "documents"])

    assert len(result["ids"]) == 1
    assert result["ids"][0] == expected_id
    meta = result["metadatas"][0]
    assert meta["protocol_id"] == "PROT-RENAL"
    assert meta["protocol_version"] == "v3.0"
    assert meta["section_id"] == "SEC-CKD-4"
    assert meta["section_heading"] == "Metformin Renal Cutoff"
    assert meta["section_text"] == "Discontinue metformin when eGFR < 30 mL/min/1.73m2."


def test_stable_document_ids():
    """Test 8: Verify deterministic stable document IDs derived from protocol, version, and section ID."""
    section1 = ProtocolSection(
        protocol_id="PROT-001",
        protocol_version="v1.0",
        section_id="SEC-1",
        section_heading="Heading",
        section_text="Text",
    )
    section2 = ProtocolSection(
        protocol_id="PROT-001",
        protocol_version="v1.0",
        section_id="SEC-1",
        section_heading="Heading Changed",
        section_text="Text Changed",
    )

    # Identical immutable identifiers yield identical stable IDs
    assert section1.document_id == "PROT-001__v1.0__SEC-1"
    assert section2.document_id == "PROT-001__v1.0__SEC-1"
    assert section1.document_id == section2.document_id


def test_top_k_retrieval(temp_protocol_index):
    """Test 9: Verify top-k semantic retrieval accurately surfaces most relevant sections."""
    service = temp_protocol_index["service"]

    doc = ProtocolDocument(
        protocol_id="PROT-MULTI",
        protocol_version="v1.0",
        sections=[
            ProtocolSection(
                protocol_id="PROT-MULTI",
                protocol_version="v1.0",
                section_id="DM-1",
                section_heading="Glycemic Control in T2D",
                section_text="Target HbA1c below 7.0% for adult patients with type 2 diabetes.",
            ),
            ProtocolSection(
                protocol_id="PROT-MULTI",
                protocol_version="v1.0",
                section_id="RESP-1",
                section_heading="Asthma Inhaler Therapy",
                section_text="Inhaled albuterol for acute bronchospasm relief.",
            ),
            ProtocolSection(
                protocol_id="PROT-MULTI",
                protocol_version="v1.0",
                section_id="CARD-1",
                section_heading="Lipid Management",
                section_text="High-intensity statin therapy for patients with clinical ASCVD.",
            ),
        ]
    )
    service.index_protocol(doc)

    recommendation = "Patients diagnosed with type 2 diabetes should achieve an HbA1c target < 7%."
    candidates = service.retrieve(recommendation_text=recommendation, top_k=3)

    assert len(candidates) == 3
    # Top match must be the diabetes glycemic section
    top_match = candidates[0]
    assert top_match.section_id == "DM-1"
    assert top_match.protocol_id == "PROT-MULTI"
    assert top_match.similarity > candidates[1].similarity
    assert top_match.similarity > candidates[2].similarity
    assert top_match.similarity >= 0.70


def test_top_k_exact_limit(temp_protocol_index):
    """Test 10: Verify top-k parameter defaults to 3 and strictly respects requested count."""
    service = temp_protocol_index["service"]

    sections = [
        ProtocolSection(
            protocol_id="PROT-LIMIT",
            protocol_version="v1.0",
            section_id=f"SEC-{i}",
            section_heading=f"Heading {i}",
            section_text=f"Clinical protocol section content number {i} with distinct text.",
        )
        for i in range(1, 6)
    ]
    doc = ProtocolDocument(protocol_id="PROT-LIMIT", protocol_version="v1.0", sections=sections)
    service.index_protocol(doc)

    # Default top_k = 3
    candidates_default = service.retrieve("Clinical protocol section content")
    assert len(candidates_default) == 3

    # Explicit top_k = 2
    candidates_2 = service.retrieve("Clinical protocol section content", top_k=2)
    assert len(candidates_2) == 2

    # Explicit top_k = 5
    candidates_5 = service.retrieve("Clinical protocol section content", top_k=5)
    assert len(candidates_5) == 5


def test_repeated_indexing_idempotency(temp_protocol_index):
    """Test 11: Verify repeated indexing of unchanged protocol does not create duplicate ChromaDB documents."""
    service = temp_protocol_index["service"]

    doc = ProtocolDocument(
        protocol_id="PROT-IDEM",
        protocol_version="v1.0",
        sections=[
            ProtocolSection(
                protocol_id="PROT-IDEM",
                protocol_version="v1.0",
                section_id="S1",
                section_heading="H1",
                section_text="T1",
            ),
            ProtocolSection(
                protocol_id="PROT-IDEM",
                protocol_version="v1.0",
                section_id="S2",
                section_heading="H2",
                section_text="T2",
            ),
        ]
    )

    # First indexing run
    indexed_1 = service.index_protocol(doc)
    assert indexed_1 == 2
    assert service.collection.count() == 2

    # Second indexing run (identical content)
    indexed_2 = service.index_protocol(doc)
    assert indexed_2 == 0  # 0 new sections
    assert service.collection.count() == 2  # Total unchanged, no duplicates


def test_version_separation(temp_protocol_index):
    """Test 12: Verify different versions of the same protocol exist as separate indexed versions."""
    service = temp_protocol_index["service"]

    doc_v1 = ProtocolDocument(
        protocol_id="PROT-VER",
        protocol_version="v1.0",
        sections=[
            ProtocolSection(
                protocol_id="PROT-VER",
                protocol_version="v1.0",
                section_id="SEC-1",
                section_heading="Glycemic Target",
                section_text="Target HbA1c < 7.5% for all patients.",
            )
        ]
    )

    doc_v2 = ProtocolDocument(
        protocol_id="PROT-VER",
        protocol_version="v2.0",
        sections=[
            ProtocolSection(
                protocol_id="PROT-VER",
                protocol_version="v2.0",
                section_id="SEC-1",
                section_heading="Glycemic Target",
                section_text="Target HbA1c < 7.0% for non-pregnant adult patients.",
            )
        ]
    )

    service.index_protocol(doc_v1)
    service.index_protocol(doc_v2)

    assert service.collection.count() == 2

    # Filtered retrieval by version
    v1_results = service.retrieve("HbA1c target", top_k=5, protocol_version="v1.0")
    assert len(v1_results) == 1
    assert v1_results[0].protocol_version == "v1.0"
    assert "< 7.5%" in v1_results[0].section_text

    v2_results = service.retrieve("HbA1c target", top_k=5, protocol_version="v2.0")
    assert len(v2_results) == 1
    assert v2_results[0].protocol_version == "v2.0"
    assert "< 7.0%" in v2_results[0].section_text


def test_historical_version_preservation(temp_protocol_index):
    """Test 13: Verify historical protocol version v1.0 is preserved and unchanged when v2.0 is added."""
    service = temp_protocol_index["service"]

    v1_sec = ProtocolSection(
        protocol_id="PROT-HIST",
        protocol_version="v1.0",
        section_id="S1",
        section_heading="Old Guideline",
        section_text="Old therapy was sulfonylurea first-line.",
    )
    service.index_protocol(ProtocolDocument(protocol_id="PROT-HIST", protocol_version="v1.0", sections=[v1_sec]))

    v2_sec = ProtocolSection(
        protocol_id="PROT-HIST",
        protocol_version="v2.0",
        section_id="S1",
        section_heading="New Guideline",
        section_text="New therapy is SGLT2 inhibitor first-line.",
    )
    service.index_protocol(ProtocolDocument(protocol_id="PROT-HIST", protocol_version="v2.0", sections=[v2_sec]))

    # Verify v1.0 document is completely intact
    v1_record = service.collection.get(ids=["PROT-HIST__v1.0__S1"])
    assert len(v1_record["ids"]) == 1
    assert v1_record["metadatas"][0]["section_text"] == "Old therapy was sulfonylurea first-line."


def test_empty_protocol_handling(tmp_path: Path):
    """Test 14: Verify empty protocol files raise EmptyProtocolError and are handled gracefully by service."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("   \n  ", encoding="utf-8")

    with pytest.raises(EmptyProtocolError):
        parse_protocol_file(empty_file)

    empty_sections_file = tmp_path / "empty_sections.json"
    empty_sections_file.write_text(json.dumps({"protocol_id": "P", "protocol_version": "1", "sections": []}), encoding="utf-8")

    with pytest.raises(EmptyProtocolError):
        parse_protocol_file(empty_sections_file)

    # Directory indexing captures failure without crashing
    service = ProtocolIndexService(chroma_dir=tmp_path / "chroma")
    summary = service.index_directory(protocol_dir=tmp_path)
    assert summary.protocols_failed >= 2
    assert len(summary.errors) >= 2


def test_invalid_protocol_handling(tmp_path: Path):
    """Test 15: Verify malformed or incomplete protocols raise ProtocolParseError and are recorded in summary."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ incomplete json ...", encoding="utf-8")

    with pytest.raises(ProtocolParseError):
        parse_protocol_file(bad_json)

    missing_fields = tmp_path / "missing_id.json"
    missing_fields.write_text(json.dumps({"protocol_version": "v1.0", "sections": [{"section_id": "1"}]}), encoding="utf-8")

    with pytest.raises(ProtocolParseError, match="Missing or empty 'protocol_id'"):
        parse_protocol_file(missing_fields)
