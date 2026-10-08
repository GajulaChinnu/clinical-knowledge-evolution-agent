import pytest
import time
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.entities import Base, IngestedDocument
from app.services.config_service import AppConfig
from app.agents.monitoring_agent import MonitoringAgent
from app.services.evaluation_corpus import make_multipage_pdf_bytes
from app.services.source_version_diff import SourceVersionDiffService, ChangedSection
from app.agents.extraction_agent import ExtractionAgent
from app.schemas.extraction import ExtractionResponse, ExtractedRecommendation

@pytest.fixture
def isolated_env(tmp_path: Path):
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    cfg = AppConfig(
        database_url=db_url,
        chroma_dir=str(tmp_path / "chroma"),
        source_dir=tmp_path / "sources"
    )
    cfg.source_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "config": cfg,
        "session_factory": sessionmaker(bind=engine, autoflush=False)
    }

def test_generic_url_and_pdf_flow_lineage(isolated_env):
    """
    Test generic source version lineage: v1 -> v2 -> v3, plus idempotency.
    Proves:
    - PDF flow is fully supported and preserved.
    - Idempotency skips unchanged files.
    - Version increments properly (1.0 -> 2.0 -> 3.0) and chains via previous_source_version_id.
    """
    cfg = isolated_env["config"]
    session_factory = isolated_env["session_factory"]
    
    url = "https://example.test/guideline"
    stable_source_id = f"url_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}"
    
    m_agent = MonitoringAgent(source_dir=cfg.source_dir, session_factory=session_factory, config=cfg)
    
    # ------------------
    # VERSION 1
    # ------------------
    v1_content = make_multipage_pdf_bytes([["Section 1", "Recommendation: Continue intervention X."]])
    v1_path = cfg.source_dir / f"test_lineage_{int(time.time())}_v1.pdf"
    v1_path.write_bytes(v1_content)
    
    res1 = m_agent.ingest_known_file(v1_path, stable_source_id, {"source_url": url})
    doc1_id = res1.ingested_document_ids[0]
    
    # ------------------
    # VERSION 2
    # ------------------
    v2_content = make_multipage_pdf_bytes([["Section 1", "Recommendation: Continue intervention Y."]])
    v2_path = cfg.source_dir / f"test_lineage_{int(time.time())}_v2.pdf"
    v2_path.write_bytes(v2_content)
    
    res2 = m_agent.ingest_known_file(v2_path, stable_source_id, {"source_url": url})
    doc2_id = res2.ingested_document_ids[0]
    
    # ------------------
    # VERSION 3
    # ------------------
    v3_content = make_multipage_pdf_bytes([["Section 1", "Recommendation: Continue intervention Z."]])
    v3_path = cfg.source_dir / f"test_lineage_{int(time.time())}_v3.pdf"
    v3_path.write_bytes(v3_content)
    
    res3 = m_agent.ingest_known_file(v3_path, stable_source_id, {"source_url": url})
    doc3_id = res3.ingested_document_ids[0]
    
    # ------------------
    # IDEMPOTENCY CHECK
    # ------------------
    res4 = m_agent.ingest_known_file(v3_path, stable_source_id, {"source_url": url})
    assert res4.skipped == 1
    
    # ------------------
    # VERIFY LINEAGE
    # ------------------
    with session_factory() as session:
        d1 = session.get(IngestedDocument, doc1_id)
        d2 = session.get(IngestedDocument, doc2_id)
        d3 = session.get(IngestedDocument, doc3_id)
        
        assert d1.source_version == "1.0"
        assert d1.previous_source_version_id is None
        
        assert d2.source_version == "2.0"
        assert d2.previous_source_version_id == d1.id
        
        assert d3.source_version == "3.0"
        assert d3.previous_source_version_id == d2.id

def test_pipeline_handoff_receives_only_diff(isolated_env):
    """
    Test that the source diff correctly feeds the pipeline and proves:
    1. SourceVersionDiffService diffs v1 and v2.
    2. ExtractionAgent only receives changed content (diff) for extraction.
    3. Old content isn't extracted again.
    """
    cfg = isolated_env["config"]
    session_factory = isolated_env["session_factory"]
    
    # 1. Setup mock LLM
    mock_llm = MagicMock()
    mock_llm.extract_recommendations.return_value = ExtractionResponse(
        recommendations=[]
    )
    
    # 2. Ingest two versions
    url = "https://example.test/diff_test"
    stable_source_id = f"url_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}"
    
    m_agent = MonitoringAgent(source_dir=cfg.source_dir, session_factory=session_factory, config=cfg)
    
    v1_content = make_multipage_pdf_bytes([["Section 1", "Recommendation X"]])
    v1_path = cfg.source_dir / "v1.pdf"
    v1_path.write_bytes(v1_content)
    m_agent.ingest_known_file(v1_path, stable_source_id, {})
    
    v2_content = make_multipage_pdf_bytes([["Section 1", "Recommendation Y"]])
    v2_path = cfg.source_dir / "v2.pdf"
    v2_path.write_bytes(v2_content)
    res2 = m_agent.ingest_known_file(v2_path, stable_source_id, {})
    
    doc2_id = res2.ingested_document_ids[0]
    
    # 3. Process v2 via ExtractionAgent
    extractor = ExtractionAgent(llm_client=mock_llm, config=cfg, session_factory=session_factory)
    
    # Spy on the diff service
    with patch.object(SourceVersionDiffService, 'diff_sections', wraps=SourceVersionDiffService().diff_sections) as spy_diff:
        extractor.process_document(doc2_id)
        
        # Assert SourceVersionDiffService diffed v1 vs v2
        assert spy_diff.call_count == 1
        
        # Assert LLM was called with the 'diffed_sections', which only contains 'Recommendation Y'
        call_kwargs = mock_llm.extract_recommendations.call_args[1]
        assert "Recommendation Y" in call_kwargs["section_text"]
        assert "Recommendation X" not in call_kwargs["section_text"]
