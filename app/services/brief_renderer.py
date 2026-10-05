"""Deterministic clinical change brief rendering service using Jinja2 templates.

Renders HTML, Markdown, and JSON companion outputs from validated StructuredBriefPayloads.
Calculates rendered file hashes for auditable persistence.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.schemas.briefs import (
    BriefCompletenessError,
    StructuredBriefPayload,
    assert_brief_completeness,
)
from app.services.config_service import AppConfig, load_config
from app.services.file_hash import compute_sha256

logger = logging.getLogger("ckea.services.brief_renderer")


@dataclass(frozen=True)
class RenderedBriefResult:
    """Immutable result containing rendered brief artifacts, file paths, and digests."""
    brief_id: str
    html_content: str
    markdown_content: str
    json_content: str
    html_path: Path
    markdown_path: Path
    json_path: Path
    rendered_file_hash: str
    primary_file_path: Path


class BriefRenderer:
    """Deterministic Jinja2-based renderer for clinical change briefs."""

    def __init__(
        self,
        templates_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or load_config()
        self.templates_dir = Path(templates_dir) if templates_dir else self.config.templates_dir
        self.output_dir = Path(output_dir) if output_dir else self.config.output_dir

        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_brief(
        self,
        payload: StructuredBriefPayload,
        write_files: bool = True,
        validate: bool = True,
    ) -> RenderedBriefResult:
        """Render HTML, Markdown, and JSON artifacts from a StructuredBriefPayload.

        Args:
            payload: Validated structured payload for the seven brief sections.
            write_files: Whether to persist files under output_dir (default: True).
            validate: Whether to run completeness validation before rendering (default: True).

        Returns:
            RenderedBriefResult with rendered strings, file paths, and file hash.

        Raises:
            BriefCompletenessError: If validation fails and validate=True.
        """
        if validate:
            assert_brief_completeness(payload)

        # Prepare context data dictionary from Pydantic model
        context = payload.model_dump()

        # 1. Render HTML
        html_template = self.jinja_env.get_template("change_brief.html")
        html_content = html_template.render(**context)

        # 2. Render Markdown
        md_template = self.jinja_env.get_template("change_brief.md")
        markdown_content = md_template.render(**context)

        # 3. Create JSON companion
        json_content = payload.model_dump_json(indent=2)

        # File paths
        brief_id = payload.brief_id
        html_path = self.output_dir / f"brief_{brief_id}.html"
        markdown_path = self.output_dir / f"brief_{brief_id}.md"
        json_path = self.output_dir / f"brief_{brief_id}.json"

        if write_files:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            html_path.write_text(html_content, encoding="utf-8")
            markdown_path.write_text(markdown_content, encoding="utf-8")
            json_path.write_text(json_content, encoding="utf-8")
            logger.info("Persisted rendered brief files for brief_id %s in %s", brief_id, self.output_dir)

            # Compute SHA-256 hash of the rendered markdown output file
            rendered_file_hash = compute_sha256(markdown_path)
        else:
            from app.services.file_hash import compute_bytes_sha256
            rendered_file_hash = compute_bytes_sha256(markdown_content.encode("utf-8"))

        return RenderedBriefResult(
            brief_id=brief_id,
            html_content=html_content,
            markdown_content=markdown_content,
            json_content=json_content,
            html_path=html_path,
            markdown_path=markdown_path,
            json_path=json_path,
            rendered_file_hash=rendered_file_hash,
            primary_file_path=markdown_path,
        )
