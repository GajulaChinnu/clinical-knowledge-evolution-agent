"""Watchlist of monitored sources and reader for the versioned SYNTHETIC corpus.

The watchlist entry id is the stable source identity. Corpus locations (`corpus:<dir>`) hold
versioned files `v<version>.md` with YAML front matter; URL locations reuse URL ingestion.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

import yaml

from app.services.taxonomy import Taxonomy, TaxonomyError, get_taxonomy

DEFAULT_WATCHLIST_PATH = Path("./config/watchlist.yaml")
DEFAULT_CORPUS_ROOT = Path("./data/corpus")
SOURCE_TYPES = ("guideline", "safety_notice", "publication")
_ENTRY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_VERSION_FILE = re.compile(r"^v(\d+(?:\.\d+)*)\.md$", re.IGNORECASE)


class WatchlistError(ValueError):
    """Raised for malformed watchlist entries or corpus documents."""


@dataclass(frozen=True)
class WatchlistEntry:
    id: str
    title: str
    location: str
    source_type: str
    publisher: str
    quality_tier: int
    quality_basis: str
    departments: Tuple[str, ...]
    treatments: Tuple[str, ...]
    check_interval_hours: int = 24

    @property
    def is_corpus(self) -> bool:
        return self.location.startswith("corpus:")

    @property
    def is_url(self) -> bool:
        return self.location.lower().startswith(("http://", "https://"))

    @property
    def source_identity(self) -> str:
        return f"wl:{self.id}"

    def metadata(self) -> dict:
        return {
            "watchlist_id": self.id,
            "title": self.title,
            "source_type": self.source_type,
            "publisher": self.publisher,
            "quality_tier": self.quality_tier,
            "quality_basis": self.quality_basis,
            "departments": list(self.departments),
            "treatments": list(self.treatments),
        }


@dataclass
class CorpusVersion:
    """One published version of a corpus source."""
    entry_id: str
    version: str
    published_date: Optional[str]
    path: Path
    front_matter: Dict[str, object] = field(default_factory=dict)
    body: str = ""

    @property
    def version_key(self) -> Tuple[int, ...]:
        return version_key(self.version)


def version_key(version: str) -> Tuple[int, ...]:
    parts = re.findall(r"\d+", str(version))
    return tuple(int(p) for p in parts) or (0,)


def split_front_matter(text: str) -> Tuple[Dict[str, object], str]:
    """Split YAML front matter (--- ... ---) from a Markdown document."""
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        raise WatchlistError("Front matter opened with '---' but never closed.")
    try:
        meta = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError as e:
        raise WatchlistError(f"Invalid front matter: {e}") from e
    if not isinstance(meta, dict):
        raise WatchlistError("Front matter must be a mapping.")
    return meta, text[end + len("\n---\n"):].lstrip("\n")


class Watchlist:
    def __init__(self, entries: List[WatchlistEntry], corpus_root: Path) -> None:
        self.entries: Dict[str, WatchlistEntry] = {e.id: e for e in entries}
        self.corpus_root = corpus_root

    def __iter__(self):
        return iter(self.entries.values())

    def get(self, entry_id: str) -> WatchlistEntry:
        if entry_id not in self.entries:
            raise WatchlistError(f"Unknown watchlist entry '{entry_id}'")
        return self.entries[entry_id]

    def entry_for_identity(self, source_identity: str) -> Optional[WatchlistEntry]:
        if source_identity and source_identity.startswith("wl:"):
            return self.entries.get(source_identity[3:])
        return None

    def select(
        self,
        department: Optional[str] = None,
        treatments: Iterable[str] = (),
        taxonomy: Optional[Taxonomy] = None,
        include_safety_notices_for_treatment: bool = True,
    ) -> List[WatchlistEntry]:
        """Entries relevant to a department and/or treatments.

        A source qualifies if it is tagged with the department, or (safety notices) it covers one
        of the treatments in any department, because safety notices are cross-departmental.
        """
        direct = set(treatments)
        same_class = set(direct)
        if taxonomy is not None:
            classes = {taxonomy.class_of(t) for t in direct} - {None}
            same_class |= {m for c in classes for m in taxonomy.members_of_class(c)}
        chosen = []
        for entry in self.entries.values():
            in_department = department is None or department in entry.departments
            tagged = set(entry.treatments)
            if in_department and (not direct or entry.source_type != "publication" or direct & tagged):
                # Guidelines and notices of the department always; publications only when they
                # are about this exact treatment (not merely a sibling in the same class).
                chosen.append(entry)
            elif include_safety_notices_for_treatment and entry.source_type == "safety_notice" and same_class & tagged:
                chosen.append(entry)
        return sorted(chosen, key=lambda e: e.id)

    # ---- corpus ------------------------------------------------------------------------
    def corpus_dir(self, entry: WatchlistEntry) -> Path:
        if not entry.is_corpus:
            raise WatchlistError(f"Entry '{entry.id}' is not a corpus source")
        rel = entry.location[len("corpus:"):].strip().strip("/")
        directory = (self.corpus_root / rel).resolve()
        if self.corpus_root.resolve() not in directory.parents and directory != self.corpus_root.resolve():
            raise WatchlistError(f"Corpus location escapes the corpus root: {entry.location}")
        return directory

    def corpus_versions(self, entry: WatchlistEntry) -> List[CorpusVersion]:
        """All published versions of a corpus source, oldest first."""
        directory = self.corpus_dir(entry)
        if not directory.is_dir():
            raise WatchlistError(f"Corpus directory not found for '{entry.id}': {directory}")
        versions: List[CorpusVersion] = []
        for path in directory.iterdir():
            match = _VERSION_FILE.match(path.name)
            if not (path.is_file() and match):
                continue
            meta, body = split_front_matter(path.read_text(encoding="utf-8"))
            if meta.get("source_id") and meta["source_id"] != entry.id:
                raise WatchlistError(f"{path}: source_id '{meta['source_id']}' does not match entry '{entry.id}'")
            if meta.get("synthetic") is not True:
                raise WatchlistError(f"{path}: corpus documents must declare 'synthetic: true'")
            published = meta.get("published_date")
            if isinstance(published, date):
                published = published.isoformat()
            versions.append(CorpusVersion(
                entry_id=entry.id,
                version=str(meta.get("version") or match.group(1)),
                published_date=str(published) if published else None,
                path=path,
                front_matter=meta,
                body=body,
            ))
        return sorted(versions, key=lambda v: v.version_key)


def _entry_from_dict(raw: dict, taxonomy: Taxonomy, index: int) -> WatchlistEntry:
    where = f"watchlist.sources[{index}]"
    for key in ("id", "title", "location", "source_type", "publisher", "quality_tier"):
        if raw.get(key) in (None, ""):
            raise WatchlistError(f"{where}: missing '{key}'")
    if not _ENTRY_ID.match(str(raw["id"])):
        raise WatchlistError(f"{where}: id must be lowercase letters, digits and hyphens")
    if raw["source_type"] not in SOURCE_TYPES:
        raise WatchlistError(f"{where}: source_type must be one of {SOURCE_TYPES}")
    tier = int(raw["quality_tier"])
    if not 1 <= tier <= 5:
        raise WatchlistError(f"{where}: quality_tier must be 1-5")
    location = str(raw["location"]).strip()
    if not (location.startswith("corpus:") or location.lower().startswith(("http://", "https://"))):
        raise WatchlistError(f"{where}: location must be 'corpus:<dir>' or an http(s) URL")
    try:
        departments = tuple(taxonomy.validate_departments(raw.get("departments") or []))
        treatments = tuple(taxonomy.validate_treatments(raw.get("treatments") or []))
    except TaxonomyError as e:
        raise WatchlistError(f"{where}: {e}") from e
    if not departments:
        raise WatchlistError(f"{where}: at least one department is required")
    return WatchlistEntry(
        id=str(raw["id"]).strip(),
        title=str(raw["title"]),
        location=location,
        source_type=raw["source_type"],
        publisher=str(raw["publisher"]),
        quality_tier=tier,
        quality_basis=str(raw.get("quality_basis") or ""),
        departments=departments,
        treatments=treatments,
        check_interval_hours=int(raw.get("check_interval_hours") or 24),
    )


def load_watchlist(
    path: Union[str, Path] = DEFAULT_WATCHLIST_PATH,
    taxonomy: Optional[Taxonomy] = None,
    corpus_root: Union[str, Path] = DEFAULT_CORPUS_ROOT,
) -> Watchlist:
    path = Path(path)
    if not path.exists():
        raise WatchlistError(f"Watchlist file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise WatchlistError(f"Watchlist '{path}' is not valid YAML: {e}") from e
    taxonomy = taxonomy or get_taxonomy()
    entries = [_entry_from_dict(raw, taxonomy, i) for i, raw in enumerate(data.get("sources") or [])]
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise WatchlistError("Duplicate watchlist entry ids.")
    return Watchlist(entries, Path(corpus_root))
