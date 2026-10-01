"""Smoke test verifying that required CKEA project scaffold directories exist."""

from pathlib import Path
import unittest

REQUIRED_DIRECTORIES = [
    Path("app"),
    Path("app/agents"),
    Path("app/schemas"),
    Path("app/models"),
    Path("app/services"),
    Path("app/orchestration"),
    Path("app/repositories"),
    Path("app/utils"),
    Path("app/ui"),
    Path("config"),
    Path("data"),
    Path("data/sources"),
    Path("data/protocols"),
    Path("data/evaluation"),
    Path("data/output"),
    Path("data/chroma"),
    Path("templates"),
    Path("tests"),
    Path("tests/unit"),
    Path("tests/integration"),
    Path("tests/evaluation"),
    Path("scripts"),
    Path("logs"),
    Path("docs"),
]


class TestProjectStructure(unittest.TestCase):
    """Test suite for verifying the CKEA project scaffold structure."""

    def test_required_directories_exist(self):
        repo_root = Path(__file__).resolve().parents[2]
        for rel_dir in REQUIRED_DIRECTORIES:
            target_dir = repo_root / rel_dir
            self.assertTrue(
                target_dir.is_dir(),
                f"Required directory does not exist: {rel_dir}",
            )


def test_required_directories_exist():
    """Pytest-compatible test verifying required project directories exist."""
    repo_root = Path(__file__).resolve().parents[2]
    for rel_dir in REQUIRED_DIRECTORIES:
        target_dir = repo_root / rel_dir
        assert target_dir.is_dir(), f"Required directory does not exist: {rel_dir}"


if __name__ == "__main__":
    unittest.main()
