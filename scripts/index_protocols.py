"""Smoke script for indexing synthetic clinical protocols into local ChromaDB."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.services.config_service import load_config
from app.services.protocol_index import ProtocolIndexService


def main() -> int:
    parser = argparse.ArgumentParser(description="Index clinical protocols into local ChromaDB.")
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default=None,
        help="Path to protocol source directory (defaults to config PROTOCOL_DIR)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=None,
        help="Path to local ChromaDB storage directory (defaults to config CHROMA_DIR)",
    )
    args = parser.parse_args()

    config = load_config()
    target_protocol_dir = Path(args.protocol_dir) if args.protocol_dir else config.protocol_dir
    target_chroma_dir = Path(args.chroma_dir) if args.chroma_dir else config.chroma_dir

    try:
        service = ProtocolIndexService(chroma_dir=target_chroma_dir, config=config)
        summary = service.index_directory(protocol_dir=target_protocol_dir)

        print(f"Protocols discovered: {summary.protocols_discovered}")
        print(f"Sections indexed: {summary.sections_indexed}")
        print(f"Skipped: {summary.sections_skipped}")
        print(f"Failed: {summary.protocols_failed}")

        if summary.errors:
            print("\nErrors encountered:")
            for err in summary.errors:
                print(f" - {err}")

        return 0 if summary.protocols_failed == 0 else 1

    except Exception as e:
        print(f"Error during protocol indexing: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
