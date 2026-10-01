"""Reusable file hashing service for computing SHA-256 digests."""

import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hexadecimal hash digest of a file.

    Args:
        file_path: Path to the target file.
        chunk_size: Number of bytes to read per chunk (defaults to 64KB).

    Returns:
        Lowercase 64-character hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the target file does not exist.
        IsADirectoryError: If the target path points to a directory.
        OSError: If reading the file fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Target path is a directory, not a file: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest().lower()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute the SHA-256 hexadecimal hash digest of in-memory bytes.

    Args:
        data: Raw bytes to hash.

    Returns:
        Lowercase 64-character hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest().lower()
