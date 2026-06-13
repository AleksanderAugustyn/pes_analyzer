"""The bundled reference docs ship with the package and are discoverable."""
from pathlib import Path

import pes_analyzer


def test_docs_path_returns_existing_dir() -> None:
    p = pes_analyzer.docs_path()
    assert isinstance(p, Path)
    assert p.is_dir()


def test_bundled_reference_docs_present() -> None:
    docs = pes_analyzer.docs_path()
    for name in ("API.md", "ALGORITHMS.md", "USAGE.md"):
        assert (docs / name).is_file(), f"missing bundled doc: {name}"
