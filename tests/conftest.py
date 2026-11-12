"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from db.repository import Repository


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Repository:
    """Create a repository with a temporary database for testing."""
    return Repository(tmp_path / "test.db")
