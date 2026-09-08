"""Regression tests for the Phase 1 static-layout preview utility."""

import pytest

from scripts.preview_phase1_layouts import PROFILES, _make_environment


@pytest.mark.parametrize("profile", PROFILES, ids=lambda item: item.name)
def test_preview_environment_has_static_shelves_after_construction(profile) -> None:
    """Preview rendering needs the RWARE static shelf placement to exist."""
    environment = _make_environment(profile)
    try:
        assert len(environment.shelfs) > 0
    finally:
        environment.close()
