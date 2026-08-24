import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import visualize
from visualize import render_layout_preview


def test_layout_preview_rejects_zero_width_rows(tmp_path):
    """Catch an empty row reaching Pillow as a zero-width image."""
    layout = tmp_path / "empty-row.txt"
    layout.write_text("\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="non-empty rectangular text map"
    ):
        render_layout_preview(layout)


def test_layout_preview_writes_default_png_with_literal_cell_colors(tmp_path):
    """Catch wrong default paths, dimensions, or character-to-color mapping."""
    layout = tmp_path / "layout.txt"
    layout.write_text("X.G\n.GX\n", encoding="utf-8")

    output = render_layout_preview(layout, cell_size=8)

    assert output == layout.with_suffix(".png")
    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (24, 16)
        assert image.getpixel((1, 1)) == (74, 91, 110)
        assert image.getpixel((9, 1)) == (244, 247, 250)
        assert image.getpixel((17, 1)) == (42, 157, 143)


def test_layout_preview_writes_an_explicit_nested_output(tmp_path):
    """Catch explicit output paths being ignored or parent creation regressing."""
    layout = tmp_path / "layout.txt"
    layout.write_text("XG\n..\n", encoding="utf-8")
    output = tmp_path / "nested" / "preview.png"

    assert render_layout_preview(layout, output, cell_size=8) == output
    assert output.is_file()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("XX\nX\n", "rectangular"),
        ("XZ\n..\n", "only supports X, ., and G"),
    ],
)
def test_layout_preview_rejects_invalid_map_text(tmp_path, text, message):
    """Catch malformed maps being rendered with ambiguous geometry or symbols."""
    layout = tmp_path / "invalid.txt"
    layout.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        render_layout_preview(layout, cell_size=8)


def test_layout_preview_rejects_cells_smaller_than_eight_pixels(tmp_path):
    """Catch unreadably small cells bypassing the documented minimum."""
    layout = tmp_path / "layout.txt"
    layout.write_text("X\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 8 pixels"):
        render_layout_preview(layout, cell_size=7)


def test_layout_preview_reports_missing_pillow(monkeypatch, tmp_path):
    """Catch a raw optional-dependency ImportError leaking from the CLI."""
    layout = tmp_path / "layout.txt"
    layout.write_text("X\n", encoding="utf-8")
    real_import = builtins.__import__

    def import_without_pillow(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("Pillow intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pillow)

    with pytest.raises(RuntimeError, match="require Pillow"):
        render_layout_preview(layout, cell_size=8)


def test_layout_preview_cli_exits_before_policy_or_environment(monkeypatch, capsys):
    """Catch preview mode falling through into checkpoint replay setup."""
    preview = Path("preview.png")
    monkeypatch.setattr(
        visualize,
        "parse_args",
        lambda: SimpleNamespace(
            layout_preview="layout.txt",
            layout_preview_output=str(preview),
            cell_size=8,
        ),
    )
    monkeypatch.setattr(
        visualize,
        "render_layout_preview",
        lambda source, output, cell_size: preview,
    )

    def forbidden_replay(**kwargs):
        raise AssertionError("preview mode entered policy/environment replay")

    monkeypatch.setattr(visualize, "run_visualization", forbidden_replay)

    visualize.main()

    assert json.loads(capsys.readouterr().out) == {
        "layout_preview": str(preview)
    }
