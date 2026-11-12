from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "00_inventory.py"


def load_inventory():
    spec = importlib.util.spec_from_file_location("inventory_00", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    module.__spec__.loader.exec_module(module)  # type: ignore[union-attr]
    return module


inventory = load_inventory()


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\nhalo\n", encoding="utf-8")
    return path


def test_subtitle_in_parent_folder_must_match_title(tmp_path):
    raw = tmp_path / "Film"
    video = touch(raw / "Me.Before.You.2016.1080p" / "Me.Before.You.2016.1080p.mkv")
    wanted = touch(raw / "Me.Before.You.2016.720p.BluRay.srt")
    touch(raw / "The.Notebook.2004.1080p_eng.srt")

    assert inventory.find_subtitle(video, raw) == wanted


def test_subtitle_without_shared_title_word_is_rejected(tmp_path):
    raw = tmp_path / "Film"
    video = touch(raw / "Perks.Wallflower.2012.1080p" / "Perks.Wallflower.2012.1080p.mkv")
    touch(raw / "The.Notebook.2004.1080p_eng.srt")

    assert inventory.find_subtitle(video, raw) is None


def test_subtitle_in_same_folder_wins(tmp_path):
    raw = tmp_path / "Film"
    video = touch(raw / "About.Time.2013.1080p.mkv")
    own = touch(raw / "About.Time.2013.1080p.srt")
    touch(raw / "Other.Movie.2004.1080p_eng.srt")

    assert inventory.find_subtitle(video, raw) == own


def test_title_tokens_drop_release_noise_and_numbers():
    tokens = inventory.title_tokens("La.La.Land.2016.1080p.BluRay.x264-SPARKS")

    assert "land" in tokens
    assert "2016" not in tokens
    assert "1080p" not in tokens
    assert "bluray" not in tokens
