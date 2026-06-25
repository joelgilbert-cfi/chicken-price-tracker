from datetime import date

import pytest

from scraper.article_api import (
    build_issue_id,
    extract_article_detail_image_url,
    flatten_article_refs,
    measure_kpta_color_ratios,
    sanitize_filename,
)
from scraper.exceptions import KPTANotFoundError


def test_build_issue_id_uses_bengaluru_date() -> None:
    assert build_issue_id(date(2026, 6, 25)) == "VVAANINEW_BEN_20260625"


def test_flatten_article_refs_extracts_article_images() -> None:
    pages = [
        {
            "pageno": "3",
            "Articles": [
                {
                    "Article": {
                        "article_image_id": "VVAANINEW_BEN_20260625_3_5",
                        "r2imagename": "https://images.example/kpta.jpg",
                        "imagename": "https://fallback.example/kpta.jpg",
                    }
                },
                {
                    "Article": {
                        "article_image_id": "OTHER_20260625_3_6",
                        "r2imagename": "https://images.example/other.jpg",
                    }
                },
            ],
        }
    ]

    assert flatten_article_refs("VVAANINEW_BEN_20260625", pages) == [
        {
            "page_number": 3,
            "article_id": "VVAANINEW_BEN_20260625_3_5",
            "image_url": "https://images.example/kpta.jpg",
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 0,
        }
    ]


def test_sanitize_filename_keeps_safe_identifier_characters() -> None:
    assert sanitize_filename("VVAANINEW/BEN 20260625:3:5") == "VVAANINEW_BEN_20260625_3_5"


def test_extract_article_detail_image_url_prefers_r2_image_path() -> None:
    payload = [
        {
            "r2imagepath": "https://images.example/kpta.jpg",
            "fallbackimagepath": "https://fallback.example/kpta.jpg",
            "Article": {"x1": "1", "y1": "203", "x2": "83", "y2": "316"},
        }
    ]

    assert extract_article_detail_image_url(payload) == "https://images.example/kpta.jpg"


def test_extract_article_detail_image_url_rejects_missing_url() -> None:
    with pytest.raises(KPTANotFoundError):
        extract_article_detail_image_url([{"Article": {}}])


def test_measure_kpta_color_ratios_detects_green_and_red(tmp_path) -> None:
    from PIL import Image, ImageDraw

    image_path = tmp_path / "kpta_like.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 99, 35), fill=(0, 180, 40))
    draw.rectangle((50, 50, 90, 70), fill=(220, 0, 0))
    image.save(image_path)

    green_ratio, red_ratio = measure_kpta_color_ratios(image_path)
    assert green_ratio > 0.25
    assert red_ratio > 0.05
