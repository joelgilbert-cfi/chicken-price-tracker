from datetime import date

import pytest

from scraper.article_api import (
    PageArticleCandidate,
    build_issue_id,
    extract_article_detail_image_url,
    flatten_article_refs,
    measure_kpta_color_ratios,
    sanitize_filename,
    score_kpta_text_signals,
    select_kpta_page_article_candidate,
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


def test_select_kpta_page_article_filters_before_ranking() -> None:
    false_positive = _candidate(
        article_id="VVAANINEW_BEN_20260627_5_2",
        green_ratio=0.3189,
        red_ratio=0.0327,
    )
    kpta = _candidate(
        article_id="VVAANINEW_BEN_20260627_5_6",
        green_ratio=0.1120,
        red_ratio=0.0727,
        text_score=100,
    )

    assert select_kpta_page_article_candidate([false_positive, kpta]) == kpta


def test_select_kpta_page_article_prefers_kpta_text_signal() -> None:
    green_red_ad = _candidate(
        article_id="VVAANINEW_BEN_20260701_1_5",
        green_ratio=0.6433,
        red_ratio=0.0434,
        text_score=0,
    )
    kpta = _candidate(
        article_id="VVAANINEW_BEN_20260701_1_7",
        green_ratio=0.1600,
        red_ratio=0.0800,
        text_score=100,
    )

    assert select_kpta_page_article_candidate([green_red_ad, kpta]) == kpta


def test_select_kpta_page_article_rejects_colour_only_candidate() -> None:
    unrelated_article = _candidate(
        article_id="VVAANINEW_BEN_20260728_12_10",
        green_ratio=0.1158,
        red_ratio=0.0739,
    )

    assert select_kpta_page_article_candidate([unrelated_article]) is None


def test_select_kpta_page_article_accepts_phone_and_date_anchor() -> None:
    candidate = _candidate(
        article_id="VVAANINEW_BEN_20260728_12_3",
        green_ratio=0.1158,
        red_ratio=0.0739,
        text_score=70,
        text_matches=("7618763488", "date"),
    )

    assert select_kpta_page_article_candidate([candidate]) == candidate


def test_score_kpta_text_signals_detects_unique_article_anchor(tmp_path, monkeypatch) -> None:
    from PIL import Image

    image_path = tmp_path / "article.png"
    Image.new("RGB", (120, 80), "white").save(image_path)

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *_args, **_kwargs: "KPTA 01-07-2026 M: 7618763488",
    )

    score, matches = score_kpta_text_signals(image_path, date(2026, 7, 1))

    assert score >= 170
    assert "kpta" in matches
    assert "7618763488" in matches
    assert "date" in matches


def _candidate(
    article_id: str,
    green_ratio: float,
    red_ratio: float,
    text_score: int = 0,
    text_matches: tuple[str, ...] | None = None,
) -> PageArticleCandidate:
    return PageArticleCandidate(
        issue_id="VVAANINEW_BEN_20260627",
        page_number=5,
        article_id=article_id,
        image_url=f"https://images.example/{article_id}.jpg",
        image_path=f"artifacts/{article_id}.jpg",
        green_ratio=green_ratio,
        red_ratio=red_ratio,
        score=text_score + (green_ratio * red_ratio),
        x1=0,
        y1=0,
        x2=0,
        y2=0,
        text_score=text_score,
        text_matches=text_matches if text_matches is not None else (("kpta",) if text_score else ()),
    )
