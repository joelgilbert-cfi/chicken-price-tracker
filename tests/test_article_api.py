from datetime import date

from scraper.article_api import build_issue_id, flatten_article_refs, sanitize_filename


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
        }
    ]


def test_sanitize_filename_keeps_safe_identifier_characters() -> None:
    assert sanitize_filename("VVAANINEW/BEN 20260625:3:5") == "VVAANINEW_BEN_20260625_3_5"
