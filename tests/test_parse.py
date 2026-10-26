from reglens.ingest.parse import extract_pages, strip_boilerplate


def test_extract_strips_letterhead_and_page_numbers(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert len(pages) == 3
    joined = "\n".join(pages)
    assert "SECURITIES AND EXCHANGE BOARD OF INDIA" not in joined
    assert "Page 1 of 3" not in joined
    assert "Mutual funds shall disclose" in pages[0]
    assert "segregate client funds" in pages[1]


def test_strip_keeps_unique_content():
    pages = ["HEAD\nunique a", "HEAD\nunique b", "HEAD\nunique c"]
    assert strip_boilerplate(pages) == ["unique a", "unique b", "unique c"]
