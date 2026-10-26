from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def make_pdf(
    path: Path, pages: list[list[str]], header: str = "SECURITIES AND EXCHANGE BOARD OF INDIA"
) -> Path:
    c = canvas.Canvas(str(path), pagesize=A4)
    for i, lines in enumerate(pages, start=1):
        c.drawString(72, 800, header)
        y = 760
        for ln in lines:
            c.drawString(72, y, ln)
            y -= 18
        c.drawString(290, 40, f"Page {i} of {len(pages)}")
        c.showPage()
    c.save()
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return make_pdf(
        tmp_path / "circ-001.pdf",
        [
            [
                "1. Mutual funds shall disclose portfolio holdings monthly.",
                "2. Disclosure within 10 days.",
            ],
            ["3. Stock brokers must segregate client funds.", "4. Reporting is quarterly."],
            ["5. This circular comes into force immediately."],
        ],
    )
