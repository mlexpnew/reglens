import pytest

from reglens.ingest.manifest import ManifestError, load_manifest

HEADER = "doc_id,circular_no,title,date,url\n"


def write(tmp_path, body):
    p = tmp_path / "sources.csv"
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_valid_manifest(tmp_path):
    p = write(tmp_path, "circ-001,SEBI/HO/1,Title,2026-01-15,https://www.sebi.gov.in/a.pdf\n")
    [src] = load_manifest(p)
    assert src.doc_id == "circ-001" and src.filename == "circ-001.pdf"


@pytest.mark.parametrize(
    "row",
    [
        "Bad ID!,x,t,2026-01-15,https://a\n",
        "circ-1,x,t,15-01-2026,https://a\n",
        "circ-1,x,t,2026-01-15,http://insecure\n",
        "circ-1,x,t,2026-01-15,https://a\ncirc-1,x,t,2026-01-15,https://b\n",
    ],
)
def test_invalid_rows_rejected(tmp_path, row):
    with pytest.raises(ManifestError):
        load_manifest(write(tmp_path, row))
