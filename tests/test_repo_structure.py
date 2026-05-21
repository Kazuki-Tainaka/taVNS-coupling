from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "data/LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "requirements-dev.txt",
    ".zenodo.json",
    ".github/workflows/ci.yml",
    "docs/data_dictionary.md",
    "docs/methods_appendix.md",
    "scripts/01_compute_coupling_metrics.py",
    "scripts/02_compute_hrv_metrics.py",
    "scripts/run_all.py",
    "scripts/lib/quality.py",
    "scripts/lib/coupling.py",
    "lib/quality.py",
    "lib/coupling.py",
]


def test_public_repository_structure_is_complete():
    missing = [p for p in REQUIRED_PATHS if not Path(p).exists()]
    assert missing == []


def test_readme_documents_public_scope():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Anchor verification mode" in readme
    assert "not provided" in readme.lower()
    assert "jRCT1032230440" in readme
    assert "python scripts/run_all.py" in readme
    assert "Katahara Y, Iijima A, Tainaka K" in readme
