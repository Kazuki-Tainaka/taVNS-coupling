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
    "scripts/compute_coupling_metrics.py",
    "scripts/compute_hrv_metrics.py",
    "scripts/compute_rhomax_windows.py",
    "scripts/compute_wtc.py",
    "scripts/compute_fixed_lag_cross_correlation.py",
    "scripts/compute_var_residuals.py",
    "scripts/compute_brs_ramps.py",
    "scripts/compute_bootstrap_replicates.py",
    "scripts/compute_temporal_classification.py",
    "scripts/compute_its_regression.py",
    "scripts/run_all.py",
    "scripts/lib/quality.py",
    "scripts/lib/coupling.py",
    "lib/quality.py",
    "lib/coupling.py",
    "data/derived/README.md",
    "data/derived/per_subject_coupling.csv",
    "data/derived/per_subject_hrv.csv",
    "data/derived/bootstrap/README.md",
    "figures/README.md",
    "figures/regenerate_all.py",
    "figures/style.py",
    "figures/main/generate_fig1.py",
    "figures/main/generate_fig2.py",
    "figures/main/generate_fig3.py",
    "figures/main/generate_fig4.py",
    "figures/supplementary/generate_figS1.py",
    "figures/supplementary/generate_figS2.py",
    "figures/supplementary/generate_figS3.py",
    "figures/supplementary/generate_figS4.py",
    "figures/supplementary/generate_figS5.py",
    "figures/supplementary/generate_figS6.py",
    "figures/supplementary/generate_figS7.py",
    "docs/reproducibility_tiers.md",
    "CHANGELOG.md",
]


def test_public_repository_structure_is_complete():
    missing = [p for p in REQUIRED_PATHS if not Path(p).exists()]
    assert missing == []


def test_readme_documents_public_scope():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Figure regeneration" in readme
    assert "Reproducibility tiers" in readme
    assert "jRCT1032230440" in readme
    assert "python scripts/run_all.py" in readme
    assert "Katahara Y, Iijima A, Tainaka K" in readme
