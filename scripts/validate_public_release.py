"""Validate public source tables, figures, metadata, and safety gates.

This validator uses an exact allow-list for the 54 author-approved public
derived beat tables under ``data/beats/``. Raw waveforms, direct identifiers,
secrets, private keys, absolute host paths, and unapproved participant files
remain hard-failure conditions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "expected_outputs"
PUBLIC_DATA = EXPECTED / "publication_source_data"
DATA3_PATH = (
    PUBLIC_DATA
    / "supplementary_data_3_brs_sensitivity_and_coupling_significance.csv"
)
ANCHORS_PATH = ROOT / "config" / "release_anchors.json"
S3_ARTWORK = (
    ROOT / "figures" / "supplementary_figure_s3_brs_specification_landscape.jpg"
)
S3_GENERATOR = ROOT / "scripts" / "generate_supplementary_figure_s3.py"
BEATS_DIR = ROOT / "data" / "beats"
BEATS_MANIFEST = ROOT / "config" / "approved_derived_beats_manifest.csv"
DATA_LICENSE = ROOT / "data" / "LICENSE"
DATA_README = ROOT / "data" / "README.md"

EXPECTED_BEATS_HEADER = ("beat_idx", "RRI_ms", "SBP_mmHg", "PAT_ms")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    selected = frame.loc[mask]
    require(len(selected) == 1, f"{label}: expected one row, found {len(selected)}")
    return selected.iloc[0]


def check_syntax() -> str:
    paths = sorted(
        path
        for folder in (ROOT / "src", ROOT / "scripts", ROOT / "tests")
        for path in folder.rglob("*.py")
    )
    require(paths, "no Python source found")
    for path in paths:
        source = path.read_text(encoding="utf-8-sig")
        compile(source, path.as_posix(), "exec")
    return f"{len(paths)} Python files compiled in memory"


def check_public_artifact_hashes(anchors: dict[str, object]) -> str:
    hashes = anchors["public_artifact_sha256"]
    assert isinstance(hashes, dict)
    locations = {
        "supplementary_data_1.csv": PUBLIC_DATA / "supplementary_data_1.csv",
        "supplementary_data_2.csv": PUBLIC_DATA / "supplementary_data_2.csv",
        "supplementary_data_3.csv": DATA3_PATH,
        "supplementary_figure_s3.jpg": S3_ARTWORK,
        "supplementary_figure_s3_generator.py": S3_GENERATOR,
    }
    for name, path in locations.items():
        require(path.is_file(), f"missing authoritative file: {name}")
        require(sha256_file(path) == hashes[name], f"SHA-256 mismatch: {name}")
    return f"{len(locations)} reviewed public artifact hashes match"


def check_central_brs(frame: pd.DataFrame, anchors: dict[str, object]) -> str:
    expected = anchors["central_brs"]
    assert isinstance(expected, dict)
    differences = pd.read_csv(
        PUBLIC_DATA / "Figure_2A_sorted_BRS_differences.csv"
    )["Stim_minus_Pre_BRS_ms_per_mmHg"].to_numpy(float)
    require(len(differences) == expected["n"], "central BRS n mismatch")
    require(
        np.isclose(
            differences.mean(), expected["mean_difference"], rtol=0.0, atol=1e-12
        ),
        "central BRS mean mismatch",
    )
    require(
        int(np.sum(differences < 0.0)) == expected["negative_participants"],
        "central BRS direction count mismatch",
    )
    base = (
        frame["analysis_family"].eq("reference_BRS")
        & frame["record_type"].eq("contrast_summary")
        & frame["contrast"].eq("Stim-Pre")
        & frame["outcome"].eq("BRS_seq_all")
    )
    mean_row = one_row(frame, base & frame["metric"].eq("mean_difference"), "BRS mean")
    dz_row = one_row(frame, base & frame["metric"].eq("cohens_dz"), "BRS dz")
    require(np.isclose(float(mean_row["value"]), -2.05071, atol=5e-6), "BRS table mean")
    require(np.isclose(float(mean_row["ci_low"]), -3.81259, atol=5e-6), "BRS CI low")
    require(np.isclose(float(mean_row["ci_high"]), -1.09512, atol=5e-6), "BRS CI high")
    require(np.isclose(float(mean_row["p_value"]), expected["wilcoxon_p"], atol=5e-8), "BRS p")
    require(np.isclose(float(dz_row["value"]), -0.740209, atol=5e-6), "BRS dz")
    return "n=18, mean=-2.050708187408464, 17/18 negative, CI/p/dz match"


def check_brs_landscape(frame: pd.DataFrame, anchors: dict[str, object]) -> str:
    expected = anchors["brs_landscape"]
    assert isinstance(expected, dict)
    rows = frame.loc[
        frame["record_type"].eq("full_factorial")
        & frame["metric"].eq("mean_difference_stim_minus_pre")
    ].copy()
    rows["numeric_value"] = pd.to_numeric(rows["value"], errors="raise")
    rows["numeric_p"] = pd.to_numeric(rows["p_value"], errors="raise")
    require(len(rows) == 216, "full-factorial row count")
    require(rows["setting_id"].nunique() == 72, "full-factorial setting count")
    for direction in ("all", "up", "down"):
        require(
            int(rows["direction"].eq(direction).sum())
            == expected["settings_per_direction"],
            f"{direction} setting count",
        )
    all_rows = rows.loc[rows["direction"].eq("all")]
    reference = one_row(
        all_rows,
        all_rows["setting_id"].eq("lag1_r0p80_sbp1p0_len3_mean"),
        "BRS landscape reference",
    )
    require(
        np.isclose(float(reference["numeric_value"]), expected["reference_all"], atol=5e-7),
        "reference all estimate",
    )
    require(int((all_rows["numeric_value"] < 0).sum()) == expected["all_negative"], "all negative count")
    require(
        int((rows.loc[rows["direction"].eq("up"), "numeric_value"] < 0).sum())
        == expected["up_negative"],
        "up negative count",
    )
    require(
        int((rows.loc[rows["direction"].eq("down"), "numeric_value"] < 0).sum())
        == expected["down_negative"],
        "down negative count",
    )
    require(int((rows["numeric_value"] < 0).sum()) == expected["all_up_down_negative_total"], "total negative count")
    require(
        int((all_rows["numeric_value"] > 0).sum()) == expected["positive_all"],
        "positive all count",
    )
    require(
        np.isclose(
            all_rows["numeric_value"].median(),
            expected["median_all"],
            atol=5e-7,
        ),
        "all median estimate",
    )
    require(
        np.isclose(
            all_rows["numeric_value"].min(), expected["minimum_all"], atol=5e-7
        ),
        "minimum all estimate",
    )
    require(
        np.isclose(
            all_rows["numeric_value"].max(), expected["maximum_positive_all"], atol=5e-7
        ),
        "maximum positive all estimate",
    )
    all_n = pd.to_numeric(all_rows["n"], errors="raise")
    require(int(all_n.min()) == expected["evaluable_n_min"], "minimum evaluable n")
    require(int(all_n.max()) == expected["evaluable_n_max"], "maximum evaluable n")
    positive = all_rows.loc[all_rows["numeric_value"] > 0]
    require(positive["numeric_p"].min() >= expected["minimum_p_among_positive_all"], "positive all setting support")
    return (
        "72 settings/direction; reference, median, range, n=11-18, "
        "67/72 all, 54/72 up, 67/72 down, and 188/216 total negative"
    )


def coherence_difference(frame: pd.DataFrame, setting: str) -> tuple[float, float]:
    rows = frame.loc[
        frame["analysis_family"].eq("coherence_segment_length_sensitivity")
        & frame["setting_id"].eq(setting)
    ].copy()
    rows["numeric_value"] = pd.to_numeric(rows["value"], errors="raise")
    pivot = rows.pivot(index="subject_id", columns="phase", values="numeric_value")
    differences = (pivot["Stim"] - pivot["Pre"]).to_numpy(float)
    result = stats.wilcoxon(differences, alternative="two-sided", method="auto")
    return float(np.mean(differences)), float(result.pvalue)


def check_coherence(frame: pd.DataFrame, anchors: dict[str, object]) -> str:
    expected = anchors["coherence"]
    assert isinstance(expected, dict)
    ref_mean, ref_p = coherence_difference(frame, "reference")
    sen_mean, sen_p = coherence_difference(frame, "segment_length_sensitivity")
    require(np.isclose(ref_mean, expected["reference_mean_difference"], atol=1e-12), "reference coherence mean")
    require(np.isclose(ref_p, expected["reference_wilcoxon_p"], atol=1e-15), "reference coherence p")
    require(np.isclose(sen_mean, expected["sensitivity_mean_difference"], atol=1e-12), "sensitivity coherence mean")
    require(np.isclose(sen_p, expected["sensitivity_wilcoxon_p"], atol=1e-15), "sensitivity coherence p")
    prevalence = frame.loc[
        frame["analysis_family"].eq("coherence_significance")
        & frame["record_type"].eq("phase_prevalence")
    ].set_index("phase")
    for phase, key in (("Pre", "pre_significant"), ("Stim", "stim_significant"), ("Post", "post_significant")):
        require(int(prevalence.loc[phase, "successes"]) == expected[key], f"{phase} prevalence")
    mcnemar = one_row(
        frame,
        frame["analysis_family"].eq("coherence_significance")
        & frame["record_type"].eq("prevalence_comparison")
        & frame["contrast"].eq("Stim-vs-Pre"),
        "coherence McNemar",
    )
    cochran = one_row(
        frame,
        frame["analysis_family"].eq("coherence_significance")
        & frame["record_type"].eq("prevalence_comparison")
        & frame["contrast"].eq("Pre-Stim-Post"),
        "coherence Cochran Q",
    )
    require(np.isclose(float(mcnemar["p_value"]), expected["mcnemar_p"]), "McNemar p")
    require(np.isclose(float(cochran["p_value"]), expected["cochran_q_p"], atol=5e-4), "Cochran Q p")
    return "512/256 and 256/128 contrasts plus 14/18, 15/18, 15/18 prevalence match"


def check_haemodynamics(frame: pd.DataFrame, anchors: dict[str, object]) -> str:
    expected = anchors["haemodynamics"]
    assert isinstance(expected, dict)
    row = one_row(
        frame,
        frame["analysis_family"].eq("haemodynamic_hrv_context")
        & frame["record_type"].eq("contrast_summary")
        & frame["outcome"].eq("mean_SBP")
        & frame["contrast"].eq("Stim-Pre"),
        "mean SBP",
    )
    require(np.isclose(float(row["mean_difference"]), expected["mean_sbp_stim_minus_pre"], atol=5e-6), "SBP mean")
    require(np.isclose(float(row["p_value"]), expected["wilcoxon_p"], atol=5e-6), "SBP p")
    require(np.isclose(float(row["q_value"]), expected["bh_q"], atol=5e-6), "SBP q")
    return "mean SBP +6.11738 mmHg; p=0.00475; q=0.038"


def check_native_var(frame: pd.DataFrame, anchors: dict[str, object]) -> str:
    expected = anchors["native_var"]
    assert isinstance(expected, dict)
    rows = frame.loc[
        frame["analysis_family"].eq("GC_significance_native_paired_beats")
        & frame["record_type"].eq("participant_phase_direction")
    ]
    fits = rows.drop_duplicates(["subject_id", "phase"])
    require(len(fits) == expected["unique_fits"], "VAR unique fits")
    require(fits["model_fit_status"].eq("fit_succeeded").all(), "VAR fit status")
    require(fits["stability_status"].eq("stable").all(), "VAR stability")
    overall = one_row(
        frame,
        frame["analysis_family"].eq("GC_significance_native_paired_beats")
        & frame["record_type"].eq("diagnostic_summary")
        & frame["phase"].eq("NA"),
        "VAR overall diagnostics",
    )
    require(int(overall["whiteness_pass_count"]) == expected["whiteness_pass"], "VAR whiteness")
    require(int(overall["normality_pass_count"]) == expected["normality_pass"], "VAR normality")
    require(int(overall["both_residual_diagnostics_pass_count"]) == expected["both_pass"], "VAR both diagnostics")
    require(
        pd.to_numeric(fits["candidate_order_min"], errors="raise").eq(1).all(),
        "VAR order minimum",
    )
    require(
        pd.to_numeric(fits["candidate_order_max"], errors="raise").eq(10).all(),
        "VAR order maximum",
    )
    return "54/54 successful and stable; diagnostics 42/54, 7/54, 4/54"


def check_methods_matched_brs() -> str:
    frame = pd.read_csv(EXPECTED / "methods_text_matched_brs_summary.csv")
    branches = ["A0_MAX", "A1_MAX", "A0_OVERLAP", "A1_OVERLAP"]
    rows = frame.loc[frame["branch"].isin(branches)]
    require(set(rows["branch"]) == set(branches), "methods branch set")
    require((rows["mean_difference"] < 0).all(), "methods branch direction")
    return "A0_MAX, A1_MAX, A0_OVERLAP, A1_OVERLAP all remain negative"


def check_nonlinear_sensitivity() -> str:
    frame = pd.read_csv(
        EXPECTED / "nonlinear_phase_contrasts.csv",
        dtype=str,
        keep_default_na=False,
    )
    rows = frame.loc[
        frame["contrast"].eq("Stim-Pre")
        & frame["primary_family"].str.lower().eq("true")
    ].copy()
    rows["q"] = pd.to_numeric(rows["BH_q_primary_family"], errors="raise")
    require(len(rows) == 6, "nonlinear primary test count")
    require((rows["q"] >= 0.05).all(), "nonlinear BH result")
    return "six primary Stim-Pre tests; none survives BH correction"


def check_metadata() -> str:
    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    require(metadata["version"] == "1.1.0", "metadata version")
    require(metadata["upload_type"] == "software", "metadata resource type")
    require(metadata["license"] == "MIT", "metadata license")
    require(len(metadata["creators"]) == 3, "metadata creators")
    require("Acute transcutaneous" in metadata["title"], "metadata title")
    description = metadata.get("description", "")
    require("data/beats" in description, "metadata description missing data/beats mention")
    require("CC BY 4.0" in description, "metadata description missing CC BY 4.0 mention")
    require("Raw ECG" in description, "metadata description missing raw-waveform exclusion")
    concept_doi = "10.5281/zenodo.20323694"
    trial_id = "jRCT1032230440"
    related = metadata.get("related_identifiers", [])
    related_identifiers = {entry.get("identifier") for entry in related if isinstance(entry, dict)}
    require(concept_doi in related_identifiers, f"metadata missing concept DOI {concept_doi}")
    require(trial_id in related_identifiers, f"metadata missing trial identifier {trial_id}")
    return (
        "Zenodo JSON parses; title, creators, software type, version, license, "
        f"description, concept DOI {concept_doi}, and trial identifier {trial_id} present"
    )


def check_public_data_boundary(frame: pd.DataFrame) -> str:
    subject_ids = frame.loc[frame["subject_id"].ne("NA"), "subject_id"]
    valid_ids = subject_ids.str.fullmatch(r"(?:S\d{2}|(?:[1-9]|1[0-8]))")
    require(bool(valid_ids.all()), "non-pseudonymous participant label")
    forbidden_columns = {
        "ecg_waveform",
        "continuous_blood_pressure",
        "rri_time_series",
        "sbp_time_series",
    }
    require(forbidden_columns.isdisjoint(frame.columns), "time-series column present")
    return "participant labels are pseudonymous; no waveform/time-series columns"


def check_s3_render() -> str:
    with tempfile.TemporaryDirectory(prefix="tavns_s3_validation_") as temp:
        output = Path(temp)
        command = [
            sys.executable,
            str(S3_GENERATOR),
            "--supplementary-data-3",
            str(DATA3_PATH),
            "--output-dir",
            str(output),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        require(result.returncode == 0, f"S3 generator failed: {result.stderr}")
        svg = output / "supplementary_figure_s3_brs_specification_landscape.svg"
        png = output / "supplementary_figure_s3_brs_specification_landscape.png"
        require(svg.is_file() and svg.stat().st_size > 50_000, "S3 SVG missing")
        require(png.is_file() and png.stat().st_size > 50_000, "S3 PNG missing")
        text = svg.read_text(encoding="utf-8")
        require("188/216" in text, "S3 direction-consistency annotation missing")
        require("67/72" in text, "S3 all-sequence annotation missing")
        require("54/72" in text, "S3 up-sequence annotation missing")
        require("Median = -0.94" in text, "S3 median annotation missing")
        require("Reference = -2.05" in text, "S3 reference annotation missing")
        require("Range: -2.58 to +0.24" in text, "S3 range annotation missing")
        require("5/72 positive" in text, "S3 positive-estimate annotation missing")

    source = S3_GENERATOR.read_text(encoding="utf-8")
    require("ax_b.scatter" in source, "S3 panel b is not a point plot")
    require("ax_b.bar" not in source and "ax_b.barh" not in source, "S3 panel b bar found")
    require("colors = [BLUE, TEAL, PURPLE]" in source, "S3 panel d palette mismatch")
    require("point_colors = np.where(estimates < 0, BLUE, ORANGE)" in source, "S3 positive palette mismatch")
    require("$r$ ≥" in source and "all nominal $p$" in source, "S3 statistical-symbol formatting mismatch")
    require("Stim–Pre" in source, "S3 contrast label mismatch")
    require(
        "f'$n$ = {n_values.min()}–{n_values.max()}'" in source,
        "S3 evaluable-sample annotation logic missing",
    )
    return "four-panel S3 generator and layout semantics match the reviewed anchors"


def _read_manifest() -> pd.DataFrame:
    require(BEATS_MANIFEST.is_file(), f"missing manifest: {BEATS_MANIFEST}")
    return pd.read_csv(BEATS_MANIFEST, dtype=str, keep_default_na=False)


def check_author_approved_derived_beats() -> str:
    require(BEATS_DIR.is_dir(), f"missing data/beats/ directory: {BEATS_DIR}")
    require(DATA_LICENSE.is_file(), f"missing data/LICENSE: {DATA_LICENSE}")
    require(DATA_README.is_file(), f"missing data/README.md: {DATA_README}")

    beats_files = sorted(p for p in BEATS_DIR.iterdir() if p.is_file())
    non_csv = [p.name for p in beats_files if p.suffix != ".csv"]
    require(not non_csv, f"non-CSV file present under data/beats/: {non_csv}")
    require(
        len(beats_files) == 54,
        f"data/beats/ contains {len(beats_files)} files; expected exactly 54",
    )

    manifest = _read_manifest()
    require(len(manifest) == 54, f"manifest has {len(manifest)} rows; expected 54")
    manifest_names = set(pd.Series(manifest["repository_path"]).str.replace("data/beats/", "", regex=False))
    beat_names = {p.name for p in beats_files}
    require(
        manifest_names == beat_names,
        f"manifest filename set does not match data/beats/: extra_in_manifest={sorted(manifest_names - beat_names)}, missing_from_manifest={sorted(beat_names - manifest_names)}",
    )
    filename_pattern = re.compile(r"^S(0[1-9]|1[0-8])_(Pre|Stim|Post)\.csv$")
    for p in beats_files:
        require(bool(filename_pattern.fullmatch(p.name)), f"unexpected filename: {p.name}")

    for _, row in manifest.iterrows():
        path = ROOT / row["repository_path"]
        require(path.is_file(), f"missing file listed in manifest: {row['repository_path']}")
        actual_sha = sha256_file(path).lower()
        expected_sha = row["sha256"].lower()
        require(actual_sha == expected_sha, f"SHA-256 mismatch: {row['repository_path']}")
        require(int(row["size_bytes"]) == path.stat().st_size, f"size mismatch: {row['repository_path']}")
        require(int(row["column_count"]) == 4, f"column_count != 4: {row['repository_path']}")
        require(row["header"] == "beat_idx,RRI_ms,SBP_mmHg,PAT_ms", f"header mismatch: {row['repository_path']}")
        raw_frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        require(
            tuple(raw_frame.columns) == EXPECTED_BEATS_HEADER,
            f"in-file header mismatch: {row['repository_path']} -> {tuple(raw_frame.columns)}",
        )
        require(len(raw_frame) == int(row["data_row_count"]), f"row-count mismatch: {row['repository_path']}")
        require(len(raw_frame) >= 100, f"unreasonably short beat table: {row['repository_path']}")
        require(len(raw_frame) <= 2000, f"unreasonably long beat table: {row['repository_path']}")
        for col in EXPECTED_BEATS_HEADER:
            empties = int((raw_frame[col].str.len() == 0).sum())
            require(empties == 0, f"empty field in {col}: {row['repository_path']}")
        numeric = raw_frame.apply(pd.to_numeric, errors="raise")
        for col in EXPECTED_BEATS_HEADER:
            arr = numeric[col].to_numpy(float)
            require(
                bool(np.isfinite(arr).all()),
                f"non-finite value in {col}: {row['repository_path']}",
            )
        beat_idx_float = numeric["beat_idx"].to_numpy(float)
        require(
            bool(np.all(beat_idx_float == np.floor(beat_idx_float))),
            f"beat_idx not integer: {row['repository_path']}",
        )
        beat_idx = beat_idx_float.astype(np.int64)
        require(int(beat_idx[0]) == 1, f"beat_idx does not start at 1: {row['repository_path']}")
        require(
            bool(np.all(np.diff(beat_idx) == 1)),
            f"beat_idx not dense strictly increasing: {row['repository_path']}",
        )
        rri = numeric["RRI_ms"]
        sbp = numeric["SBP_mmHg"]
        require(bool((rri > 0).all()), f"non-positive RRI_ms: {row['repository_path']}")
        require(bool((sbp > 0).all()), f"non-positive SBP_mmHg: {row['repository_path']}")
        require(bool(((rri > 200) & (rri < 3000)).all()), f"RRI_ms outside 200-3000 ms: {row['repository_path']}")
        require(bool(((sbp > 40) & (sbp < 260)).all()), f"SBP_mmHg outside 40-260 mmHg: {row['repository_path']}")
        require(
            row["classification"] == "author_approved_public_pseudonymised_participant_level_derived_beat_table",
            f"classification mismatch: {row['repository_path']}",
        )
    return (
        "54 approved beat tables validated by filename, header, SHA-256, size, row count, "
        "numeric-finite four-column schema, dense integer beat_idx starting at 1, and numeric ranges"
    )


def _placeholder_markers() -> tuple[str, ...]:
    """Return release-text placeholder markers built via string concatenation.

    The individual token pieces are combined at runtime so that no fully
    assembled placeholder literal appears in the validator source itself. That
    way, the validator scanning its own tree does not match on this file. The
    scan is applied case-insensitively.
    """
    def _mk(*parts: str) -> str:
        return "".join(parts)

    return (
        _mk("[", "NEW", " ", "DOI", "]"),
        _mk("[", "GitHub", " ", "repository", "]"),
        _mk("DOI/URL", " ", "placeholder", " ", "pending", " ", "release"),
        _mk("[", "INSERT", " ", "DOI", "]"),
        _mk("[", "PLACEHOLDER", "]"),
        _mk("[", "TODO", "]"),
        _mk("[", "TBD", "]"),
        _mk("[", "FIXME", "]"),
        _mk("<", "DOI", "-", "TBD", ">"),
        _mk("<", "DOI-", "PENDING", ">"),
        _mk("XXX", "XXX"),
        _mk("YYYY", "-MM", "-DD"),
    )


def check_public_tree_safety() -> str:
    excluded_dirs = {".git"}
    relative_files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ROOT).parts
        if any(part in excluded_dirs for part in rel_parts):
            continue
        relative_files.append(path.relative_to(ROOT).as_posix())

    approved_beat_paths = {f"data/beats/{p.name}" for p in BEATS_DIR.iterdir() if p.is_file()}
    approved_top_data = {"data/LICENSE", "data/README.md"} | approved_beat_paths

    raw_binary_pattern = re.compile(
        r"\.(?:acq|edf|mat|pem|key|env|npy|npz|h5|hdf5|wav|bin|dat|xdf|set|eeg|cnt|vhdr|vmrk)$"
        r"|paired_beats_\d+\.csv$",
        flags=re.IGNORECASE,
    )
    data_prefix_pattern = re.compile(r"^data/(.+)$", flags=re.IGNORECASE)
    forbidden_paths: list[str] = []
    for path in relative_files:
        if raw_binary_pattern.search(path):
            forbidden_paths.append(path)
            continue
        m = data_prefix_pattern.match(path)
        if m is not None and path not in approved_top_data:
            forbidden_paths.append(path)
    require(not forbidden_paths, f"restricted or unapproved paths present: {forbidden_paths}")

    local_path_markers = (
        "C:" + "\\Users\\",
        "/mnt" + "/data",
        "sand" + "box:",
    )
    credential_pattern = re.compile(
        r"(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^<'\"\s]{8,}",
        flags=re.IGNORECASE,
    )
    identifier_pattern = re.compile(
        r"\b(?:MRN|patient_id|dob|date_of_birth|birthdate|ssn|jrct[a-z0-9]*)\b",
        flags=re.IGNORECASE,
    )
    placeholder_markers = _placeholder_markers()
    local_hits: list[str] = []
    credential_hits: list[str] = []
    identifier_hits: list[str] = []
    placeholder_hits: list[str] = []
    binary_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf", ".ico"}
    for relative in relative_files:
        path = ROOT / relative
        if path.suffix.lower() in binary_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in local_path_markers):
            local_hits.append(relative)
        if credential_pattern.search(text):
            credential_hits.append(relative)
        if relative.startswith("data/beats/") and identifier_pattern.search(text):
            identifier_hits.append(relative)
        lowered = text.lower()
        for marker in placeholder_markers:
            if marker.lower() in lowered:
                placeholder_hits.append(f"{relative}::{marker}")
                break
    require(not local_hits, f"absolute/local path markers present: {local_hits}")
    require(not credential_hits, f"credential-like assignments present: {credential_hits}")
    require(not identifier_hits, f"direct-identifier terms in public beat data: {identifier_hits}")
    require(not placeholder_hits, f"release-text placeholder markers present: {placeholder_hits}")
    return (
        "no restricted files, host paths, environment files, credential assignments, "
        "direct identifiers in public data, or release-text placeholder markers "
        "(.git internals excluded from scan)"
    )


def check_public_filenames() -> str:
    forbidden_tokens = (
        "v" + "10",
        "v" + "11",
        "v" + "12",
        "v" + "12_s3",
        "_" + "final",
        "_" + "revised",
        "_" + "codex",
        "2026" + "08",
        "(" + "1)",
        "(" + "2)",
        "co" + "py",
        "inter" + "im",
        "sand" + "box",
        "tai" + "naka",
    )
    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ROOT).parts
        if ".git" in rel_parts:
            continue
        relative = path.relative_to(ROOT).as_posix().lower()
        if any(token.lower() in relative for token in forbidden_tokens):
            failures.append(relative)
    require(not failures, f"forbidden public filename tokens: {failures}")
    return "public filenames contain no internal manuscript-version or host tokens"


def main() -> int:
    anchors = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA3_PATH, dtype=str, keep_default_na=False)
    checks: list[tuple[str, Callable[[], str]]] = [
        ("syntax", check_syntax),
        ("public_artifact_hashes", lambda: check_public_artifact_hashes(anchors)),
        ("central_brs", lambda: check_central_brs(frame, anchors)),
        ("brs_landscape", lambda: check_brs_landscape(frame, anchors)),
        ("coherence", lambda: check_coherence(frame, anchors)),
        ("haemodynamics", lambda: check_haemodynamics(frame, anchors)),
        ("native_var", lambda: check_native_var(frame, anchors)),
        ("methods_text_matched_brs", check_methods_matched_brs),
        ("nonlinear_sensitivity", check_nonlinear_sensitivity),
        ("metadata", check_metadata),
        ("public_data_boundary", lambda: check_public_data_boundary(frame)),
        ("author_approved_derived_beats", check_author_approved_derived_beats),
        ("s3_render", check_s3_render),
        ("public_tree_safety", check_public_tree_safety),
        ("public_filenames", check_public_filenames),
    ]
    passed = 0
    failures: list[str] = []
    for name, function in checks:
        try:
            detail = function()
        except Exception as error:  # noqa: BLE001 - report every gate failure
            failures.append(f"{name}: {error}")
            print(f"FAIL {name}: {error}")
        else:
            passed += 1
            print(f"PASS {name}: {detail}")
    print(f"VALIDATION_PASS_COUNT={passed}")
    print(f"VALIDATION_TOTAL_COUNT={len(checks)}")
    if failures:
        print("PUBLIC_RELEASE_VALIDATION_FAIL")
        return 1
    print("PUBLIC_RELEASE_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
