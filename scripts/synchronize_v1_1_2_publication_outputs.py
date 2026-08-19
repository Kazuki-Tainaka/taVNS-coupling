"""Synchronize reviewed v1.1.2 publication outputs from pinned sources.

The two Supplementary Data targets are the repository's publication copies
of the final revised-submission CSVs, so byte copying is appropriate only
after their source hashes and table semantics have passed the checks below.
The methods-text-matched REF row is populated from the exact central
participant-level BCa aggregate, never from rounded manuscript values or an
independent bootstrap run.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "v1_1_2_publication_output_sync.json"
SUBJECT_PATTERN = re.compile(r"^S(?:0[1-9]|1[0-8])$")


def sha256_file(path: Path) -> str:
    """Return the lowercase raw-byte SHA-256 of *path*."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Raise an actionable validation error when *condition* is false."""
    if not condition:
        raise ValueError(message)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV without converting explicit ``NA`` strings."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"missing header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def one_row(
    rows: list[dict[str, str]],
    label: str,
    **criteria: str,
) -> dict[str, str]:
    """Select exactly one row matching all string-valued criteria."""
    selected = [
        row
        for row in rows
        if all(row.get(column) == value for column, value in criteria.items())
    ]
    require(len(selected) == 1, f"{label}: expected one row, found {len(selected)}")
    return selected[0]


def validate_table_shape(
    fields: list[str],
    rows: list[dict[str, str]],
    expected_rows: int,
    expected_columns: int,
    label: str,
) -> None:
    """Validate table dimensions and explicit-cell semantics."""
    require(len(rows) == expected_rows, f"{label}: row count {len(rows)}")
    require(len(fields) == expected_columns, f"{label}: column count {len(fields)}")
    require(len(fields) == len(set(fields)), f"{label}: duplicate header")
    for row_number, row in enumerate(rows, start=2):
        require(None not in row, f"{label}: malformed row {row_number}")
        for column in fields:
            value = row[column]
            require(value != "", f"{label}: blank cell {row_number}/{column}")
            require(value == value.strip(), f"{label}: whitespace {row_number}/{column}")


def validate_data_1(
    fields: list[str],
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> None:
    """Validate the reviewed Data 1 family and central metadata fixes."""
    validate_table_shape(
        fields,
        rows,
        int(config["row_count"]),
        int(config["column_count"]),
        "Supplementary Data 1",
    )
    require(
        len({row["source_metric"] for row in rows}) == len(rows),
        "Supplementary Data 1: source_metric is not unique",
    )
    brs_tf = one_row(rows, "BRS_TF_mean", source_metric="BRS_TF_mean")
    for column, expected in config["brs_tf_expected"].items():
        require(
            brs_tf[column] == expected,
            f"BRS_TF_mean {column}: expected {expected}, got {brs_tf[column]}",
        )

    for suffix, expected_size in (
        ("Stim_Pre", 46),
        ("Post_Pre", 46),
        ("Post_Stim", int(config["post_stim_bh_family_size"])),
    ):
        nominal = [row for row in rows if row[f"p_{suffix}"] != "NA"]
        adjusted = [row for row in rows if row[f"p_FDR_{suffix}"] != "NA"]
        require(len(nominal) == expected_size, f"{suffix}: nominal family size")
        require(len(adjusted) == expected_size, f"{suffix}: BH family size")
        require(
            {row["source_metric"] for row in nominal}
            == {row["source_metric"] for row in adjusted},
            f"{suffix}: nominal/BH eligibility mismatch",
        )

    coherence = one_row(rows, "Coh_mean", source_metric="Coh_mean")
    require(coherence["p_FDR_Post_Stim"] == "0.954", "Coh_mean Post-Stim q")
    require(coherence["Friedman_chi2"] == "1.33333", "Coh_mean Friedman chi2")
    require(coherence["Friedman_p"] == "0.513", "Coh_mean Friedman p")
    central_brs = one_row(rows, "BRS_seq_all", source_metric="BRS_seq_all")
    require(central_brs["Friedman_chi2"] == "12.3333", "BRS Friedman chi2")
    require(central_brs["Friedman_p"] == "0.0021", "BRS Friedman p")


def validate_data_3(
    fields: list[str],
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> None:
    """Validate the reviewed Data 3 traceability and degenerate-Q fixes."""
    validate_table_shape(
        fields,
        rows,
        int(config["row_count"]),
        int(config["column_count"]),
        "Supplementary Data 3",
    )
    require(
        fields.index("excluded_subject_id") == fields.index("subject_id") + 1,
        "excluded_subject_id must immediately follow subject_id",
    )

    leave_one_out = [row for row in rows if row["record_type"] == "leave_one_out"]
    require(len(leave_one_out) == 36, "leave-one-out row count")
    excluded_counts = {
        subject: sum(row["excluded_subject_id"] == subject for row in leave_one_out)
        for subject in (f"S{index:02d}" for index in range(1, 19))
    }
    require(set(excluded_counts.values()) == {2}, "leave-one-out exclusion completeness")
    require(
        all(
            row["excluded_subject_id"] == "NA"
            for row in rows
            if row["record_type"] != "leave_one_out"
        ),
        "excluded_subject_id must be structural NA outside leave-one-out rows",
    )

    non_na_subjects = {row["subject_id"] for row in rows if row["subject_id"] != "NA"}
    require(
        non_na_subjects == {f"S{index:02d}" for index in range(1, 19)},
        "participant identifier set",
    )
    require(all(SUBJECT_PATTERN.fullmatch(value) for value in non_na_subjects), "ID format")

    defined_metrics = {
        "BEI": "0",
        "n_qualifying_brs_sequences": "0",
        "no_valid_sequence_indicator": "True",
    }
    undefined_metrics = {
        "mean_rri_response_abs_ms",
        "mean_sbp_ramp_amplitude_abs_mmHg",
        "mean_sequence_length_beats",
        "mean_sequence_slope_ms_per_mmHg",
        "mean_within_sequence_r",
    }
    for subject in ("S06", "S12", "S17"):
        base = {
            "analysis_family": "sequence_quality",
            "subject_id": subject,
            "phase": "Stim",
            "direction": "down",
        }
        for metric, expected_value in defined_metrics.items():
            row = one_row(rows, f"{subject}/{metric}", metric=metric, **base)
            require(row["value"] == expected_value, f"{subject}/{metric} value")
            require(row["status"] == "estimable", f"{subject}/{metric} status")
            require(row["NA_reason"] == "NA", f"{subject}/{metric} NA_reason")
        ramps = one_row(rows, f"{subject}/n_sbp_ramps", metric="n_sbp_ramps", **base)
        require(int(ramps["value"]) > 0, f"{subject}/n_sbp_ramps value")
        require(ramps["status"] == "estimable", f"{subject}/n_sbp_ramps status")
        require(ramps["NA_reason"] == "NA", f"{subject}/n_sbp_ramps NA_reason")
        for metric in undefined_metrics:
            row = one_row(rows, f"{subject}/{metric}", metric=metric, **base)
            require(row["value"] == "NA", f"{subject}/{metric} value")
            require(row["status"] == "not_estimable", f"{subject}/{metric} status")
            require(
                row["NA_reason"] == "no_ramps_met_correlation_threshold",
                f"{subject}/{metric} NA_reason",
            )

    coherence = one_row(
        rows,
        "coherence Stim-vs-Pre",
        analysis_family="coherence_significance",
        record_type="prevalence_comparison",
        contrast="Stim-vs-Pre",
    )
    require(coherence["discordant_pre_only"] == "2", "coherence Pre-only count")
    require(coherence["discordant_stim_only"] == "3", "coherence Stim-only count")
    require(coherence["p_value"] == "1", "coherence McNemar p")

    participant = [
        row
        for row in rows
        if row["analysis_family"] == "GC_significance_native_paired_beats"
        and row["record_type"] == "participant_phase_direction"
        and row["direction"] == "past_RRI_to_SBP"
    ]
    require(len(participant) == 54, "past-RRI-to-SBP participant matrix size")
    require(
        len({(row["subject_id"], row["phase"]) for row in participant}) == 54,
        "past-RRI-to-SBP participant matrix duplicates",
    )
    require(
        all(row["nominal_significant"] == "True" for row in participant),
        "nominal matrix is not all True",
    )
    require(
        all(row["fdr_significant"] == "True" for row in participant),
        "FDR matrix is not all True",
    )

    q_rows = [
        row
        for row in rows
        if row["analysis_family"] == "GC_significance_native_paired_beats"
        and row["record_type"] == "paired_prevalence_comparison"
        and row["direction"] == "past_RRI_to_SBP"
        and row["contrast"] == "Pre-Stim-Post"
        and row["test"] == "Cochran_Q_chi_square"
    ]
    require(len(q_rows) == 2, "degenerate Cochran Q row count")
    require(
        {row["outcome"] for row in q_rows}
        == {"nominal_significant", "fdr_significant"},
        "degenerate Cochran Q outcomes",
    )
    for row in q_rows:
        for column in ("value", "statistic", "p_value", "exact_or_asymptotic"):
            require(row[column] == "NA", f"degenerate Q {row['outcome']}/{column}")
        require(row["n"] == "18", f"degenerate Q {row['outcome']}/n")
        require(row["status"] == "not_estimable", "degenerate Q status")
        require(
            row["NA_reason"] == "no_within_participant_variation_across_phases",
            "degenerate Q reason",
        )

    mcnemar = [
        row
        for row in rows
        if row["analysis_family"] == "GC_significance_native_paired_beats"
        and row["record_type"] == "paired_prevalence_comparison"
        and row["direction"] == "past_RRI_to_SBP"
        and row["contrast"] == "Stim-vs-Pre"
        and row["test"] == "McNemar_exact_conditional_binomial"
    ]
    require(len(mcnemar) == 2, "McNemar row count")
    require(
        all(row["p_value"] == "1" and row["status"] == "estimable" for row in mcnemar),
        "McNemar p=1 estimability",
    )

    forbidden_source_fragments = (
        "/mnt" + "/data",
        "sand" + "box",
        chr(92) + "users" + chr(92),
    )
    for row in rows:
        lowered = row["source_file"].lower()
        require(
            not any(fragment in lowered for fragment in forbidden_source_fragments),
            "local or environment-specific source_file value",
        )
        require(
            not re.match(r"^[a-z]:[\\/]", row["source_file"], flags=re.IGNORECASE),
            "absolute source_file value",
        )


def comparator_rows_sha256(
    fields: list[str],
    rows: list[dict[str, str]],
) -> str:
    """Hash non-REF methods rows in stable field order."""
    payload = "\n".join(
        ",".join(row[field] for field in fields)
        for row in rows
        if row["branch"] != "REF"
    ) + "\n"
    return sha256(payload.encode("utf-8")).hexdigest()


def validate_canonical_brs(
    fields: list[str],
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, str]:
    """Return the exact central REF source row after validating the table."""
    validate_table_shape(
        fields,
        rows,
        int(config["row_count"]),
        int(config["column_count"]),
        "canonical BRS contrasts",
    )
    reference = one_row(
        rows,
        "canonical BRS REF",
        metric=str(config["reference_metric"]),
        contrast=str(config["reference_contrast"]),
    )
    require(reference["n"] == "18", "canonical BRS n")
    require(reference["n_negative"] == "17", "canonical BRS negative count")
    require(reference["bootstrap_method"] == "participant_level_BCa", "BCa method")
    require(reference["bootstrap_resamples"] == "10000", "BCa resamples")
    require(reference["bootstrap_seed_mean"] == "20260805", "BCa mean seed")
    require(
        reference["mean_difference_ci_low"] == "-3.8125937386490705"
        and reference["mean_difference_ci_high"] == "-1.09512491343104",
        "canonical BRS exact BCa interval",
    )
    return reference


def validate_methods_summary(
    path: Path,
    canonical_reference: dict[str, str],
    config: dict[str, Any],
) -> None:
    """Validate REF provenance and byte-stable comparator branches."""
    fields, rows = read_rows(path)
    require(len(rows) == 5 and len(fields) == 12, "methods summary shape")
    require(
        comparator_rows_sha256(fields, rows) == config["comparator_rows_sha256"],
        "methods comparator branches changed",
    )
    reference = one_row(rows, "methods REF", branch="REF")
    for target_field, source_field in config["field_mapping"].items():
        require(
            reference[target_field] == canonical_reference[source_field],
            f"methods REF provenance mismatch: {target_field}",
        )
    require(reference["estimable"] == "True", "methods REF estimability")
    require(reference["NA_reason"] == "NA", "methods REF NA_reason")
    require(sha256_file(path) == config["target_sha256"], "methods summary SHA-256")


def copy_atomically(source: Path, target: Path) -> None:
    """Copy source bytes to target using a same-directory atomic replace."""
    temporary = target.with_name(f".{target.name}.v1_1_2_sync_tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def write_methods_summary(
    path: Path,
    canonical_reference: dict[str, str],
    config: dict[str, Any],
) -> None:
    """Replace only REF fields that map to the canonical central result."""
    fields, rows = read_rows(path)
    before_comparators = comparator_rows_sha256(fields, rows)
    reference = one_row(rows, "methods REF", branch="REF")
    for target_field, source_field in config["field_mapping"].items():
        reference[target_field] = canonical_reference[source_field]

    temporary = path.with_name(f".{path.name}.v1_1_2_sync_tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)

    after_fields, after_rows = read_rows(path)
    require(after_fields == fields, "methods summary column order changed")
    require(
        comparator_rows_sha256(after_fields, after_rows) == before_comparators,
        "methods comparator branches changed during REF synchronization",
    )


def verify_source(path: Path, expected_hash: str, label: str) -> None:
    """Verify a pinned external source before it can affect the tree."""
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected_hash, f"{label} SHA-256 mismatch: {actual}")


def verify_repository_outputs(config: dict[str, Any]) -> dict[str, str]:
    """Validate every synchronized in-repository output."""
    data_1_config = config["supplementary_data_1"]
    data_1_path = ROOT / data_1_config["repository_target"]
    require(sha256_file(data_1_path) == data_1_config["target_sha256"], "Data 1 SHA-256")
    data_1_fields, data_1_rows = read_rows(data_1_path)
    validate_data_1(data_1_fields, data_1_rows, data_1_config)

    data_3_config = config["supplementary_data_3"]
    data_3_path = ROOT / data_3_config["repository_target"]
    require(sha256_file(data_3_path) == data_3_config["target_sha256"], "Data 3 SHA-256")
    data_3_fields, data_3_rows = read_rows(data_3_path)
    validate_data_3(data_3_fields, data_3_rows, data_3_config)

    canonical_config = config["canonical_brs_contrasts"]
    canonical_path = ROOT / canonical_config["repository_target"]
    require(
        sha256_file(canonical_path) == canonical_config["target_sha256"],
        "canonical BRS SHA-256",
    )
    canonical_fields, canonical_rows = read_rows(canonical_path)
    reference = validate_canonical_brs(
        canonical_fields,
        canonical_rows,
        canonical_config,
    )

    methods_config = config["methods_text_matched_brs_summary"]
    methods_path = ROOT / methods_config["repository_target"]
    validate_methods_summary(methods_path, reference, methods_config)
    return {
        "supplementary_data_1_sha256": sha256_file(data_1_path),
        "supplementary_data_3_sha256": sha256_file(data_3_path),
        "canonical_brs_contrasts_sha256": sha256_file(canonical_path),
        "methods_text_matched_brs_summary_sha256": sha256_file(methods_path),
    }


def apply_sync(
    config: dict[str, Any],
    data_1_source: Path,
    data_3_source: Path,
    canonical_source: Path,
) -> dict[str, str]:
    """Validate pinned sources, synchronize targets, and verify results."""
    data_1_config = config["supplementary_data_1"]
    data_3_config = config["supplementary_data_3"]
    canonical_config = config["canonical_brs_contrasts"]
    verify_source(
        data_1_source,
        data_1_config["submission_source_sha256"],
        "Supplementary Data 1 source",
    )
    verify_source(
        data_3_source,
        data_3_config["submission_source_sha256"],
        "Supplementary Data 3 source",
    )
    verify_source(
        canonical_source,
        canonical_config["submission_source_sha256"],
        "canonical BRS source",
    )

    data_1_fields, data_1_rows = read_rows(data_1_source)
    validate_data_1(data_1_fields, data_1_rows, data_1_config)
    data_3_fields, data_3_rows = read_rows(data_3_source)
    validate_data_3(data_3_fields, data_3_rows, data_3_config)
    canonical_fields, canonical_rows = read_rows(canonical_source)
    reference = validate_canonical_brs(
        canonical_fields,
        canonical_rows,
        canonical_config,
    )

    data_1_target = ROOT / data_1_config["repository_target"]
    existing_data_1_fields, existing_data_1_rows = read_rows(data_1_target)
    require(existing_data_1_fields == data_1_fields, "Data 1 target schema mismatch")
    require(len(existing_data_1_rows) == len(data_1_rows), "Data 1 target row mismatch")

    data_3_target = ROOT / data_3_config["repository_target"]
    existing_data_3_fields, existing_data_3_rows = read_rows(data_3_target)
    allowed_existing_headers = (
        data_3_fields,
        [field for field in data_3_fields if field != "excluded_subject_id"],
    )
    require(
        existing_data_3_fields in allowed_existing_headers,
        "Data 3 target schema is not the reviewed v1.1.1 or v1.1.2 schema",
    )
    require(len(existing_data_3_rows) == len(data_3_rows), "Data 3 target row mismatch")

    copy_atomically(data_1_source, data_1_target)
    copy_atomically(data_3_source, data_3_target)
    canonical_target = ROOT / canonical_config["repository_target"]
    copy_atomically(canonical_source, canonical_target)
    write_methods_summary(
        ROOT / config["methods_text_matched_brs_summary"]["repository_target"],
        reference,
        config["methods_text_matched_brs_summary"],
    )
    return verify_repository_outputs(config)


def parse_args() -> argparse.Namespace:
    """Parse explicit apply/verify modes and external source locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true", help="apply the pinned synchronization")
    mode.add_argument("--verify", action="store_true", help="verify repository outputs only")
    parser.add_argument("--supplementary-data-1-source", type=Path)
    parser.add_argument("--supplementary-data-3-source", type=Path)
    parser.add_argument("--canonical-brs-source", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the selected synchronization mode."""
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.apply:
        require(
            args.supplementary_data_1_source is not None
            and args.supplementary_data_3_source is not None
            and args.canonical_brs_source is not None,
            "--apply requires all three explicit source paths",
        )
        hashes = apply_sync(
            config,
            args.supplementary_data_1_source,
            args.supplementary_data_3_source,
            args.canonical_brs_source,
        )
    else:
        hashes = verify_repository_outputs(config)
    print(json.dumps({"status": "PASS", **hashes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
