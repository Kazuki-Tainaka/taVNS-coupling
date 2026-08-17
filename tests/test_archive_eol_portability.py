"""Archive-level EOL portability regression tests (v1.1.1 hotfix).

These tests exercise the extracted GitHub source archive integrity gate
described in ``docs/EOL_POLICY.md``. They are deliberately independent of
``.git``: they build a synthetic GitHub-style ZIP archive (one top-level
directory, mirroring GitHub's ``<repo>-<sha>/`` layout) from the current
in-tree bytes, extract it, verify the extraction inventory round-trips
exactly, and then re-run the hotfix validator's
``check_public_artifact_hashes(...)`` against the extracted tree.

Two archive variants are exercised for the four reviewed repository-authored
text anchors:

1. **LF variant** — text bytes written with LF line endings only.
2. **CRLF variant** — text bytes written with CRLF line endings.

Both variants MUST pass the artefact-hash gate, proving the
``eol_normalized_lf_bytes`` hash mode is truly EOL-independent.

Simultaneously, the tests prove that the raw-byte hash mode used for
``data/beats/*.csv`` is NOT EOL-normalised: mutating a beat table's line
endings changes its raw SHA-256, and the raw-mode helper detects the
mutation while the EOL-normalising helper would erase it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
VALIDATOR_PATH = SCRIPTS_DIR / "validate_public_release.py"
ANCHORS_PATH = REPO_ROOT / "config" / "release_anchors.json"

FAKE_TOP_LEVEL = "taVNS-coupling-0000000000000000000000000000000000000000"

RELATIVE_ARTIFACTS = {
    "supplementary_data_1.csv":
        "expected_outputs/publication_source_data/supplementary_data_1.csv",
    "supplementary_data_2.csv":
        "expected_outputs/publication_source_data/supplementary_data_2.csv",
    "supplementary_data_3.csv":
        "expected_outputs/publication_source_data/"
        "supplementary_data_3_brs_sensitivity_and_coupling_significance.csv",
    "supplementary_figure_s3.jpg":
        "figures/supplementary_figure_s3_brs_specification_landscape.jpg",
    "supplementary_figure_s3_generator.py":
        "scripts/generate_supplementary_figure_s3.py",
}

TEXT_ANCHOR_NAMES = frozenset(
    {
        "supplementary_data_1.csv",
        "supplementary_data_2.csv",
        "supplementary_data_3.csv",
        "supplementary_figure_s3_generator.py",
    }
)

BEAT_SAMPLE_RELATIVE = "data/beats/S01_Pre.csv"


def _load_validator_module():
    """Import scripts/validate_public_release.py without relying on .git."""
    spec = importlib.util.spec_from_file_location(
        "tavns_hotfix_validate_public_release", str(VALIDATOR_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _to_crlf(data: bytes) -> bytes:
    return _to_lf(data).replace(b"\n", b"\r\n")


def _load_anchors() -> dict:
    return json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))


def _build_archive_bytes(text_eol: str) -> tuple[bytes, dict[str, bytes]]:
    """Return (zip_bytes, expected_inventory).

    ``text_eol`` selects the line-ending encoding written into the archive
    for the four reviewed text anchors: ``"lf"`` or ``"crlf"``. Binary
    artefacts and the sampled beat table are always written with raw bytes.
    """
    assert text_eol in ("lf", "crlf")
    inventory: dict[str, bytes] = {}
    # Reviewed public artefacts.
    for name, relative in RELATIVE_ARTIFACTS.items():
        source = REPO_ROOT / relative
        raw = source.read_bytes()
        if name in TEXT_ANCHOR_NAMES:
            payload = _to_crlf(raw) if text_eol == "crlf" else _to_lf(raw)
        else:
            payload = raw
        inventory[relative] = payload
    # A representative beat table: proves data/beats/*.csv are hashed raw.
    beat_path = REPO_ROOT / BEAT_SAMPLE_RELATIVE
    inventory[BEAT_SAMPLE_RELATIVE] = beat_path.read_bytes()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # GitHub source archives place everything under a single top-level dir.
        zf.writestr(f"{FAKE_TOP_LEVEL}/", b"")
        for relative, payload in inventory.items():
            arcname = f"{FAKE_TOP_LEVEL}/{relative}"
            zf.writestr(arcname, payload)
    return buffer.getvalue(), inventory


def _extract_to(tmp: Path, archive_bytes: bytes) -> Path:
    archive_path = tmp / "source_archive.zip"
    archive_path.write_bytes(archive_bytes)
    extract_dir = tmp / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)
    top_children = list(extract_dir.iterdir())
    assert len(top_children) == 1, (
        f"expected exactly one top-level directory in the archive, "
        f"found: {[c.name for c in top_children]}"
    )
    assert top_children[0].is_dir()
    assert top_children[0].name == FAKE_TOP_LEVEL
    return top_children[0]


def _extracted_inventory(extract_root: Path) -> dict[str, bytes]:
    inventory: dict[str, bytes] = {}
    for path in extract_root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(extract_root).as_posix()
            inventory[rel] = path.read_bytes()
    return inventory


def _run_artifact_check(root: Path) -> str:
    module = _load_validator_module()
    anchors = _load_anchors()
    # Positional call preserves the CLI signature; supplying an alternate
    # root proves the check is archive-portable.
    return module.check_public_artifact_hashes(anchors, root=root)


def _assert_inventory_roundtrip(expected: dict[str, bytes], actual: dict[str, bytes]) -> None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    assert not missing, f"extraction missing files: {missing}"
    assert not extra, f"extraction produced unexpected files: {extra}"
    for rel in sorted(expected):
        assert expected[rel] == actual[rel], f"byte mismatch in extraction: {rel}"


def test_lf_variant_extracted_archive_passes_artifact_hash_gate():
    archive_bytes, inventory = _build_archive_bytes("lf")
    with tempfile.TemporaryDirectory(prefix="tavns_archive_lf_") as tmp:
        extract_root = _extract_to(Path(tmp), archive_bytes)
        _assert_inventory_roundtrip(inventory, _extracted_inventory(extract_root))
        detail = _run_artifact_check(extract_root)
        assert "reviewed public artifact hashes match" in detail


def test_crlf_variant_extracted_archive_passes_artifact_hash_gate():
    archive_bytes, inventory = _build_archive_bytes("crlf")
    with tempfile.TemporaryDirectory(prefix="tavns_archive_crlf_") as tmp:
        extract_root = _extract_to(Path(tmp), archive_bytes)
        _assert_inventory_roundtrip(inventory, _extracted_inventory(extract_root))
        # Sanity: at least one text anchor actually contains CRLF on disk.
        one_text = extract_root / RELATIVE_ARTIFACTS["supplementary_data_1.csv"]
        assert b"\r\n" in one_text.read_bytes(), (
            "CRLF variant did not produce CRLF bytes on disk; "
            "the portability regression test setup is broken."
        )
        detail = _run_artifact_check(extract_root)
        assert "reviewed public artifact hashes match" in detail


def test_beat_table_raw_hash_mode_is_not_eol_normalized():
    """A ``data/beats/*.csv`` participant table is hashed raw. Mutating its
    line endings must change its raw SHA-256 (i.e., we must not silently
    normalise beat bytes)."""
    module = _load_validator_module()
    beat_path = REPO_ROOT / BEAT_SAMPLE_RELATIVE
    original = beat_path.read_bytes()
    lf_bytes = _to_lf(original)
    crlf_bytes = _to_crlf(original)

    raw_lf = hashlib.sha256(lf_bytes).hexdigest().upper()
    raw_crlf = hashlib.sha256(crlf_bytes).hexdigest().upper()
    # Beat tables must be LF (declared ``-text binary``); if this assertion
    # fires, the repository has drifted from the v1.1.0 byte-identical guarantee.
    assert lf_bytes == original, (
        "beat file bytes drifted from LF-only storage; "
        "raw SHA / blob OID guarantees would be violated"
    )
    assert raw_lf != raw_crlf, (
        "test setup broken: LF and CRLF variants of the beat table "
        "unexpectedly produced the same raw SHA"
    )

    with tempfile.TemporaryDirectory(prefix="tavns_beat_raw_") as tmp:
        mutated = Path(tmp) / "S01_Pre.csv"
        mutated.write_bytes(crlf_bytes)
        # sha256_file must return the raw SHA of the CRLF-mutated bytes:
        # i.e., it must NOT normalise. That means it differs from the LF SHA.
        assert module.sha256_file(mutated) == raw_crlf
        assert module.sha256_file(mutated) != raw_lf
        # The EOL-normalising helper would erase the mutation. It is only
        # applied to the four reviewed text anchors, never to beat tables.
        assert module.sha256_file_eol_normalized(mutated) == raw_lf


def test_hash_helpers_agree_with_hash_mode_declarations():
    """The four text anchors are declared ``eol_normalized_lf_bytes`` and the
    JPG is declared ``raw_bytes``. Confirm the on-disk in-tree hashes agree
    with the declared modes so the archive tests can trust the same helpers."""
    module = _load_validator_module()
    anchors = _load_anchors()
    hashes = anchors["public_artifact_sha256"]
    for name, relative in RELATIVE_ARTIFACTS.items():
        path = REPO_ROOT / relative
        entry = hashes[name]
        assert isinstance(entry, dict), (
            f"v1.1.1 anchor schema requires nested object for {name}"
        )
        expected = entry["sha256"].upper()
        mode = entry["mode"]
        if name in TEXT_ANCHOR_NAMES:
            assert mode == module.HASH_MODE_EOL_LF
            assert module.sha256_file_eol_normalized(path) == expected
        else:
            assert mode == module.HASH_MODE_RAW
            assert module.sha256_file(path) == expected


def test_gitattributes_declares_beats_as_binary_with_eol_explicitly_unset():
    """The `.gitattributes` file must declare `data/beats/*.csv` with
    both `binary` and an explicit `-eol` token, so the default
    `* text=auto eol=lf` policy cannot leave `eol=lf` set on byte-frozen
    paths. This is verified by reading the file directly; no `.git`
    machinery is required."""
    attributes_path = REPO_ROOT / ".gitattributes"
    lines = attributes_path.read_text(encoding="utf-8").splitlines()
    beat_rules = [
        stripped.split()
        for stripped in (line.strip() for line in lines)
        if stripped and not stripped.startswith("#")
        and stripped.split()[0] == "data/beats/*.csv"
    ]
    assert beat_rules, "no explicit rule found for data/beats/*.csv"
    tokens = beat_rules[0][1:]
    assert "binary" in tokens, (
        f"data/beats/*.csv rule must include `binary` token, got: {tokens}"
    )
    assert "-eol" in tokens, (
        "data/beats/*.csv rule must include an explicit `-eol` token so the "
        "default `eol=lf` from `* text=auto eol=lf` cannot leak onto "
        f"byte-frozen beat tables; got: {tokens}"
    )


def test_outer_archive_zip_sha_is_documented_as_non_gating():
    """The outer container SHA of a GitHub source archive is explicitly not
    used as a hard gate. Two structurally identical archive builds with the
    same inventory MAY hash differently; the extracted per-file inventory is
    the auditable object."""
    archive_a, inventory_a = _build_archive_bytes("lf")
    archive_b, inventory_b = _build_archive_bytes("lf")
    assert inventory_a == inventory_b
    # We do not assert that hashlib.sha256(archive_a).digest() ==
    # hashlib.sha256(archive_b).digest(); the point is that the per-file
    # inventory is what the hotfix validates, per docs/EOL_POLICY.md #4.
    with tempfile.TemporaryDirectory(prefix="tavns_archive_outer_") as tmp:
        sub_a = Path(tmp) / "a"
        sub_a.mkdir()
        sub_b = Path(tmp) / "b"
        sub_b.mkdir()
        root_a = _extract_to(sub_a, archive_a)
        root_b = _extract_to(sub_b, archive_b)
        _assert_inventory_roundtrip(inventory_a, _extracted_inventory(root_a))
        _assert_inventory_roundtrip(inventory_b, _extracted_inventory(root_b))
        assert _run_artifact_check(root_a).startswith("5 reviewed")
        assert _run_artifact_check(root_b).startswith("5 reviewed")
