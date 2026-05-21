from pathlib import Path
import hashlib

CANONICAL_HASHES = {
    "results/Additional_File_2.csv": "474f5e1792065b62b5711830ad585d95",
    "results/Additional_File_3.csv": "c6bf4816b45165ef24a458a151c50d54",
}


def test_md5_match():
    for path, expected in CANONICAL_HASHES.items():
        assert hashlib.md5(Path(path).read_bytes()).hexdigest() == expected
