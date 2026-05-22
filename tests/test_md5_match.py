from pathlib import Path
import hashlib

CANONICAL_HASHES = {
    "data/reference/Additional_File_2.csv": "474f5e1792065b62b5711830ad585d95",
    "data/reference/Additional_File_3.csv": "5fd37ceb5269c0558131a02efbb6ba95",
    "results/Additional_File_2.csv": "474f5e1792065b62b5711830ad585d95",
    "results/Additional_File_3.csv": "5fd37ceb5269c0558131a02efbb6ba95",
}


def test_md5_match():
    for path, expected in CANONICAL_HASHES.items():
        assert hashlib.md5(Path(path).read_bytes()).hexdigest() == expected
