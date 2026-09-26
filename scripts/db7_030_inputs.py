"""Verify the frozen trace whether Kaggle keeps gzip or decompresses it."""
from pathlib import Path
import gzip
import hashlib


def canonical_trace_bytes(path):
    raw = Path(path).read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    return raw.replace(b"\r\n", b"\n")


def find_verified_trace(root, expected_sha256):
    root = Path(root)
    candidates = sorted(p for p in root.rglob("db7-030-si-i-trace*")
                        if p.is_file() and p.name.endswith((".csv", ".csv.gz")))
    print("Frozen trace candidate files:", [str(p) for p in candidates], flush=True)
    if not candidates:
        nearby = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                        if p.is_file() and "trace" in str(p).lower())
        raise FileNotFoundError(f"No CSV or CSV.GZ DB7-016 trace under {root}; trace inventory={nearby[:50]}")
    verified = []
    for path in candidates:
        digest = hashlib.sha256(canonical_trace_bytes(path)).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"Frozen trace content checksum mismatch: {path}")
        verified.append(path)
    # Kaggle may expose a source through more than one mount. Identical verified
    # copies are acceptable; their contents are the exact same prediction trace.
    return verified[0]
