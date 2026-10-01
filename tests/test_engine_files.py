import hashlib
from pathlib import Path

import pytest

from noface.engine import _Model, ensure_model, file_sha256, is_lfs_pointer
from noface.errors import NofaceError


def test_lfs_pointer_and_checksum(tmp_path: Path) -> None:
    pointer = tmp_path / "model.onnx"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    assert is_lfs_pointer(pointer)
    payload = tmp_path / "weights.bin"
    payload.write_bytes(b"abc")
    assert file_sha256(payload) == hashlib.sha256(b"abc").hexdigest()


def test_existing_model_is_reused(tmp_path: Path) -> None:
    payload = b"weights" * 30
    spec = _Model(
        filename="tiny.onnx",
        urls=("http://127.0.0.1:1/unused",),
        sha256=hashlib.sha256(payload).hexdigest(),
        min_bytes=10,
    )
    dest = tmp_path / spec.filename
    dest.write_bytes(payload)
    assert ensure_model(tmp_path, spec) == dest


def test_bad_checksum_is_not_trusted(tmp_path: Path) -> None:
    spec = _Model(
        filename="tiny.onnx",
        urls=("http://127.0.0.1:1/unused",),
        sha256="0" * 64,
        min_bytes=10,
    )
    (tmp_path / spec.filename).write_bytes(b"this is not the model file")
    with pytest.raises(NofaceError):
        ensure_model(tmp_path, spec)
    assert not (tmp_path / spec.filename).exists()
