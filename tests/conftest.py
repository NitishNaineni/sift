from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ensure_c_artifacts():
    """Ensure C-side artifacts exist; build and generate if missing.

    - Builds sift_cli if bin is missing
    - Generates GSS and raw extrema dumps into tests/artifacts/
    - Skips tests gracefully if prerequisites (image or toolchain) are missing
    """
    root = Path(__file__).resolve().parents[1]
    img_path = root / "data/oxford_affine/graf/img1.png"
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")

    gss_dir = root / "tests/artifacts/gss_c_output"
    ext_dir = root / "tests/artifacts/ext_c_output"
    gss_meta = gss_dir / "gss_meta.json"
    ext_meta = ext_dir / "extrema_meta.json"

    need_gss = not gss_meta.exists()
    need_ext = not ext_meta.exists()

    if need_gss or need_ext:
        bin_path = root / "sift_anatomy/bin/sift_cli"
        if not bin_path.exists():
            # try to build the CLI
            try:
                subprocess.run(
                    ["make", "-C", str(root / "sift_anatomy")],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception as e:
                pytest.skip(f"Failed to build sift_cli: {e}")

        gss_dir.mkdir(parents=True, exist_ok=True)
        ext_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(bin_path),
            str(img_path),
            "-dump_ss_dir",
            str(gss_dir),
            "-dump_extrema_dir",
            str(ext_dir),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ,
            )
        except Exception as e:
            pytest.skip(f"Failed to run sift_cli: {e}")

    return {"gss_dir": gss_dir, "ext_dir": ext_dir}
