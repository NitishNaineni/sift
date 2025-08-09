import json
from pathlib import Path
import sys

import numpy as np
import pytest


def test_gss_and_dog_match_c_dump(ensure_c_artifacts):
    try:
        import cupy as cp
    except Exception:
        pytest.skip("CuPy not available")

    try:
        # Ensure project root is on sys.path
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from proto2 import (
            SiftParams,
            create_sift_data,
            set_seed,
            set_first_scale,
            compute_gss,
            compute_dog,
            read_gray_bt709,
        )
    except Exception as e:
        pytest.skip(f"proto2 import failed: {e}")

    img_path = Path("data/oxford_affine/graf/img1.png")
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")

    gss_dir = ensure_c_artifacts["gss_dir"]
    meta_path = gss_dir / "gss_meta.json"
    if not meta_path.exists():
        pytest.skip(f"C GSS dump not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load image and run proto2 pipeline to compute GSS/DoG
    img = read_gray_bt709(str(img_path))
    params = SiftParams(img.shape)
    data = create_sift_data(params)
    data.input_img[...] = cp.asarray(img, dtype=cp.float32)

    set_seed(data, params)
    for o in range(params.n_oct):
        if o > 0:
            set_first_scale(data, params, o)
        compute_gss(data, params, o)
        compute_dog(data, params, o)

    tol = 1e-6

    # Compare GSS
    for o, octave_info in enumerate(meta["octaves"]):
        files = octave_info["files"]
        for s, fname in enumerate(files):
            c_arr = np.fromfile(gss_dir / fname, dtype=np.float32)
            h, w = octave_info["h"], octave_info["w"]
            c_arr = c_arr.reshape(h, w)
            p_arr = cp.asnumpy(data.gss[o][s])
            assert c_arr.shape == p_arr.shape
            diff = np.abs(c_arr - p_arr)
            assert diff.max() <= tol

    # Compare DoG derived from C GSS versus proto2 DoG
    for o, octave_info in enumerate(meta["octaves"]):
        files = octave_info["files"]
        for s in range(len(files) - 1):
            c0 = np.fromfile(gss_dir / files[s], dtype=np.float32).reshape(
                octave_info["h"], octave_info["w"]
            )
            c1 = np.fromfile(gss_dir / files[s + 1], dtype=np.float32).reshape(
                octave_info["h"], octave_info["w"]
            )
            dog_c = c1 - c0
            dog_p = cp.asnumpy(data.dog[o][s])
            assert dog_c.shape == dog_p.shape
            diff = np.abs(dog_c - dog_p)
            assert diff.max() <= tol
