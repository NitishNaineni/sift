from pathlib import Path
import json
import numpy as np
import pytest
import sys


def _load_c_extrema(d: Path):
    meta = json.loads(Path(d / "extrema_meta.json").read_text())
    ints = np.fromfile(
        d / meta.get("int_file", "extrema_int.i32"), dtype=np.int32
    ).reshape(-1, 4)
    flts = np.fromfile(
        d / meta.get("float_file", "extrema_float.f32"), dtype=np.float32
    ).reshape(-1, 4)
    return ints, flts


def test_raw_extrema_match_c_dump(ensure_c_artifacts):
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
            detect_extrema,
            read_gray_bt709,
        )
    except Exception as e:
        pytest.skip(f"proto2 import failed: {e}")

    img_path = Path("data/oxford_affine/graf/img1.png")
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")

    ext_dir = ensure_c_artifacts["ext_dir"]
    if not (ext_dir / "extrema_meta.json").exists():
        pytest.skip(f"C extrema dump not found: {ext_dir}")

    ints_c, flts_c = _load_c_extrema(ext_dir)

    img = read_gray_bt709(str(img_path))
    params = SiftParams(img.shape)
    data = create_sift_data(params)
    data.input_img[...] = cp.asarray(img, dtype=cp.float32)

    set_seed(data, params)
    ints_list = []
    flts_list = []
    for o in range(params.n_oct):
        if o > 0:
            set_first_scale(data, params, o)
        compute_gss(data, params, o)
        compute_dog(data, params, o)
        detect_extrema(data, params, o)
        # read current octave detections
        h_ctr = np.empty(2, dtype=np.int32)
        data.extrema.counter.copy_to_host(h_ctr)
        n = int(h_ctr[0])
        if n > 0:
            ints_list.append(cp.asnumpy(data.extrema.int_buffer[:n]))
            flts_list.append(cp.asnumpy(data.extrema.float_buffer[:n]))

    if len(ints_list) > 0:
        ints_p = np.concatenate(ints_list, axis=0)
        flts_p = np.concatenate(flts_list, axis=0)
    else:
        ints_p = np.empty((0, 4), dtype=np.int32)
        flts_p = np.empty((0, 4), dtype=np.float32)

    # exact set match on (o,s,i,j)
    set_c = set(map(tuple, ints_c.tolist()))
    set_p = set(map(tuple, ints_p.tolist()))
    assert set_c == set_p

    # on matched indices, floats should be equal within tolerance
    idx_map_c = {tuple(ints_c[i]): i for i in range(ints_c.shape[0])}
    idx_map_p = {tuple(ints_p[i]): i for i in range(ints_p.shape[0])}
    max_diffs = np.zeros(4, dtype=np.float64)
    for k in set_c:
        ic = idx_map_c[k]
        ip = idx_map_p[k]
        max_diffs = np.maximum(max_diffs, np.abs(flts_c[ic] - flts_p[ip]))
    assert max_diffs[0] == 0.0  # y
    assert max_diffs[1] == 0.0  # x
    assert max_diffs[2] == 0.0  # sigma
    assert max_diffs[3] <= 1e-6  # val (minor float noise)
