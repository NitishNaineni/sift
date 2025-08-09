from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def load_c_extrema(dir_path: str | Path):
    d = Path(dir_path)
    meta = json.loads((d / "extrema_meta.json").read_text())
    ints = np.fromfile(
        d / meta.get("int_file", "extrema_int.i32"), dtype=np.int32
    ).reshape(-1, 4)
    flts = np.fromfile(
        d / meta.get("float_file", "extrema_float.f32"), dtype=np.float32
    ).reshape(-1, 4)
    return ints, flts
