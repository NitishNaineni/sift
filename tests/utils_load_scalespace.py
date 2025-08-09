from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def load_gss_dump(dir_path: str | Path):
    d = Path(dir_path)
    meta = json.loads((d / "gss_meta.json").read_text())
    out = {}
    for octave in meta["octaves"]:
        o = octave["o"]
        w, h = octave["w"], octave["h"]
        out[o] = {}
        for s, fname in enumerate(octave["files"]):
            arr = np.fromfile(d / fname, dtype=np.float32).reshape(h, w)
            out[o][s] = arr
    return out, meta
