import os
import sys
import json
import unittest
from pathlib import Path
import numpy as np


class SiftComputeMixin:
    TOL_ARRAY = 1e-5
    ORI_TOL = 5e-2
    HAM_FRAC = 0.15
    MAX_SET_DIFF = 50
    BORDER_LAMBDA = 1.0
    REFINED_ATOL = np.array([5e-3, 5e-3, 6e-4, 1e-6], dtype=np.float32)

    IMG_PATH: str | None = None

    @classmethod
    def setUpClass(cls):
        try:
            from numba import cuda  # noqa: F401

            if not cuda.is_available():
                raise unittest.SkipTest("Numba CUDA not available")
        except Exception:
            raise unittest.SkipTest("Numba CUDA not available")

        cls.root = Path(__file__).resolve().parents[1]
        if cls.IMG_PATH:
            p = Path(cls.IMG_PATH)
            cls.img_path = p if p.is_absolute() else (cls.root / cls.IMG_PATH).resolve()
        else:
            cls.img_path = (cls.root / "data/oxford_affine/graf/img6.png").resolve()
        if not cls.img_path.exists():
            raise unittest.SkipTest(f"Test image not found: {cls.img_path}")

        sys.path.append(str(cls.root))
        from proto2 import (
            SiftParams,
            create_sift_data,
            read_gray_bt709,
            compute,
        )

        img = read_gray_bt709(str(cls.img_path))
        cls.params = SiftParams(img.shape)
        cls.data = create_sift_data(cls.params)
        from numba import cuda

        stream = cuda.stream()
        cls.snapshots = compute(cls.data, cls.params, stream, img, record=True)

        # Use the image filename stem to name the artifact directory
        img_stem = cls.img_path.stem
        cls.record_dir = cls.root / f"tests/artifacts/record_c_output_{img_stem}"
        cli_bin = cls.root / "sift_anatomy/bin/sift_cli"
        import subprocess

        try:
            subprocess.run(
                ["make", "-C", str(cls.root / "sift_anatomy"), "clean"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                ["make", "-C", str(cls.root / "sift_anatomy"), "BINFLAGS=-O3"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to build sift_cli: {e}")
        # Always regenerate dumps into a fresh directory
        if cls.record_dir.exists():
            import shutil

            shutil.rmtree(cls.record_dir)
        cls.record_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [str(cli_bin), str(cls.img_path), "--record", str(cls.record_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ,
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to run sift_cli: {e}")

        # Shared resources
        cls.popcnt = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
            axis=1
        )

    @staticmethod
    def _load_json(path: Path):
        with open(path) as f:
            return json.load(f)

    def _concat_pairs(self, key: str):
        pairs = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o].get(key)
            if pair is None:
                continue
            ib, fb = pair
            if ib.size > 0:
                pairs.append((ib, fb))
        if not pairs:
            return np.empty((0, 4), np.int32), np.empty((0, 4), np.float32)
        ints = np.concatenate([p[0] for p in pairs], axis=0)
        flts = np.concatenate([p[1] for p in pairs], axis=0)
        return ints, flts

    def _concat_ints(self, key: str):
        ints, _ = self._concat_pairs(key)
        return ints

    @staticmethod
    def _load_matrix(path: Path, h: int, w: int, dtype=np.float32) -> np.ndarray:
        return np.fromfile(path, dtype=dtype).reshape(h, w)

    def _assert_octave_layer_mats_equal(
        self, meta_path: Path, dump_dir: Path, snapshot_key: str, tol: float
    ) -> None:
        meta = self._load_json(meta_path)
        for o, octave_info in enumerate(meta["octaves"]):
            files = octave_info["files"]
            h, w = octave_info["h"], octave_info["w"]
            snap = self.snapshots[o][snapshot_key]
            if snap is None:
                continue
            for s, fname in enumerate(files):
                c_arr = self._load_matrix(dump_dir / fname, h, w)
                p_arr = snap[s]
                self.assertEqual(c_arr.shape, p_arr.shape)
                diff = np.abs(c_arr - p_arr)
                self.assertLessEqual(diff.max(), tol)

    def _assert_set_parity(
        self,
        set_a,
        set_b,
        *,
        prefix: str = "set mismatch",
        label_a: str = "only_a",
        label_b: str = "only_b",
        max_diff: int | None = None,
    ):
        limit = self.MAX_SET_DIFF if max_diff is None else max_diff
        self.assertTrue(
            len(set_a - set_b) <= limit and len(set_b - set_a) <= limit,
            f"{prefix}: {label_a}={len(set_a - set_b)}, {label_b}={len(set_b - set_a)}",
        )

    def test_gss_dog_internal_consistency(self):
        tol = self.TOL_ARRAY
        for o in range(self.params.n_oct):
            gss = self.snapshots[o]["gss"]
            dog = self.snapshots[o]["dog"]
            if gss is None or dog is None:
                continue
            self.assertEqual(dog.shape[0], gss.shape[0] - 1)
            for s in range(dog.shape[0]):
                diff = np.abs((gss[s + 1] - gss[s]) - dog[s])
                self.assertLessEqual(diff.max(), tol)

    def test_gss_matches_cli_dump(self):
        self._assert_octave_layer_mats_equal(
            self.record_dir / "gss/gss_meta.json",
            self.record_dir / "gss",
            "gss",
            self.TOL_ARRAY,
        )

    def test_dog_matches_cli_dump(self):
        self._assert_octave_layer_mats_equal(
            self.record_dir / "dog/dog_meta.json",
            self.record_dir / "dog",
            "dog",
            self.TOL_ARRAY,
        )

    def test_oriented_keypoints_match_cli_dump(self):
        keys_dir = self.record_dir / "keys"
        meta_path = keys_dir / "keys_meta.json"
        if not meta_path.exists():
            self.skipTest("keys dump not found (CLI didn't dump keys)")
        meta = self._load_json(meta_path)
        ints_c = np.fromfile(
            keys_dir / meta.get("int_file", "keys_int.i32"), dtype=np.int32
        ).reshape(-1, 4)
        flts_c = np.fromfile(
            keys_dir / meta.get("float_file", "keys_float.f32"), dtype=np.float32
        ).reshape(-1, 4)

        ints_list = []
        flts_list = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o].get("keys")
            if pair is None:
                continue
            if len(pair) == 3:
                ib, fb, _ = pair
            else:
                ib, fb = pair
            if ib.size > 0:
                ints_list.append(ib)
                flts_list.append(fb)
        if ints_list:
            ints_p = np.concatenate(ints_list, axis=0)
            flts_p = np.concatenate(flts_list, axis=0)
        else:
            ints_p = np.empty((0, 4), np.int32)
            flts_p = np.empty((0, 4), np.float32)

        oc = ints_c[:, 0].astype(np.int32)
        sc = ints_c[:, 1].astype(np.int32)
        yi_c = ints_c[:, 2].astype(np.int32)
        xi_c = ints_c[:, 3].astype(np.int32)

        keys_c = [
            (int(o), int(s), int(yi), int(xi))
            for o, s, yi, xi in zip(oc, sc, yi_c, xi_c)
        ]
        keys_p = [(int(o), int(s), int(yi), int(xi)) for (o, s, yi, xi) in ints_p]

        set_c = set(keys_c)
        set_p = set(keys_p)
        self._assert_set_parity(
            set_c,
            set_p,
            prefix="pre-orientation identity mismatch",
            label_a="only_in_c",
            label_b="only_in_p",
        )
        common_ids = set_c & set_p
        from collections import defaultdict

        grp_c: dict[tuple, list[float]] = defaultdict(list)
        grp_p: dict[tuple, list[float]] = defaultdict(list)
        for k, row in zip(keys_c, flts_c):
            grp_c[k].append(float(row[3]))
        for k, row in zip(keys_p, flts_p):
            grp_p[k].append(float(row[3]))

        def wrap_2pi(a: np.ndarray) -> np.ndarray:
            t = np.float32(2.0 * np.pi)
            return (a % t + t) % t

        def circ_diff(a: float, b: float) -> float:
            twopi = 2.0 * np.pi
            da = abs(((a - b) + np.pi) % twopi - np.pi)
            return float(da)

        th_tol = self.ORI_TOL
        worst = 0.0
        bad = 0
        count_mismatch = 0
        for k in common_ids:
            arr_c = np.array(grp_c.get(k, []), dtype=np.float32)
            arr_p = np.array(grp_p.get(k, []), dtype=np.float32)
            if arr_c.size != arr_p.size:
                count_mismatch += 1
                continue
            if arr_c.size == 0:
                continue
            arr_c = wrap_2pi(arr_c)
            arr_p = wrap_2pi(arr_p)
            used = np.zeros(arr_p.size, dtype=bool)
            for ang_c in arr_c:
                diffs = np.array(
                    [circ_diff(float(ang_c), float(ang_p)) for ang_p in arr_p]
                )
                diffs[used] = 1e9
                j = int(np.argmin(diffs))
                used[j] = True
                d = float(diffs[j])
                worst = max(worst, d)
                if d > th_tol:
                    bad += 1
        self.assertEqual(
            count_mismatch,
            0,
            msg=f"orientation count mismatch groups: {count_mismatch}",
        )
        self.assertEqual(
            bad, 0, msg=f"orientation mismatches > tol: {bad}, worst={worst}"
        )

    def test_descriptors_match_cli_dump(self):
        keys_dir = self.record_dir / "keys"
        meta_path = keys_dir / "keys_meta.json"
        if not meta_path.exists():
            self.skipTest("keys dump not found (CLI didn't dump keys)")
        meta = self._load_json(meta_path)
        ints_c = np.fromfile(
            keys_dir / meta.get("int_file", "keys_int.i32"), dtype=np.int32
        ).reshape(-1, 4)
        desc_len = int(meta.get("desc_len", 128))
        desc_c = np.fromfile(
            keys_dir / meta.get("desc_file", "keys_desc.u8"), dtype=np.uint8
        ).reshape(-1, desc_len)

        ints_list = []
        desc_list = []
        for o in range(self.params.n_oct):
            triple = self.snapshots[o].get("keys")
            if triple is None:
                continue
            if len(triple) == 3:
                ib, _, db = triple
            else:
                continue
            if ib.size > 0:
                ints_list.append(ib)
                desc_list.append(db)
        if ints_list:
            ints_p = np.concatenate(ints_list, axis=0)
            desc_p = np.concatenate(desc_list, axis=0)
        else:
            ints_p = np.empty((0, 4), np.int32)
            desc_p = np.empty((0, desc_len), np.uint8)

        from collections import defaultdict

        grp_c = defaultdict(list)
        grp_p = defaultdict(list)
        for row, d in zip(ints_c, desc_c):
            grp_c[tuple(row.astype(np.int32))].append(d)
        for row, d in zip(ints_p, desc_p):
            grp_p[tuple(row.astype(np.int32))].append(d)

        ham_thresh = int(self.HAM_FRAC * desc_len * 8)
        mismatches = 0
        worst = 0
        for key in set(grp_c.keys()) & set(grp_p.keys()):
            arr_c = np.stack(grp_c[key], axis=0).astype(np.uint8)
            arr_p = np.stack(grp_p[key], axis=0).astype(np.uint8)
            used = np.zeros(arr_p.shape[0], dtype=bool)
            for dc in arr_c:
                x = np.bitwise_xor(arr_p, dc[None, :])
                ham = self.popcnt[x].sum(axis=1)
                ham[used] = 1e9
                j = int(np.argmin(ham))
                used[j] = True
                h = int(ham[j])
                worst = max(worst, h)
                if h > ham_thresh:
                    mismatches += 1
        self.assertTrue(
            mismatches <= 10 and worst <= ham_thresh,
            f"descriptor mismatches: {mismatches} (worst Hamming={worst}, thresh={ham_thresh})",
        )

    def test_extrema_matches_cli_dump(self):
        meta = self._load_json(self.record_dir / "extrema/extrema_meta.json")
        ints_c = np.fromfile(
            self.record_dir / "extrema" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)
        ints_p = self._concat_ints("extrema")

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        self._assert_set_parity(
            set_c,
            set_p,
            prefix="c_post set mismatch",
            label_a="only_in_c",
            label_b="only_in_py",
        )

    def test_c_pre_matches_cli_dump(self):
        meta = self._load_json(self.record_dir / "c_pre/extrema_meta.json")
        ints_c = np.fromfile(
            self.record_dir / "c_pre" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)
        ints_p = self._concat_ints("c_pre")

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        self._assert_set_parity(
            set_c,
            set_p,
            prefix="c_pre set mismatch",
            label_a="only_in_c",
            label_b="only_in_py",
        )

    def test_c_post_matches_cli_dump(self):
        meta = self._load_json(self.record_dir / "c_post/extrema_meta.json")
        ints_c = np.fromfile(
            self.record_dir / "c_post" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)
        ints_p = self._concat_ints("c_post")

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        self._assert_set_parity(
            set_c,
            set_p,
            prefix="c_post set mismatch",
            label_a="only_in_c",
            label_b="only_in_py",
        )

    def test_edge_matches_cli_dump(self):
        meta = self._load_json(self.record_dir / "edge/extrema_meta.json")
        ints_c = np.fromfile(
            self.record_dir / "edge" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)
        ints_p = self._concat_ints("edge")

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        self._assert_set_parity(
            set_c,
            set_p,
            prefix="edge set mismatch",
            label_a="only_in_c",
            label_b="only_in_py",
        )

    def test_border_matches_cli_dump(self):
        meta = self._load_json(self.record_dir / "border/extrema_meta.json")
        ints_c = np.fromfile(
            self.record_dir / "border" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)
        ints_p = self._concat_ints("border")

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        self._assert_set_parity(
            set_c,
            set_p,
            prefix="border set mismatch",
            label_a="only_in_c",
            label_b="only_in_py",
        )

    def test_border_world_mask_consistency(self):
        H, W = self.params.img_dims
        lam = self.BORDER_LAMBDA
        keep_sets = []
        got_sets = []
        for o in range(self.params.n_oct):
            edge = self.snapshots[o]["edge"]
            border = self.snapshots[o]["border"]
            if edge is None:
                continue
            ints_e, flts_e = edge
            y, x, sigma = flts_e[:, 0], flts_e[:, 1], flts_e[:, 2]
            cond = (
                (y - lam * sigma > 0.0)
                & (y + lam * sigma < float(H))
                & (x - lam * sigma > 0.0)
                & (x + lam * sigma < float(W))
            )
            keep = set(map(tuple, ints_e[cond].tolist()))
            keep_sets.append(keep)
            got = set() if border is None else set(map(tuple, border[0].tolist()))
            got_sets.append(got)
        keep_all = set().union(*keep_sets) if keep_sets else set()
        got_all = set().union(*got_sets) if got_sets else set()

        self._assert_set_parity(
            keep_all,
            got_all,
            prefix="border (world-mask) mismatch",
            label_a="only_keep",
            label_b="only_got",
        )

    def test_refined_matches_cli_dump(self):
        meta = self._load_json(self.record_dir / "refined/extrema_refined_meta.json")
        ints_c = np.fromfile(
            self.record_dir
            / "refined"
            / meta.get("int_file", "extrema_refined_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)
        flts_c = np.fromfile(
            self.record_dir
            / "refined"
            / meta.get("float_file", "extrema_refined_float.f32"),
            dtype=np.float32,
        ).reshape(-1, 4)
        ints_p, flts_p = self._concat_pairs("refined")

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        self._assert_set_parity(
            set_p,
            set_c,
            prefix="refined set mismatch",
            label_a="only_in_py",
            label_b="only_in_c",
        )

        common = set_c & set_p
        idx_c = {tuple(ints_c[i]): i for i in range(ints_c.shape[0])}
        idx_p = {tuple(ints_p[i]): i for i in range(ints_p.shape[0])}
        if common:
            keys = list(common)
            idxs_c = np.array([idx_c[k] for k in keys], dtype=np.int64)
            idxs_p = np.array([idx_p[k] for k in keys], dtype=np.int64)
            diffs = np.abs(flts_c[idxs_c] - flts_p[idxs_p])
            atol = self.REFINED_ATOL
            matches = diffs <= atol
            col_counts = matches.sum(axis=0)
            total = matches.shape[0]
            overall_count = int(matches.all(axis=1).sum())
            min_ok = max(total - 10, 0)
            if not (overall_count >= min_ok and np.all(col_counts >= min_ok)):
                max_diffs = diffs.max(axis=0)
                bad_counts = (~(diffs <= atol)).sum(axis=0)
                self.fail(
                    f"refined float matches too low: overall {overall_count}/{total}, "
                    f"cols {col_counts.tolist()}/{total}, "
                    f"bad_counts {bad_counts.tolist()}, max_diffs {max_diffs.tolist()}"
                )

    def test_keys_present_per_octave(self):
        for o in range(self.params.n_oct):
            snap = self.snapshots[o]
            for key in (
                "gss",
                "dog",
                "extrema",
                "c_pre",
                "refined",
                "c_post",
                "edge",
                "border",
            ):
                with self.subTest(octave=o, key=key):
                    self.assertIn(key, snap)

    def test_monotonic_counts(self):
        for o in range(self.params.n_oct):
            snap = self.snapshots[o]

            def count(pair):
                return 0 if pair is None else int(pair[0].shape[0])

            counts = {
                "extrema": count(snap["extrema"]),
                "c_pre": count(snap["c_pre"]),
                "refined": count(snap["refined"]),
                "c_post": count(snap["c_post"]),
                "edge": count(snap["edge"]),
                "border": count(snap["border"]),
            }
            with self.subTest(octave=o, counts=counts):
                self.assertGreaterEqual(counts["extrema"], counts["c_pre"])
                self.assertGreaterEqual(counts["c_pre"], counts["refined"])
                self.assertGreaterEqual(counts["refined"], counts["c_post"])
                self.assertGreaterEqual(counts["c_post"], counts["edge"])
                self.assertGreaterEqual(counts["edge"], counts["border"])
                self.assertGreaterEqual(counts["border"], 0)

    def _assert_shapes_dtypes(self, pair, *, floats_cols):
        if pair is None:
            return
        ints, flts = pair
        self.assertEqual(ints.dtype, np.int32)
        self.assertEqual(flts.dtype, np.float32)
        self.assertEqual(ints.shape[1], 4)
        self.assertEqual(flts.shape[1], floats_cols)

    def test_shapes_dtypes_extrema(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["extrema"], floats_cols=4)

    def test_shapes_dtypes_c_pre(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["c_pre"], floats_cols=4)

    def test_shapes_dtypes_refined(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["refined"], floats_cols=4)

    def test_shapes_dtypes_c_post(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["c_post"], floats_cols=4)

    def test_shapes_dtypes_edge(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["edge"], floats_cols=4)

    def test_shapes_dtypes_border(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["border"], floats_cols=4)


class TestSiftImg1(SiftComputeMixin, unittest.TestCase):
    IMG_PATH = "data/oxford_affine/graf/img1.png"


# class TestSiftImg2(SiftComputeMixin, unittest.TestCase):
#     IMG_PATH = "data/oxford_affine/graf/img2.png"
