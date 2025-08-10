import os
import sys
import json
import unittest
from pathlib import Path
import numpy as np


class TestSiftCompute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp  # noqa: F401
        except Exception:
            raise unittest.SkipTest("CuPy not available")

        cls.root = Path(__file__).resolve().parents[1]
        cls.img_path = cls.root / "data/oxford_affine/graf/img1.png"
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
        import cupy as cp

        cls.data.input_img[...] = cp.asarray(img, dtype=cp.float32)
        cls.snapshots = compute(cls.data, cls.params, record=True)

        # Always build and run C CLI to record fresh reference dumps
        cls.record_dir = cls.root / "tests/artifacts/record_c_output"
        cli_bin = cls.root / "sift_anatomy/bin/sift_cli"
        import subprocess

        try:
            subprocess.run(
                ["make", "-C", str(cls.root / "sift_anatomy")],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to build sift_cli: {e}")
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

    def test_gss_dog_internal_consistency(self):
        tol = 1e-6
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
        tol = 1e-6
        with open(self.record_dir / "gss/gss_meta.json") as f:
            meta = json.load(f)
        gss_dir = self.record_dir / "gss"
        for o, octave_info in enumerate(meta["octaves"]):
            files = octave_info["files"]
            h, w = octave_info["h"], octave_info["w"]
            gss = self.snapshots[o]["gss"]
            if gss is None:
                continue
            for s, fname in enumerate(files):
                c_arr = np.fromfile(gss_dir / fname, dtype=np.float32).reshape(h, w)
                p_arr = gss[s]
                self.assertEqual(c_arr.shape, p_arr.shape)
                diff = np.abs(c_arr - p_arr)
                self.assertLessEqual(diff.max(), tol)

    def test_dog_matches_cli_dump(self):
        tol = 1e-6
        with open(self.record_dir / "dog/dog_meta.json") as f:
            meta = json.load(f)
        dog_dir = self.record_dir / "dog"
        for o, octave_info in enumerate(meta["octaves"]):
            files = octave_info["files"]
            h, w = octave_info["h"], octave_info["w"]
            dog = self.snapshots[o]["dog"]
            if dog is None:
                continue
            for s, fname in enumerate(files):
                c_arr = np.fromfile(dog_dir / fname, dtype=np.float32).reshape(h, w)
                p_arr = dog[s]
                self.assertEqual(c_arr.shape, p_arr.shape)
                diff = np.abs(c_arr - p_arr)
                self.assertLessEqual(diff.max(), tol)

    def test_gradients_match_cli_dump(self):
        tol = 1e-6
        # Load grad_x and grad_y from CLI and compare directly with Python grad_x/grad_y
        with open(self.record_dir / "grad_x/grad_x_meta.json") as fx:
            meta_x = json.load(fx)
        with open(self.record_dir / "grad_y/grad_y_meta.json") as fy:
            meta_y = json.load(fy)
        dir_x = self.record_dir / "grad_x"
        dir_y = self.record_dir / "grad_y"
        for o, (ox, oy) in enumerate(zip(meta_x["octaves"], meta_y["octaves"])):
            files_x = ox["files"]
            files_y = oy["files"]
            h, w = ox["h"], ox["w"]
            # Our Python snapshot now includes grad_x/grad_y
            grad_x = self.snapshots[o].get("grad_x")
            grad_y = self.snapshots[o].get("grad_y")
            if grad_x is None or grad_y is None:
                continue
            for s, (fxname, fyname) in enumerate(zip(files_x, files_y)):
                gx_cli = np.fromfile(dir_x / fxname, dtype=np.float32).reshape(h, w)
                gy_cli = np.fromfile(dir_y / fyname, dtype=np.float32).reshape(h, w)
                # Our convention: grad_x = dI/dx (columns), grad_y = dI/dy (rows)
                # CLI convention: grad_x = dI/dy, grad_y = dI/dx
                # Therefore compare gx_cli to our grad_y, and gy_cli to our grad_x
                pgx = grad_x[s]
                pgy = grad_y[s]
                self.assertEqual(gx_cli.shape, pgy.shape)
                self.assertEqual(gy_cli.shape, pgx.shape)
                self.assertLessEqual(np.max(np.abs(gx_cli - pgy)), tol)
                self.assertLessEqual(np.max(np.abs(gy_cli - pgx)), tol)

    def test_oriented_keypoints_match_cli_dump(self):
        # Load C dump (oriented keypoints)
        keys_dir = self.record_dir / "keys"
        meta_path = keys_dir / "keys_meta.json"
        if not meta_path.exists():
            self.skipTest("keys dump not found (CLI didn't dump keys)")
        with open(meta_path) as f:
            meta = json.load(f)
        ints_c = np.fromfile(
            keys_dir / meta.get("int_file", "keys_int.i32"), dtype=np.int32
        ).reshape(-1, 4)
        flts_c = np.fromfile(
            keys_dir / meta.get("float_file", "keys_float.f32"), dtype=np.float32
        ).reshape(-1, 4)

        # Collect python keys from snapshots (assign_orientations returns per-octave keys when record=True)
        ints_list = []
        flts_list = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o].get("keys")
            if pair is None:
                continue
            ib, fb = pair
            if ib.size > 0:
                ints_list.append(ib)
                flts_list.append(fb)
        if ints_list:
            import numpy as _np

            ints_p = _np.concatenate(ints_list, axis=0)
            flts_p = _np.concatenate(flts_list, axis=0)
        else:
            import numpy as _np

            ints_p = _np.empty((0, 4), _np.int32)
            flts_p = _np.empty((0, 4), _np.float32)

        # Identity parity using (o, s, i, j) only
        oc = ints_c[:, 0].astype(np.int32)
        sc = ints_c[:, 1].astype(np.int32)
        yi_c = ints_c[:, 2].astype(np.int32)
        xi_c = ints_c[:, 3].astype(np.int32)

        keys_c = [
            (int(o), int(s), int(yi), int(xi))
            for o, s, yi, xi in zip(oc, sc, yi_c, xi_c)
        ]
        keys_p = [(int(o), int(s), int(yi), int(xi)) for (o, s, yi, xi) in ints_p]

        # Assert identity parity only (o, s, i, j)
        set_c = set(keys_c)
        set_p = set(keys_p)
        only_c = set_c - set_p
        only_p = set_p - set_c
        if len(only_c) > 10 or len(only_p) > 10:
            self.fail(
                "pre-orientation identity mismatch: "
                f"only_in_c={len(only_c)}, only_in_p={len(only_p)}"
            )
        # Orientation comparison for common identities
        common_ids = set_c & set_p
        from collections import defaultdict

        grp_c: dict[tuple, list[float]] = defaultdict(list)
        grp_p: dict[tuple, list[float]] = defaultdict(list)
        for k, row in zip(keys_c, flts_c):
            grp_c[k].append(float(row[3]))
        for k, row in zip(keys_p, flts_p):
            grp_p[k].append(float(row[3]))

        def wrap_2pi(a: np.ndarray) -> np.ndarray:
            twopi = np.float32(2.0 * np.pi)
            return (a % twopi + twopi) % twopi

        def circ_diff(a: float, b: float) -> float:
            twopi = 2.0 * np.pi
            da = abs(((a - b) + np.pi) % twopi - np.pi)
            return float(da)

        th_tol = 5e-2
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
        return

    def test_extrema_matches_cli_dump(self):
        # Load C dump (raw extrema kA)
        with open(self.record_dir / "extrema/extrema_meta.json") as f:
            meta = json.load(f)
        ints_c = np.fromfile(
            self.record_dir / "extrema" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)

        # Concatenate python extrema across octaves
        ints_list = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o]["extrema"]
            if pair is None:
                continue
            ib, _ = pair
            if ib.size > 0:
                ints_list.append(ib)
        ints_p = (
            np.concatenate(ints_list, axis=0)
            if ints_list
            else np.empty((0, 4), np.int32)
        )

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        only_c = set_c - set_p
        only_p = set_p - set_c
        self.assertTrue(
            len(only_c) <= 10 and len(only_p) <= 10,
            (
                f"contrast_post set mismatch: only_in_c={len(only_c)}, only_in_py={len(only_p)}"
            ),
        )

    def test_contrast_pre_matches_cli_dump(self):
        # Load C dump (contrast_pre kB)
        with open(self.record_dir / "contrast_pre/extrema_meta.json") as f:
            meta = json.load(f)
        ints_c = np.fromfile(
            self.record_dir / "contrast_pre" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)

        # Concatenate python contrast_pre across octaves
        ints_list = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o]["contrast_pre"]
            if pair is None:
                continue
            ib, _ = pair
            if ib.size > 0:
                ints_list.append(ib)
        ints_p = (
            np.concatenate(ints_list, axis=0)
            if ints_list
            else np.empty((0, 4), np.int32)
        )

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        only_c = set_c - set_p
        only_p = set_p - set_c
        self.assertTrue(
            len(only_c) <= 10 and len(only_p) <= 10,
            (
                f"contrast_pre set mismatch: only_in_c={len(only_c)}, only_in_py={len(only_p)}"
            ),
        )

    def test_contrast_post_matches_cli_dump(self):
        # Load C dump (contrast_post kD)
        with open(self.record_dir / "contrast_post/extrema_meta.json") as f:
            meta = json.load(f)
        ints_c = np.fromfile(
            self.record_dir / "contrast_post" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)

        # Concatenate python contrast_post across octaves
        ints_list = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o]["contrast_post"]
            if pair is None:
                continue
            ib, _ = pair
            if ib.size > 0:
                ints_list.append(ib)
        ints_p = (
            np.concatenate(ints_list, axis=0)
            if ints_list
            else np.empty((0, 4), np.int32)
        )

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        only_c = set_c - set_p
        only_p = set_p - set_c
        self.assertTrue(
            len(only_c) <= 10 and len(only_p) <= 10,
            (
                f"contrast_post set mismatch: only_in_c={len(only_c)}, only_in_py={len(only_p)}"
            ),
        )

    def test_edge_matches_cli_dump(self):
        # Load C dump (edge kE)
        with open(self.record_dir / "edge/extrema_meta.json") as f:
            meta = json.load(f)
        ints_c = np.fromfile(
            self.record_dir / "edge" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)

        # Concatenate python edge across octaves
        ints_list = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o]["edge"]
            if pair is None:
                continue
            ib, _ = pair
            if ib.size > 0:
                ints_list.append(ib)
        ints_p = (
            np.concatenate(ints_list, axis=0)
            if ints_list
            else np.empty((0, 4), np.int32)
        )

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        only_c = set_c - set_p
        only_p = set_p - set_c
        self.assertTrue(
            len(only_c) <= 10 and len(only_p) <= 10,
            (f"edge set mismatch: only_in_c={len(only_c)}, only_in_py={len(only_p)}"),
        )

    def test_border_matches_cli_dump(self):
        # Load C dump (border kF)
        with open(self.record_dir / "border/extrema_meta.json") as f:
            meta = json.load(f)
        ints_c = np.fromfile(
            self.record_dir / "border" / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)

        # Concatenate python border across octaves
        ints_list = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o]["border"]
            if pair is None:
                continue
            ib, _ = pair
            if ib.size > 0:
                ints_list.append(ib)
        ints_p = (
            np.concatenate(ints_list, axis=0)
            if ints_list
            else np.empty((0, 4), np.int32)
        )

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        only_c = set_c - set_p
        only_p = set_p - set_c
        self.assertTrue(
            len(only_c) <= 10 and len(only_p) <= 10,
            (f"border set mismatch: only_in_c={len(only_c)}, only_in_py={len(only_p)}"),
        )

    def test_border_world_mask_consistency(self):
        # Recompute border mask from world coordinates and compare to our border set
        H, W = self.params.img_dims
        lam = 1.0
        # Build python sets from edge -> mask
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
        only_c = keep_all - got_all
        only_p = got_all - keep_all
        self.assertTrue(
            len(only_c) <= 10 and len(only_p) <= 10,
            (
                f"border (world-mask) mismatch: only_keep={len(only_c)}, only_got={len(only_p)}"
            ),
        )

    def test_refined_matches_cli_dump(self):
        # Load C dump (refined kC)
        with open(self.record_dir / "refined/extrema_refined_meta.json") as f:
            meta = json.load(f)
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

        # Concatenate python refined across octaves
        ints_list = []
        flts_list = []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o]["refined"]
            if pair is None:
                continue
            ib, fb = pair
            if ib.size > 0:
                ints_list.append(ib)
                flts_list.append(fb)
        ints_p = (
            np.concatenate(ints_list, axis=0)
            if ints_list
            else np.empty((0, 4), np.int32)
        )
        flts_p = (
            np.concatenate(flts_list, axis=0)
            if flts_list
            else np.empty((0, 4), np.float32)
        )

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        only_c = set_c - set_p
        only_p = set_p - set_c
        self.assertTrue(
            len(only_p) <= 10 and len(only_c) <= 10,
            (
                f"refined set mismatch: only_in_c={len(only_c)}, only_in_py={len(only_p)}"
            ),
        )

        common = set_c & set_p
        idx_c = {tuple(ints_c[i]): i for i in range(ints_c.shape[0])}
        idx_p = {tuple(ints_p[i]): i for i in range(ints_p.shape[0])}
        if common:
            keys = list(common)
            idxs_c = np.array([idx_c[k] for k in keys], dtype=np.int64)
            idxs_p = np.array([idx_p[k] for k in keys], dtype=np.int64)
            diffs = np.abs(flts_c[idxs_c] - flts_p[idxs_p])
            # Tolerances per column: [y, x, sigma, val]
            # Slightly relaxed to reflect numeric differences while staying strict
            atol = np.array([5e-3, 5e-3, 6e-4, 1e-6], dtype=np.float32)
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

        # Floats exist too, but we only need integer index parity here

    def test_keys_present_per_octave(self):
        for o in range(self.params.n_oct):
            snap = self.snapshots[o]
            for key in (
                "gss",
                "grad_x",
                "grad_y",
                "dog",
                "extrema",
                "contrast_pre",
                "refined",
                "contrast_post",
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
                "contrast_pre": count(snap["contrast_pre"]),
                "refined": count(snap["refined"]),
                "contrast_post": count(snap["contrast_post"]),
                "edge": count(snap["edge"]),
                "border": count(snap["border"]),
            }
            with self.subTest(octave=o, counts=counts):
                self.assertGreaterEqual(counts["extrema"], counts["contrast_pre"])
                self.assertGreaterEqual(counts["contrast_pre"], counts["refined"])
                self.assertGreaterEqual(counts["refined"], counts["contrast_post"])
                self.assertGreaterEqual(counts["contrast_post"], counts["edge"])
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

    def test_shapes_dtypes_contrast_pre(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(
                    self.snapshots[o]["contrast_pre"], floats_cols=4
                )

    def test_shapes_dtypes_refined(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["refined"], floats_cols=4)

    def test_shapes_dtypes_contrast_post(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(
                    self.snapshots[o]["contrast_post"], floats_cols=4
                )

    def test_shapes_dtypes_edge(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["edge"], floats_cols=4)

    def test_shapes_dtypes_border(self):
        for o in range(self.params.n_oct):
            with self.subTest(octave=o):
                self._assert_shapes_dtypes(self.snapshots[o]["border"], floats_cols=4)


if __name__ == "__main__":
    unittest.main()
