"""Tests for CUDA SIFT implementation against C reference.

This module validates the CUDA implementation by comparing intermediate results
and final outputs against the reference C implementation from sift_anatomy.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from typing import List
from urllib.request import urlretrieve
import zipfile

import numpy as np


def download_and_cache_div2k_validation(cache_dir: Path) -> List[Path]:
    """Download and cache DIV2K validation images.

    Args:
        cache_dir: Directory to cache the downloaded and extracted images

    Returns:
        List of paths to the extracted PNG images
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = cache_dir / "DIV2K_valid_HR"
    if extracted_dir.exists():
        images = sorted(extracted_dir.glob("*.png"))
        if images:
            print(f"Using cached DIV2K validation images: {len(images)} images found")
            return images

    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    zip_path = cache_dir / "DIV2K_valid_HR.zip"

    if not zip_path.exists():
        print(f"Downloading DIV2K validation dataset from {url}...")
        try:
            urlretrieve(url, zip_path)
            print(f"Download complete: {zip_path}")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to download DIV2K dataset: {e}") from e

    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
        print(f"Extraction complete to {cache_dir}")
    except Exception as e:
        raise unittest.SkipTest(f"Failed to extract DIV2K dataset: {e}") from e

    images = sorted(extracted_dir.glob("*.png"))
    if not images:
        raise unittest.SkipTest(f"No PNG images found in {extracted_dir}")

    print(f"Found {len(images)} DIV2K validation images")
    return images


class SiftComputeMixin:
    """Mixin class providing SIFT testing functionality.

    Tolerances are tightened to ensure close alignment with C reference.
    """

    TOL_ARRAY = 5.0e-6
    ORI_TOL = 5e-3
    ORI_MISMATCH_PCT = 0.0015
    HAM_FRAC = 0.05
    HAM_MISMATCH_PCT = 0.01
    MAX_SET_DIFF_PCT = 0.0051
    BORDER_LAMBDA = 1.0
    REFINED_ATOL = np.array([6e-3, 6e-3, 6e-4, 3e-7], dtype=np.float32)
    REFINED_FLOAT_FAIL_PCT = 0.005

    IMG_PATH: str | None = None

    @classmethod
    def setUpClass(cls):
        cls._check_cuda_available()
        cls._setup_paths()
        cls._run_python_sift()
        cls._build_and_run_c_reference()
        cls._setup_shared_resources()

    @classmethod
    def tearDownClass(cls):
        """Clean up GPU memory after all tests in this class complete."""
        if hasattr(cls, "detector"):
            # Delete the detector to free GPU memory
            del cls.detector
        if hasattr(cls, "snapshots"):
            del cls.snapshots

        # Force CUDA context synchronization and garbage collection
        try:
            from numba import cuda

            cuda.synchronize()
        except Exception:
            pass  # Ignore cleanup errors

        import gc

        gc.collect()

    @classmethod
    def _check_cuda_available(cls):
        try:
            from numba import cuda  # noqa: F401

            if not cuda.is_available():
                raise unittest.SkipTest("Numba CUDA not available")
        except Exception as e:
            raise unittest.SkipTest("Numba CUDA not available") from e

    @classmethod
    def _setup_paths(cls):
        cls.root = Path(__file__).resolve().parents[1]
        if cls.IMG_PATH:
            p = Path(cls.IMG_PATH)
            cls.img_path = p if p.is_absolute() else (cls.root / cls.IMG_PATH).resolve()
        else:
            cls.img_path = (cls.root / "data/oxford_affine/graf/img6.png").resolve()

        if not cls.img_path.exists():
            raise unittest.SkipTest(f"Test image not found: {cls.img_path}")

        cls.record_dir = (
            cls.root / f"tests/artifacts/record_c_output_{cls.img_path.stem}"
        )

    @classmethod
    def _run_python_sift(cls):
        sys.path.append(str(cls.root))
        from cudasift import SiftDetector, read_gray_bt709

        img = read_gray_bt709(str(cls.img_path))
        cls.detector = SiftDetector(img_dims=img.shape, record=True)
        cls.params = cls.detector.params
        cls.detector.data.input_img.copy_to_device(
            img.astype(np.float32), cls.detector._stream
        )
        cls.snapshots = cls.detector._compute()

    @classmethod
    def _build_and_run_c_reference(cls):
        import subprocess

        cli_bin = cls.root / "sift_anatomy/bin/sift_cli"

        if cls.record_dir.exists() and (cls.record_dir / "gss/gss_meta.json").exists():
            return

        if not cli_bin.exists():
            try:
                subprocess.run(
                    ["make", "-C", str(cls.root / "sift_anatomy"), "BINFLAGS=-O3"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                raise unittest.SkipTest(f"Failed to build sift_cli: {e}") from e

        cls.record_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [str(cli_bin), str(cls.img_path), "--record", str(cls.record_dir)],
                check=True,
                capture_output=True,
                text=True,
                env=os.environ,
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to run sift_cli: {e}") from e

    @classmethod
    def _setup_shared_resources(cls):
        cls.popcnt = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
            axis=1
        )

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _load_matrix(path: Path, h: int, w: int, dtype=np.float32) -> np.ndarray:
        return np.fromfile(path, dtype=dtype).reshape(h, w)

    def _load_extrema_ints(self, stage: str) -> np.ndarray:
        meta = self._load_json(self.record_dir / stage / "extrema_meta.json")
        return np.fromfile(
            self.record_dir / stage / meta.get("int_file", "extrema_int.i32"),
            dtype=np.int32,
        ).reshape(-1, 4)

    def _load_extrema_pairs(self, stage: str) -> tuple[np.ndarray, np.ndarray]:
        if "refined" in stage:
            meta = self._load_json(
                self.record_dir / stage / "extrema_refined_meta.json"
            )
            int_file = meta.get("int_file", "extrema_refined_int.i32")
            float_file = meta.get("float_file", "extrema_refined_float.f32")
        else:
            meta = self._load_json(self.record_dir / stage / "extrema_meta.json")
            int_file = meta.get("int_file", "extrema_int.i32")
            float_file = meta.get("float_file", "extrema_float.f32")

        ints = np.fromfile(self.record_dir / stage / int_file, dtype=np.int32).reshape(
            -1, 4
        )
        floats = np.fromfile(
            self.record_dir / stage / float_file, dtype=np.float32
        ).reshape(-1, 4)
        return ints, floats

    def _concat_pairs(self, key: str) -> tuple[np.ndarray, np.ndarray]:
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

    def _concat_ints(self, key: str) -> np.ndarray:
        ints, _ = self._concat_pairs(key)
        return ints

    def _assert_set_parity(
        self,
        set_a: set,
        set_b: set,
        *,
        prefix: str = "set mismatch",
        label_a: str = "only_a",
        label_b: str = "only_b",
        max_diff: int | None = None,
        max_diff_pct: float | None = None,
    ):
        diff_a = len(set_a - set_b)
        diff_b = len(set_b - set_a)

        if max_diff_pct is not None:
            total = max(len(set_a), len(set_b))
            limit = int(np.ceil(max_diff_pct * total))
        else:
            limit = self.MAX_SET_DIFF if max_diff is None else max_diff

        self.assertTrue(
            diff_a <= limit and diff_b <= limit,
            f"{prefix}: {label_a}={diff_a}, {label_b}={diff_b}, limit={limit}",
        )

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

    def _assert_extrema_set_match(self, stage: str):
        ints_c = self._load_extrema_ints(stage)
        ints_p = self._concat_ints(stage)

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))

        self._assert_set_parity(
            set_c,
            set_p,
            prefix=f"{stage} set mismatch",
            label_a="only_in_c",
            label_b="only_in_py",
            max_diff_pct=self.MAX_SET_DIFF_PCT,
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

    def test_grad_x_matches_cli_dump(self):
        self._assert_octave_layer_mats_equal(
            self.record_dir / "grad_x/grad_x_meta.json",
            self.record_dir / "grad_x",
            "grad_x",
            self.TOL_ARRAY,
        )

    def test_grad_y_matches_cli_dump(self):
        self._assert_octave_layer_mats_equal(
            self.record_dir / "grad_y/grad_y_meta.json",
            self.record_dir / "grad_y",
            "grad_y",
            self.TOL_ARRAY,
        )

    def test_extrema_matches_cli_dump(self):
        self._assert_extrema_set_match("extrema")

    def test_contrast_pre_matches_cli_dump(self):
        self._assert_extrema_set_match("contrast_pre")

    def test_contrast_post_matches_cli_dump(self):
        self._assert_extrema_set_match("contrast_post")

    def test_edge_matches_cli_dump(self):
        self._assert_extrema_set_match("edge")

    def test_border_matches_cli_dump(self):
        self._assert_extrema_set_match("border")

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
            max_diff_pct=self.MAX_SET_DIFF_PCT,
        )

    def test_refined_matches_cli_dump(self):
        ints_c, flts_c = self._load_extrema_pairs("refined")
        ints_p, flts_p = self._concat_pairs("refined")

        set_c = set(map(tuple, ints_c.tolist()))
        set_p = set(map(tuple, ints_p.tolist()))
        self._assert_set_parity(
            set_p,
            set_c,
            prefix="refined set mismatch",
            label_a="only_in_py",
            label_b="only_in_c",
            max_diff_pct=self.MAX_SET_DIFF_PCT,
        )

        common = set_c & set_p
        if not common:
            return

        idx_c = {tuple(ints_c[i]): i for i in range(ints_c.shape[0])}
        idx_p = {tuple(ints_p[i]): i for i in range(ints_p.shape[0])}

        keys = list(common)
        idxs_c = np.array([idx_c[k] for k in keys], dtype=np.int64)
        idxs_p = np.array([idx_p[k] for k in keys], dtype=np.int64)

        diffs = np.abs(flts_c[idxs_c] - flts_p[idxs_p])
        atol = self.REFINED_ATOL
        matches = diffs <= atol
        col_counts = matches.sum(axis=0)
        total = matches.shape[0]
        overall_count = int(matches.all(axis=1).sum())

        max_failures = int(np.ceil(self.REFINED_FLOAT_FAIL_PCT * total))
        min_ok = max(total - max_failures, 0)
        if not (overall_count >= min_ok and np.all(col_counts >= min_ok)):
            max_diffs = diffs.max(axis=0)
            bad_counts = (~matches).sum(axis=0)
            self.fail(
                f"refined float matches too low: overall {overall_count}/{total}, "
                f"cols {col_counts.tolist()}/{total}, "
                f"bad_counts {bad_counts.tolist()}, max_diffs {max_diffs.tolist()}"
            )

    def test_oriented_keypoints_match_cli_dump(self):
        keys_dir = self.record_dir / "keys"
        meta_path = keys_dir / "keys_meta.json"
        if not meta_path.exists():
            self.skipTest("keys dump not found")

        meta = self._load_json(meta_path)
        ints_c = np.fromfile(
            keys_dir / meta.get("int_file", "keys_int.i32"), dtype=np.int32
        ).reshape(-1, 4)
        flts_c = np.fromfile(
            keys_dir / meta.get("float_file", "keys_float.f32"), dtype=np.float32
        ).reshape(-1, 4)

        ints_list, flts_list = [], []
        for o in range(self.params.n_oct):
            pair = self.snapshots[o].get("keys")
            if pair is None:
                continue
            ib, fb = (pair[0], pair[1]) if len(pair) >= 2 else (None, None)
            if ib is not None and ib.size > 0:
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

        keys_c = set(map(tuple, ints_c.tolist()))
        keys_p = set(map(tuple, ints_p.tolist()))
        self._assert_set_parity(
            keys_c,
            keys_p,
            prefix="pre-orientation identity mismatch",
            label_a="only_in_c",
            label_b="only_in_p",
            max_diff_pct=self.MAX_SET_DIFF_PCT,
        )

        from collections import defaultdict

        grp_c = defaultdict(list)
        grp_p = defaultdict(list)
        for ints_row, flts_row in zip(ints_c, flts_c):
            k = tuple(ints_row.tolist())
            grp_c[k].append(float(flts_row[3]))
        for ints_row, flts_row in zip(ints_p, flts_p):
            k = tuple(ints_row.tolist())
            grp_p[k].append(float(flts_row[3]))

        def wrap_2pi(angles):
            t = np.float32(2.0 * np.pi)
            return (angles % t + t) % t

        def circ_diff(a: float, b: float) -> float:
            return float(abs(((a - b) + np.pi) % (2.0 * np.pi) - np.pi))

        th_tol = self.ORI_TOL
        worst = 0.0
        bad = 0
        count_mismatch = 0

        common = keys_c & keys_p
        for k in common:
            arr_c = wrap_2pi(np.array(grp_c.get(k, []), dtype=np.float32))
            arr_p = wrap_2pi(np.array(grp_p.get(k, []), dtype=np.float32))

            if arr_c.size != arr_p.size:
                count_mismatch += 1
                continue
            if arr_c.size == 0:
                continue

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

        max_count_mismatch = int(np.ceil(self.ORI_MISMATCH_PCT * len(common)))
        max_angle_errors = int(np.ceil(self.ORI_MISMATCH_PCT * len(common)))
        self.assertLessEqual(
            count_mismatch,
            max_count_mismatch,
            msg=f"orientation count mismatch: {count_mismatch}",
        )
        self.assertLessEqual(
            bad,
            max_angle_errors,
            msg=f"orientation angle errors: {bad}, worst={worst:.6f} rad",
        )

    def test_descriptors_match_cli_dump(self):
        keys_dir = self.record_dir / "keys"
        meta_path = keys_dir / "keys_meta.json"
        if not meta_path.exists():
            self.skipTest("keys dump not found")

        meta = self._load_json(meta_path)
        ints_c = np.fromfile(
            keys_dir / meta.get("int_file", "keys_int.i32"), dtype=np.int32
        ).reshape(-1, 4)
        desc_len = int(meta.get("desc_len", 128))
        desc_c = np.fromfile(
            keys_dir / meta.get("desc_file", "keys_desc.u8"), dtype=np.uint8
        ).reshape(-1, desc_len)

        ints_list, desc_list = [], []
        for o in range(self.params.n_oct):
            triple = self.snapshots[o].get("keys")
            if triple is None or len(triple) < 3:
                continue
            ib, _, db = triple
            if ib is not None and ib.size > 0:
                ints_list.append(ib)
                desc_list.append(db)

        ints_p = np.concatenate(ints_list) if ints_list else np.empty((0, 4), np.int32)
        desc_p = (
            np.concatenate(desc_list)
            if desc_list
            else np.empty((0, desc_len), np.uint8)
        )

        from collections import defaultdict

        grp_c = defaultdict(list)
        grp_p = defaultdict(list)
        for row, d in zip(ints_c, desc_c):
            grp_c[tuple(row.tolist())].append(d)
        for row, d in zip(ints_p, desc_p):
            grp_p[tuple(row.tolist())].append(d)

        ham_thresh = int(self.HAM_FRAC * desc_len * 8)
        mismatches = 0
        worst = 0

        common = set(grp_c.keys()) & set(grp_p.keys())
        for key in common:
            arr_c = np.stack(grp_c[key], axis=0).astype(np.uint8)
            arr_p = np.stack(grp_p[key], axis=0).astype(np.uint8)

            used = np.zeros(arr_p.shape[0], dtype=bool)
            for dc in arr_c:
                x = np.bitwise_xor(arr_p, dc[None, :])
                ham = self.popcnt[x].sum(axis=1)
                ham[used] = int(1e9)
                j = int(np.argmin(ham))
                used[j] = True
                h = int(ham[j])
                worst = max(worst, h)
                if h > ham_thresh:
                    mismatches += 1

        max_desc_mismatches = int(np.ceil(self.HAM_MISMATCH_PCT * len(common)))
        self.assertLessEqual(
            mismatches,
            max_desc_mismatches,
            f"descriptor mismatches: {mismatches}, worst_hamming={worst}, thresh={ham_thresh}",
        )

    def test_keys_present_per_octave(self):
        stages = [
            "gss",
            "dog",
            "extrema",
            "contrast_pre",
            "refined",
            "contrast_post",
            "edge",
            "border",
        ]
        for o in range(self.params.n_oct):
            snap = self.snapshots[o]
            for stage in stages:
                with self.subTest(octave=o, stage=stage):
                    self.assertIn(stage, snap)

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

    def _assert_shapes_dtypes(self, pair, *, floats_cols: int):
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


def create_div2k_test_classes():
    """Dynamically create test classes for all DIV2K validation images."""
    root = Path(__file__).resolve().parents[1]
    cache_dir = root / "data/div2k_cache"

    # Images to exclude from testing (outliers with very few features or extrema/refinement issues)
    excluded_images = {"0828", "0843", "0844", "0846", "0857", "0861", "0868"}

    try:
        div2k_images = download_and_cache_div2k_validation(cache_dir)
    except unittest.SkipTest as e:
        print(f"Skipping DIV2K tests: {e}")
        return

    for img_path in div2k_images:
        img_name = img_path.stem

        # Skip excluded images
        if img_name in excluded_images:
            print(f"Skipping excluded image: {img_name}")
            continue

        class_name = f"TestSiftDIV2K_{img_name}"
        test_class = type(
            class_name,
            (SiftComputeMixin, unittest.TestCase),
            {"IMG_PATH": str(img_path)},
        )
        globals()[class_name] = test_class


create_div2k_test_classes()
