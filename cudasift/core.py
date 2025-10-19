"""CUDA SIFT detector implementation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda

from .kernels import (
    TX,
    TY,
    build_descriptors,
    compute_dog,
    compute_gss,
    detect_extrema,
    discard_near_the_border,
    discard_on_edge,
    discard_with_low_response,
    downsample_kernel,
    gaussian_blur,
    refine_extrema,
    reset_counters_kernel,
    upscale,
)
from .types import SiftData, SiftParams
from .utils import (
    create_sift_data,
    format_keypoints,
    read_gray_bt709,
    validate_image_dims,
)


def _compute_octave(
    data: SiftData,
    params: SiftParams,
    octave_index: int,
    stream,
) -> dict[str, Any] | None:
    snapshot: dict[str, Any] = {}
    record = params.record

    data.extrema.counter.copy_to_device(np.array([0, 0], dtype=np.int32), stream)

    if octave_index == 0:
        _set_seed(data, params, stream)
    else:
        _set_first_scale(data, params, octave_index, stream)

    snapshot["gss"], snapshot["grad_x"], snapshot["grad_y"] = compute_gss(
        data, params, octave_index, stream, record
    )
    snapshot["dog"] = compute_dog(data, params, octave_index, stream, record)
    snapshot["extrema"] = detect_extrema(data, params, octave_index, stream, record)
    snapshot["contrast_pre"] = discard_with_low_response(
        data, params, 0.8, octave_index, stream, record
    )
    snapshot["refined"] = refine_extrema(data, params, octave_index, stream, record)
    snapshot["contrast_post"] = discard_with_low_response(
        data, params, 1.0, octave_index, stream, record
    )
    snapshot["edge"] = discard_on_edge(data, params, octave_index, stream, record)
    snapshot["border"] = discard_near_the_border(
        data, params, octave_index, stream, record
    )
    snapshot["keys"] = build_descriptors(data, params, octave_index, stream, record)

    return snapshot


def _set_seed(data: SiftData, params: SiftParams, stream):
    assert params.sigma_min >= params.sigma_in
    upscale(data.input_img, data.seed_img, params.delta_min, stream)
    assert params.inc_sigmas is not None and params.gauss_kernels is not None
    sigma = float(params.inc_sigmas[0, 0])
    gauss_kernel, radius = params.gauss_kernels[sigma]
    gaussian_blur(
        data.seed_img,
        data.gss[0][0],
        data.scratch[0],
        stream,
        gauss_kernel,
        radius,
    )


def _set_first_scale(data: SiftData, params: SiftParams, octave_index: int, stream):
    src = data.gss[octave_index - 1][params.n_spo]
    dst = data.gss[octave_index][0]
    assert params.gss_shapes is not None
    height, width = params.gss_shapes[octave_index]
    grid = ((width + TX - 1) // TX, (height + TY - 1) // TY)
    downsample_kernel[grid, (TX, TY), stream](src, dst)


class SiftDetector:
    """CUDA-accelerated SIFT feature detector."""

    def __init__(
        self,
        img_dims: tuple[int, int],
        *,
        n_oct: int = -1,
        n_spo: int = 3,
        sigma_min: float = 0.8,
        max_keypoints: int = 100_000,
        record: bool = False,
        **kwargs,
    ):
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please ensure you have a CUDA-capable GPU "
                "and the necessary drivers installed."
            )

        self.params = SiftParams(
            img_dims=img_dims,
            n_oct=n_oct,
            n_spo=n_spo,
            sigma_min=sigma_min,
            max_keypoints=max_keypoints,
            record=record,
            **kwargs,
        )
        self.data = create_sift_data(self.params)
        self._stream = cuda.stream()
        self.record = bool(self.params.record)

    def _compute(self) -> list[dict[str, Any]]:
        """Execute SIFT computation pipeline."""
        snapshots: list[dict[str, Any]] = []

        reset_counters_kernel[1, 1, self._stream](
            self.data.extrema.counter, self.data.keypoints.counter
        )

        for octave_idx in range(self.params.n_oct):
            snapshot = _compute_octave(self.data, self.params, octave_idx, self._stream)
            if snapshot is not None:
                snapshots.append(snapshot)

        return snapshots

    def _transfer_results(self) -> None:
        """Transfer results from GPU to CPU."""
        self.data.keypoints.counter.copy_to_host(
            self.data.keypoints_host.counter, self._stream
        )
        self._stream.synchronize()

        num_keypoints = int(self.data.keypoints_host.counter[0])
        if num_keypoints > 0:
            num_to_copy = min(num_keypoints, self.params.max_keypoints)
            self.data.keypoints.int_buffer[:num_to_copy].copy_to_host(
                self.data.keypoints_host.int_buffer[:num_to_copy], self._stream
            )
            self.data.keypoints.float_buffer[:num_to_copy].copy_to_host(
                self.data.keypoints_host.float_buffer[:num_to_copy], self._stream
            )
            self.data.keypoints.descriptors[:num_to_copy].copy_to_host(
                self.data.keypoints_host.descriptors[:num_to_copy], self._stream
            )

    def detect(self, image: str | Path | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect SIFT keypoints and compute descriptors."""
        img = self._load_image(image)
        self.data.input_img.copy_to_device(img, self._stream)
        self._compute()
        self._transfer_results()
        self._warn_overflows()
        return format_keypoints(self.data.keypoints_host)

    def detect_with_snapshots(
        self, image: str | Path | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Detect keypoints and return intermediate computation snapshots."""
        if not self.record:
            warnings.warn(
                "Snapshots not enabled. Create detector with record=True.",
                stacklevel=2,
            )

        img = self._load_image(image)
        self.data.input_img.copy_to_device(img, self._stream)
        snapshots = self._compute()
        self._transfer_results()
        self._warn_overflows()
        keypoints, descriptors = format_keypoints(self.data.keypoints_host)
        return keypoints, descriptors, snapshots

    def _load_image(self, image: str | Path | np.ndarray) -> np.ndarray:
        """Load and validate image."""
        if isinstance(image, (str, Path)):
            img = read_gray_bt709(image)
        else:
            img = image
        validate_image_dims(img, self.params.img_dims)
        if img.dtype != np.float32:
            img = np.asarray(img, dtype=np.float32)
        return img

    def _warn_overflows(self) -> None:
        """Warn if keypoint or extrema buffers overflowed."""
        self._stream.synchronize()

        if int(self.data.keypoints_host.counter[2]) > 0:
            warnings.warn(
                f"Keypoint overflow: {int(self.data.keypoints_host.counter[2])} dropped "
                f"(capacity {self.params.max_keypoints})",
                stacklevel=3,
            )

        ext = np.empty(2, dtype=np.int32)
        self.data.extrema.counter.copy_to_host(ext, self._stream)
        self._stream.synchronize()

        if int(ext[1]) > 0:
            warnings.warn(
                f"Extrema overflow: {int(ext[1])} dropped "
                f"(capacity {self.params.max_extrema})",
                stacklevel=3,
            )
