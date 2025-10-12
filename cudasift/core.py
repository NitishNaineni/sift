"""CUDA SIFT detector implementation."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cupy as cp
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
    """
    Compute SIFT features for a single octave.

    Args:
        data: SIFT data structures
        params: SIFT parameters
        octave_index: Index of the octave to compute
        stream: CUDA stream for computation

    Returns:
        Snapshot dictionary if recording is enabled, None otherwise
    """
    snapshot: dict[str, Any] = {}
    record = bool(params.record)

    # Reset extrema counter for this octave
    data.extrema.counter.copy_to_device(np.array([0, 0], dtype=np.int32), stream)

    # Set up the first scale of this octave
    if octave_index == 0:
        _set_seed(data, params, stream)
    else:
        _set_first_scale(data, params, octave_index, stream)

    # Compute Gaussian scale space and gradients
    snapshot["gss"], snapshot["grad_x"], snapshot["grad_y"] = compute_gss(
        data, params, octave_index, stream, record
    )

    # Compute Difference of Gaussians
    snapshot["dog"] = compute_dog(data, params, octave_index, stream, record)

    # Detect extrema
    snapshot["extrema"] = detect_extrema(data, params, octave_index, stream, record)

    # Filter by contrast (pre-refinement)
    snapshot["contrast_pre"] = discard_with_low_response(
        data, params, 0.8, octave_index, stream, record
    )

    # Refine extrema to subpixel accuracy
    snapshot["refined"] = refine_extrema(data, params, octave_index, stream, record)

    # Filter by contrast (post-refinement)
    snapshot["contrast_post"] = discard_with_low_response(
        data, params, 1.0, octave_index, stream, record
    )

    # Filter edge responses
    snapshot["edge"] = discard_on_edge(data, params, octave_index, stream, record)

    # Filter keypoints near the border
    snapshot["border"] = discard_near_the_border(data, params, octave_index, stream, record)

    # Compute orientations and descriptors
    snapshot["keys"] = build_descriptors(data, params, octave_index, stream, record)

    return snapshot


def _set_seed(data: SiftData, params: SiftParams, stream):
    """Initialize the seed image for the first octave."""
    assert params.sigma_min >= params.sigma_in
    upscale(data.input_img, data.seed_img, params.delta_min, stream)
    assert params.inc_sigmas is not None and params.gauss_kernels is not None
    sigma = params.inc_sigmas[0, 0]
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
    """Initialize the first scale of an octave by downsampling the previous octave."""
    src = data.gss[octave_index - 1][params.n_spo]
    dst = data.gss[octave_index][0]
    assert params.gss_shapes is not None
    height, width = params.gss_shapes[octave_index]
    grid = ((width + TX - 1) // TX, (height + TY - 1) // TY)
    downsample_kernel[grid, (TX, TY), stream](src, dst)


class SiftDetector:
    """
    CUDA-accelerated SIFT feature detector.

    This class provides a high-level API for detecting and extracting SIFT features
    from images using GPU acceleration via CUDA.

    Example:
        >>> detector = SiftDetector(img_dims=(480, 640))
        >>> keypoints, descriptors = detector.detect("image.png")
        >>> print(f"Found {len(keypoints)} keypoints")

    Attributes:
        params: SIFT parameters controlling detection behavior
        data: Internal GPU data structures
    """

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
        """
        Initialize the SIFT detector.

        Args:
            img_dims: Image dimensions as (height, width)
            n_oct: Number of octaves (-1 for automatic)
            n_spo: Number of scales per octave
            sigma_min: Minimum sigma for SIFT
            max_keypoints: Maximum number of keypoints to extract
            record: Whether to record intermediate computation results
            **kwargs: Additional parameters passed to SiftParams

        Raises:
            RuntimeError: If CUDA is not available
        """
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please ensure you have a CUDA-capable GPU "
                "and the necessary drivers installed."
            )

        # Create parameters
        self.params = SiftParams(
            img_dims=img_dims,
            n_oct=n_oct,
            n_spo=n_spo,
            sigma_min=sigma_min,
            max_keypoints=max_keypoints,
            record=record,
            **kwargs,
        )

        # Allocate GPU memory
        self.data = create_sift_data(self.params)
        self._stream = cuda.stream()
        self.record = bool(self.params.record)

        # Warm up and optionally create CUDA graph
        self._warmup()

    def _warmup(self) -> None:
        """Warm up the detector and create CUDA graph if not recording."""
        h, w = self.params.img_dims
        dummy = np.random.rand(h, w).astype(np.float32)
        self.data.input_img.copy_to_device(dummy, self._stream)
        self._exec_graph()

        if not self.record:
            # Create CUDA graph for faster repeated execution
            ptr = int(self._stream.handle.value)
            self._ext_stream = cp.cuda.ExternalStream(ptr)
            with self._ext_stream:
                self._ext_stream.begin_capture()
                self._exec_graph()
                self._graph = self._ext_stream.end_capture()
                self._graph.upload(self._ext_stream)

    def _exec_graph(self) -> list[dict[str, Any]]:
        """Execute the SIFT computation graph."""
        snapshots: list[dict[str, Any]] = []

        # Reset counters once (not per octave)
        reset_counters_kernel[1, 1, self._stream](
            self.data.extrema.counter, self.data.keypoints.counter
        )

        # Process each octave sequentially (dependencies exist)
        for o in range(self.params.n_oct):
            snapshot = _compute_octave(self.data, self.params, o, self._stream)
            if snapshot is not None:
                snapshots.append(snapshot)

        # Single batch copy to host at the end (async)
        self.data.keypoints.int_buffer.copy_to_host(
            self.data.keypoints_host.int_buffer, self._stream
        )
        self.data.keypoints.float_buffer.copy_to_host(
            self.data.keypoints_host.float_buffer, self._stream
        )
        self.data.keypoints.descriptors.copy_to_host(
            self.data.keypoints_host.descriptors, self._stream
        )
        self.data.keypoints.counter.copy_to_host(self.data.keypoints_host.counter, self._stream)

        return snapshots

    def detect(self, image: str | Path | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect and extract SIFT features from an image.

        Args:
            image: Input image as file path or numpy array

        Returns:
            Tuple of (keypoints, descriptors) where:
            - keypoints: Nx4 array with columns [x, y, scale, orientation]
            - descriptors: Nx128 array of uint8 descriptors

        Raises:
            ValueError: If image dimensions don't match
            FileNotFoundError: If image file doesn't exist
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = read_gray_bt709(image)
        else:
            img = image

        # Validate dimensions
        validate_image_dims(img, self.params.img_dims)

        # Copy to device and compute
        self.data.input_img.copy_to_device(img.astype(np.float32), self._stream)

        if not self.record:
            # Use CUDA graph for faster execution
            with self._ext_stream:
                self._graph.launch(self._ext_stream)
        else:
            self._exec_graph()

        # Check for overflows
        self._warn_overflows()

        # Format and return results
        return format_keypoints(self.data.keypoints_host)

    def detect_with_snapshots(
        self, image: str | Path | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """
        Detect SIFT features and return intermediate computation results.

        Note: This method is slower as it records intermediate results.

        Args:
            image: Input image as file path or numpy array

        Returns:
            Tuple of (keypoints, descriptors, snapshots)
        """
        if not self.record:
            warnings.warn(
                "Snapshots not enabled. Create detector with record=True to get snapshots.",
                stacklevel=2,
            )

        # Load image
        if isinstance(image, (str, Path)):
            img = read_gray_bt709(image)
        else:
            img = image

        # Validate dimensions
        validate_image_dims(img, self.params.img_dims)

        # Copy to device and compute
        self.data.input_img.copy_to_device(img.astype(np.float32), self._stream)
        snapshots = self._exec_graph()

        # Check for overflows
        self._warn_overflows()

        # Format and return results
        keypoints, descriptors = format_keypoints(self.data.keypoints_host)
        return keypoints, descriptors, snapshots

    def detect_batch(
        self, images: Iterable[str | Path | np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Detect SIFT features from multiple images.

        Args:
            images: Iterable of image paths or arrays

        Returns:
            List of (keypoints, descriptors) tuples
        """
        return [self.detect(img) for img in images]

    def _warn_overflows(self) -> None:
        """Check for buffer overflows and issue warnings."""
        # Ensure all device-to-host copies have completed
        self._stream.synchronize()

        # Check keypoint overflow
        kctr = self.data.keypoints_host.counter
        if int(kctr[2]) > 0:
            warnings.warn(
                f"Keypoint overflow: {int(kctr[2])} entries dropped "
                f"(capacity {self.params.max_keypoints}). "
                f"Consider increasing max_keypoints parameter.",
                stacklevel=3,
            )

        # Check extrema overflow
        ext = np.empty(2, dtype=np.int32)
        self.data.extrema.counter.copy_to_host(ext, self._stream)
        self._stream.synchronize()
        if int(ext[1]) > 0:
            warnings.warn(
                f"Extrema overflow: {int(ext[1])} entries dropped "
                f"(capacity {self.params.max_extrema}). "
                f"Consider increasing max_extrema parameter.",
                stacklevel=3,
            )
