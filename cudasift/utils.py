"""SIFT utility functions."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

if TYPE_CHECKING:
    from .types import (
        Extrema,
        Keypoints,
        KeypointsHost,
        SiftData,
        SiftParams,
    )

# Set environment variables for optimal CUDA performance
os.environ["NUMBA_CUDA_ARRAY_INTERFACE_SYNC"] = "0"

# Filter out common warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"pynvjitlink")
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# BT.709 coefficients for RGB to grayscale conversion
W709_BGR = np.array([0.072192315360734, 0.715168678767756, 0.212639005871510], dtype=np.float32)


def read_gray_bt709(path: str | Path) -> np.ndarray:
    path_str = str(path)
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Image file not found: {path_str}")
    img_buffer = np.fromfile(path_str, np.uint8)
    img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {path_str}")
    gray = (img.astype(np.float32) * W709_BGR).sum(axis=2) / 256.0
    return gray


def check_cuda_available() -> bool:
    try:
        return cuda.is_available()
    except Exception:
        return False


def get_cuda_device_info() -> dict[str, str | int] | None:
    if not check_cuda_available():
        return None
    try:
        device = cuda.get_current_device()
        return {
            "name": device.name.decode() if isinstance(device.name, bytes) else device.name,
            "compute_capability": f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
            "max_block_dim_x": device.MAX_BLOCK_DIM_X,
            "max_block_dim_y": device.MAX_BLOCK_DIM_Y,
            "max_grid_dim_x": device.MAX_GRID_DIM_X,
            "max_grid_dim_y": device.MAX_GRID_DIM_Y,
        }
    except Exception:
        return None


def validate_image_dims(img: np.ndarray, expected_dims: tuple[int, int]) -> None:
    if img.shape[:2] != expected_dims:
        raise ValueError(f"Image dimensions {img.shape[:2]} don't match expected {expected_dims}")


def create_extrema(params: SiftParams) -> Extrema:
    from .types import Extrema, device_array

    return Extrema(
        float_buffer=device_array((params.max_extrema, 4), np.float32),
        int_buffer=device_array((params.max_extrema, 4), np.int32),
    )


def create_keypoints(params: SiftParams) -> Keypoints:
    from .types import Keypoints, device_array

    n = params.max_keypoints
    return Keypoints(
        int_buffer=device_array((n, 4), np.int32),
        float_buffer=device_array((n, 4), np.float32),
        descriptors=device_array((n, 128), np.uint8),
    )


def create_keypoints_host(params: SiftParams) -> KeypointsHost:
    from .types import KeypointsHost

    n = params.max_keypoints
    return KeypointsHost(
        int_buffer=cuda.pinned_array((n, 4), dtype=np.int32),
        float_buffer=cuda.pinned_array((n, 4), dtype=np.float32),
        descriptors=cuda.pinned_array((n, 128), dtype=np.uint8),
        counter=cuda.pinned_array(3, dtype=np.int32),
    )


def alloc_octave_tensors(params: SiftParams, octave_index: int):
    from .types import device_array

    assert params.gss_shapes is not None, "gss_shapes must be initialized"
    height, width = params.gss_shapes[octave_index]
    num_gss_scales = params.n_spo + 3
    num_dog_scales = params.n_spo + 2
    gss = device_array((num_gss_scales, height, width), np.float32)
    dog = device_array((num_dog_scales, height, width), np.float32)
    scratch = device_array((height, width), np.float32)
    gx = device_array((num_gss_scales, height, width), np.float32)
    gy = device_array((num_gss_scales, height, width), np.float32)
    return gss, dog, scratch, gx, gy


def create_sift_data(params: SiftParams) -> SiftData:
    from .types import SiftData, device_array

    gss, dog, scratch, gx, gy = zip(
        *(alloc_octave_tensors(params, octave_index) for octave_index in range(params.n_oct))
    )
    assert params.gss_shapes is not None, "gss_shapes must be initialized"
    h0, w0 = params.gss_shapes[0]
    return SiftData(
        input_img=device_array(params.img_dims, np.float32),
        seed_img=device_array((h0, w0), np.float32),
        scratch=tuple(scratch),
        gss=tuple(gss),
        dog=tuple(dog),
        gx=tuple(gx),
        gy=tuple(gy),
        extrema=create_extrema(params),
        keypoints=create_keypoints(params),
        keypoints_host=create_keypoints_host(params),
    )


def format_keypoints(keypoints_host: KeypointsHost) -> tuple[np.ndarray, np.ndarray]:
    num_kpts = int(keypoints_host.counter[0])
    if num_kpts == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 128), dtype=np.uint8)
    kpts_data = keypoints_host.float_buffer[:num_kpts]
    keypoints = np.column_stack(
        [
            kpts_data[:, 1],
            kpts_data[:, 0],
            kpts_data[:, 2],
            kpts_data[:, 3],
        ]
    ).astype(np.float32)
    descriptors = keypoints_host.descriptors[:num_kpts].copy()
    return keypoints, descriptors
