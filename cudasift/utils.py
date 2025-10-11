"""
Utility functions for CUDA SIFT implementation.
"""

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
    """
    Read an image and convert to grayscale using BT.709 standard.

    The BT.709 standard is recommended for HDTV and provides better
    perceptual accuracy than simple averaging of RGB channels.

    Args:
        path: Path to the image file

    Returns:
        Grayscale image as float32 array in range [0, 1]

    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image cannot be decoded
    """
    path_str = str(path)

    if not Path(path_str).exists():
        raise FileNotFoundError(f"Image file not found: {path_str}")

    img_buffer = np.fromfile(path_str, np.uint8)
    img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode image: {path_str}")

    # Convert BGR to grayscale using BT.709 coefficients
    gray = (img.astype(np.float32) * W709_BGR).sum(axis=2) / 256.0
    return gray


def check_cuda_available() -> bool:
    """
    Check if CUDA is available for computation.

    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        return cuda.is_available()
    except Exception:
        return False


def get_cuda_device_info() -> dict[str, str | int] | None:
    """
    Get information about the current CUDA device.

    Returns:
        Dictionary with device information, or None if CUDA is not available
    """
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
    """
    Validate that image dimensions match expected dimensions.

    Args:
        img: Input image array
        expected_dims: Expected (height, width)

    Raises:
        ValueError: If dimensions don't match
    """
    if img.shape[:2] != expected_dims:
        raise ValueError(f"Image dimensions {img.shape[:2]} don't match expected {expected_dims}")


def create_extrema(params: SiftParams) -> Extrema:
    """
    Allocate GPU memory for extrema detection.

    Args:
        params: SIFT parameters

    Returns:
        Extrema data structure with allocated GPU buffers
    """
    from .types import Extrema, device_array

    return Extrema(
        float_buffer=device_array((params.max_extrema, 4), np.float32),
        int_buffer=device_array((params.max_extrema, 4), np.int32),
    )


def create_keypoints(params: SiftParams) -> Keypoints:
    """
    Allocate GPU memory for keypoint storage.

    Args:
        params: SIFT parameters

    Returns:
        Keypoints data structure with allocated GPU buffers
    """
    from .types import Keypoints, device_array

    n = params.max_keypoints
    return Keypoints(
        int_buffer=device_array((n, 4), np.int32),
        float_buffer=device_array((n, 4), np.float32),
        descriptors=device_array((n, 128), np.uint8),
    )


def create_keypoints_host(params: SiftParams) -> KeypointsHost:
    """
    Allocate CPU memory for keypoint storage.

    Args:
        params: SIFT parameters

    Returns:
        KeypointsHost data structure with allocated CPU buffers
    """
    from .types import KeypointsHost

    n = params.max_keypoints
    return KeypointsHost(
        int_buffer=np.empty((n, 4), dtype=np.int32),
        float_buffer=np.empty((n, 4), dtype=np.float32),
        descriptors=np.empty((n, 128), dtype=np.uint8),
    )


def alloc_octave_tensors(params: SiftParams, octave_index: int):
    """
    Allocate GPU memory for a single octave in the scale space pyramid.

    Args:
        params: SIFT parameters
        octave_index: Index of the octave to allocate

    Returns:
        Tuple of (gss, dog, scratch, gx, gy) arrays for the octave
    """
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
    """
    Allocate all GPU memory needed for SIFT computation.

    Args:
        params: SIFT parameters

    Returns:
        SiftData structure with all allocated buffers
    """
    from .types import SiftData, device_array

    # Allocate per-octave tensors
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
    """
    Extract keypoints and descriptors in a user-friendly format.

    Args:
        keypoints_host: KeypointsHost data structure

    Returns:
        Tuple of (keypoints, descriptors) where:
        - keypoints: Nx4 array with columns [x, y, scale, orientation]
        - descriptors: Nx128 array of uint8 descriptors
    """
    num_kpts = int(keypoints_host.counter[0])

    if num_kpts == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 128), dtype=np.uint8)

    # Extract keypoint data: [y_world, x_world, sigma, orientation]
    kpts_data = keypoints_host.float_buffer[:num_kpts]

    # Reorder to [x, y, sigma, orientation] for user convenience
    keypoints = np.column_stack(
        [
            kpts_data[:, 1],  # x
            kpts_data[:, 0],  # y
            kpts_data[:, 2],  # sigma (scale)
            kpts_data[:, 3],  # orientation
        ]
    ).astype(np.float32)

    descriptors = keypoints_host.descriptors[:num_kpts].copy()

    return keypoints, descriptors
