"""
CUDA-accelerated SIFT (Scale-Invariant Feature Transform) implementation.

This package provides a high-performance GPU-accelerated implementation of SIFT
feature detection and extraction using CUDA and Numba.

Basic Usage:
    >>> import cudasift
    >>>
    >>> # Create a detector for images of specific dimensions
    >>> detector = cudasift.SiftDetector(img_dims=(480, 640))
    >>>
    >>> # Detect features from an image
    >>> keypoints, descriptors = detector.detect("image.png")
    >>>
    >>> # Keypoints: Nx4 array with [x, y, scale, orientation]
    >>> # Descriptors: Nx128 array of uint8 SIFT descriptors
    >>> print(f"Found {len(keypoints)} keypoints")

Advanced Usage:
    >>> # Customize detection parameters
    >>> detector = cudasift.SiftDetector(
    ...     img_dims=(480, 640),
    ...     n_oct=5,              # Number of octaves
    ...     n_spo=3,              # Scales per octave
    ...     sigma_min=0.8,        # Minimum sigma
    ...     max_keypoints=50000   # Maximum keypoints
    ... )
    >>>
    >>> # Process multiple images
    >>> results = detector.detect_batch(["img1.png", "img2.png"])
    >>>
    >>> # Get intermediate computation results for debugging
    >>> detector_debug = cudasift.SiftDetector(img_dims=(480, 640), record=True)
    >>> kpts, desc, snapshots = detector_debug.detect_with_snapshots("image.png")

Utility Functions:
    >>> # Check CUDA availability
    >>> if cudasift.check_cuda_available():
    ...     print("CUDA is available")
    >>>
    >>> # Get device information
    >>> info = cudasift.get_cuda_device_info()
    >>> print(f"Using: {info['name']}")

For more information, see the documentation at:
https://github.com/yourusername/cudasift
"""

from .core import Sift, SiftDetector
from .types import Extrema, Keypoints, KeypointsHost, SiftData, SiftParams
from .utils import (
    check_cuda_available,
    get_cuda_device_info,
    read_gray_bt709,
)
from .version import __version__

__all__ = [
    # Main API
    "SiftDetector",
    "Sift",
    # Data structures
    "SiftParams",
    "SiftData",
    "Keypoints",
    "KeypointsHost",
    "Extrema",
    # Utilities
    "check_cuda_available",
    "get_cuda_device_info",
    "read_gray_bt709",
    # Version
    "__version__",
]

# Module metadata
__author__ = "SIFT CUDA Team"
__license__ = "MIT"
__description__ = "CUDA-accelerated SIFT feature detection and extraction"
