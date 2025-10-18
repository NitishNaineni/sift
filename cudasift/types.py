"""SIFT data structures and types."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray


def device_array(shape: tuple[int, ...] | int, dtype: Any) -> DeviceNDArray:
    from typing import cast

    return cast(DeviceNDArray, cuda.device_array(shape, dtype))


@dataclass
class SiftParams:
    """Parameters for SIFT feature detection and extraction."""

    img_dims: tuple[int, int]
    record: bool = False
    n_oct: int = -1
    n_spo: int = 3
    sigma_in: float = 0.5
    delta_min: float = 0.5
    sigma_min: float = 0.8
    max_extrema: int = 100_000
    max_keypoints: int = 100_000
    C_dog: float = 0.013333333
    C_edge: float = 10.0
    lambda_ori: float = 1.5

    sigmas: np.ndarray | None = None
    gss_shapes: np.ndarray | None = None
    inc_sigmas: np.ndarray | None = None
    gauss_kernels: dict[float, tuple[DeviceNDArray, int]] | None = None

    def __post_init__(self) -> None:
        self._update_octave_count()
        self._scale_invariant_C_dog()
        self.sigmas = self._make_sigmas()
        self.gss_shapes = self._make_gss_shapes()
        self.inc_sigmas = self._make_sigma_increments()
        self._precompute_gaussian_kernels()

    def _update_octave_count(self) -> None:
        max_n_oct = math.floor(math.log2(min(self.img_dims) / self.delta_min / 12)) + 1
        self.n_oct = max_n_oct if self.n_oct == -1 else min(max_n_oct, self.n_oct)

    def _scale_invariant_C_dog(self) -> None:
        kn = np.exp(np.log(2) / self.n_spo)
        k3 = np.exp(np.log(2) / 3.0)
        self.C_dog *= (kn - 1) / (k3 - 1)

    def _make_sigmas(self) -> np.ndarray:
        num_octaves = self.n_oct
        num_scales_total = self.n_spo + 3
        octave_indices = np.arange(num_octaves, dtype=np.float32)[:, None]
        scale_offsets = (np.arange(num_scales_total, dtype=np.float32) / self.n_spo)[None, :]
        return (self.sigma_min * (2.0 ** (octave_indices + scale_offsets))).astype(np.float32)

    def _make_gss_shapes(self) -> np.ndarray:
        base = np.array(
            [
                int(self.img_dims[0] / self.delta_min),
                int(self.img_dims[1] / self.delta_min),
            ],
            dtype=np.int64,
        )
        hw = base // (1 << np.arange(self.n_oct, dtype=np.int64))[:, None]
        return hw

    def _make_sigma_increments(self) -> np.ndarray:
        assert self.sigmas is not None, "sigmas must be initialized"
        sig = self.sigmas.astype(np.float32)
        num_octaves, num_scales_total = sig.shape
        inc = np.empty_like(sig, dtype=np.float32)

        prev = np.empty_like(sig, dtype=np.float32)
        prev[:, 1:] = sig[:, :-1]
        prev[0, 0] = np.float32(self.sigma_in)
        if num_octaves > 1:
            prev[1:, 0] = sig[:-1, self.n_spo]

        deltas = (self.delta_min * (2.0 ** np.arange(num_octaves, dtype=np.float32)))[:, None]

        diff2 = sig * sig - prev * prev
        np.maximum(diff2, 0.0, out=diff2, dtype=np.float32)
        np.sqrt(diff2, out=diff2)
        inc[:, :] = diff2 / deltas
        return inc

    def _precompute_gaussian_kernels(self) -> None:
        from .kernels import gaussian_symm_kernel

        kernels: dict[float, tuple[DeviceNDArray, int]] = {}
        if self.inc_sigmas is None:
            self.gauss_kernels = kernels
            return
        unique_sigmas = np.unique(self.inc_sigmas.astype(np.float32))
        for sig in unique_sigmas.tolist():
            g_dev, r = gaussian_symm_kernel(float(sig))
            kernels[float(sig)] = (g_dev, r)
        self.gauss_kernels = kernels


@dataclass
class Extrema:
    """Storage for detected extrema (before keypoint refinement)."""

    int_buffer: DeviceNDArray
    float_buffer: DeviceNDArray
    counter: DeviceNDArray = field(
        default_factory=lambda: cuda.to_device(np.zeros(2, dtype=np.int32))
    )


@dataclass
class Keypoints:
    """Storage for detected and refined keypoints on GPU."""

    int_buffer: DeviceNDArray
    float_buffer: DeviceNDArray
    descriptors: DeviceNDArray
    counter: DeviceNDArray = field(
        default_factory=lambda: cuda.to_device(np.zeros(3, dtype=np.int32))
    )


@dataclass
class KeypointsHost:
    """Storage for keypoints on CPU (host memory)."""

    int_buffer: np.ndarray
    float_buffer: np.ndarray
    descriptors: np.ndarray
    counter: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int32))

    def copy(self) -> KeypointsHost:
        return KeypointsHost(
            int_buffer=self.int_buffer.copy(),
            float_buffer=self.float_buffer.copy(),
            descriptors=self.descriptors.copy(),
            counter=self.counter.copy(),
        )


@dataclass
class SiftData:
    """Complete SIFT computation data structures."""

    input_img: DeviceNDArray
    seed_img: DeviceNDArray
    scratch: tuple[DeviceNDArray, ...]
    gss: tuple[DeviceNDArray, ...]
    dog: tuple[DeviceNDArray, ...]
    gx: tuple[DeviceNDArray, ...]
    gy: tuple[DeviceNDArray, ...]
    extrema: Extrema
    keypoints: Keypoints
    keypoints_host: KeypointsHost
