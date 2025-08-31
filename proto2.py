from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import cupy as cp
import cv2
import numba
import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda import libdevice as ld
from numba.cuda.cudadrv.devicearray import DeviceNDArray

os.environ["NUMBA_CUDA_ARRAY_INTERFACE_SYNC"] = "0"

warnings.filterwarnings("ignore", category=UserWarning, message=r"pynvjitlink")
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

TX, TY = 16, 16
TWO_PI = np.float32(6.28318530718)
ORI_BINS = 36
NHIST, NORIBIN = 4, 8
NHIST2 = NHIST * NHIST
DESC_LEN = NHIST2 * NORIBIN
LAMBDA_DESC = numba.float32(6.0)
ORI_THRESHOLD = numba.float32(0.8)
W709_BGR = np.array(
    [0.072192315360734, 0.715168678767756, 0.212639005871510], dtype=np.float32
)

BLUR_TH = 128
MAX_GAUSS_RADIUS = 16
GRAD_TILE_SIZE = (TX + 2) * (TY + 2)

GAUSS_HORZ_TILE_SIZE = BLUR_TH + 2 * MAX_GAUSS_RADIUS
GAUSS_COEFF_TILE_SIZE = MAX_GAUSS_RADIUS + 1

GAUSS_VERT_TILE_H = TY + 2 * MAX_GAUSS_RADIUS
GAUSS_VERT_TILE_SIZE = GAUSS_VERT_TILE_H * TX


def read_gray_bt709(path: str) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    return (im.astype(np.float32) * W709_BGR).sum(axis=2) / 256.0


@dataclass
class SiftParams:
    img_dims: tuple[int, int]
    depth_dims: tuple[int, int]
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

    min_aspect: float = 0.1
    max_depth: float = 10.0

    sigmas: np.ndarray | None = None
    gss_shapes: np.ndarray | None = None
    inc_sigmas: np.ndarray | None = None
    gauss_kernels: Dict[float, tuple[DeviceNDArray, int]] | None = None

    def __post_init__(self) -> None:
        self._update_octave_count()
        self._scale_invariant_C_dog()
        self.sigmas = self._make_sigmas()
        self.gss_shapes = self._make_gss_shapes()
        self.inc_sigmas = self._make_sigma_increments()
        self.gauss_kernels = self._precompute_gaussian_kernels()

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
        scale_offsets = (np.arange(num_scales_total, dtype=np.float32) / self.n_spo)[
            None, :
        ]
        return (self.sigma_min * (2.0 ** (octave_indices + scale_offsets))).astype(
            np.float32
        )

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
        sig = self.sigmas.astype(np.float32)
        num_octaves, num_scales_total = sig.shape
        inc = np.empty_like(sig, dtype=np.float32)

        prev = np.empty_like(sig, dtype=np.float32)
        prev[:, 1:] = sig[:, :-1]
        prev[0, 0] = np.float32(self.sigma_in)
        if num_octaves > 1:
            prev[1:, 0] = sig[:-1, self.n_spo]

        deltas = (self.delta_min * (2.0 ** np.arange(num_octaves, dtype=np.float32)))[
            :, None
        ]

        diff2 = sig * sig - prev * prev
        np.maximum(diff2, 0.0, out=diff2, dtype=np.float32)
        np.sqrt(diff2, out=diff2)
        inc[:, :] = diff2 / deltas
        return inc

    def _precompute_gaussian_kernels(self) -> Dict[float, tuple[DeviceNDArray, int]]:
        kernels: Dict[float, tuple[DeviceNDArray, int]] = {}
        if self.inc_sigmas is None:
            return kernels
        unique_sigmas = np.unique(self.inc_sigmas.astype(np.float32))
        for sig in unique_sigmas.tolist():
            g_dev, r = gaussian_symm_kernel(float(sig))
            kernels[float(sig)] = (g_dev, r)
        return kernels


@dataclass
class Extrema:
    int_buffer: DeviceNDArray  # o, s, y_int, x_int
    float_buffer: DeviceNDArray  # y_world, x_world, sigma, dog_val
    counter: DeviceNDArray = field(  # extrema count, overflow count
        default_factory=lambda: cuda.to_device(np.zeros(2, dtype=np.int32))
    )


@dataclass
class Keypoints:
    int_buffer: DeviceNDArray  # o, s, y_int, x_int
    float_buffer: DeviceNDArray  # y_world, x_world, sigma, orientation
    descriptors: DeviceNDArray  # 128-dim SIFT descriptor per keypoint (uint8)
    tilt_map: DeviceNDArray  # <- NEW
    # keypoint count, overflow count
    counter: DeviceNDArray = field(
        default_factory=lambda: cuda.to_device(np.zeros(3, dtype=np.int32))
    )


@dataclass
class KeypointsHost:
    int_buffer: np.ndarray  # o, s, y_int, x_int
    float_buffer: np.ndarray  # y_world, x_world, sigma, orientation
    descriptors: np.ndarray  # 128-dim SIFT descriptor per keypoint (uint8)
    tilt_map: np.ndarray
    # keypoint count, overflow count
    counter: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int32))

    def copy(self) -> "KeypointsHost":
        return KeypointsHost(
            int_buffer=self.int_buffer.copy(),
            float_buffer=self.float_buffer.copy(),
            descriptors=self.descriptors.copy(),
            tilt_map=self.tilt_map.copy(),
            counter=self.counter.copy(),
        )


@dataclass
class SiftData:
    input_img: DeviceNDArray
    input_depth: DeviceNDArray
    seed_img: DeviceNDArray
    scratch: tuple[DeviceNDArray, ...]
    gss: tuple[DeviceNDArray, ...]
    dog: tuple[DeviceNDArray, ...]
    gx: tuple[DeviceNDArray, ...]
    gy: tuple[DeviceNDArray, ...]
    depth: tuple[DeviceNDArray, ...]
    extrema: Extrema
    keypoints: Keypoints
    keypoints_host: KeypointsHost
    tilt_map: DeviceNDArray


def _alloc_octave_tensors(params: SiftParams, octave_index: int):
    height, width = params.gss_shapes[octave_index]
    num_gss_scales = params.n_spo + 3
    num_dog_scales = params.n_spo + 2

    gss = cuda.device_array((num_gss_scales, height, width), np.float32)
    dog = cuda.device_array((num_dog_scales, height, width), np.float32)
    scratch = cuda.device_array((height, width), np.float32)
    gx = cuda.device_array((num_gss_scales, height, width), np.float32)
    gy = cuda.device_array((num_gss_scales, height, width), np.float32)
    depth = cuda.device_array((height, width), np.float32)

    return gss, dog, scratch, gx, gy, depth


def create_extrema(params: SiftParams) -> Extrema:
    return Extrema(
        float_buffer=cuda.device_array((params.max_extrema, 4), np.float32),
        int_buffer=cuda.device_array((params.max_extrema, 4), np.int32),
    )


def create_keypoints(params: SiftParams) -> Keypoints:
    n = params.max_keypoints
    return Keypoints(
        int_buffer=cuda.device_array((n, 4), np.int32),
        float_buffer=cuda.device_array((n, 4), np.float32),
        descriptors=cuda.device_array((n, 128), np.uint8),
        tilt_map=cuda.device_array((n, 2, 2), np.float32),
    )


def create_keypoints_host(params: SiftParams) -> KeypointsHost:
    n = params.max_keypoints
    return KeypointsHost(
        int_buffer=np.empty((n, 4), dtype=np.int32),
        float_buffer=np.empty((n, 4), dtype=np.float32),
        descriptors=np.empty((n, 128), dtype=np.uint8),
        tilt_map=np.empty((n, 2, 2), dtype=np.float32),
    )


def create_sift_data(params: SiftParams) -> SiftData:
    gss, dog, scratch, gx, gy, depth = zip(
        *(
            _alloc_octave_tensors(params, octave_index)
            for octave_index in range(params.n_oct)
        )
    )
    h0, w0 = params.gss_shapes[0]
    return SiftData(
        input_img=cuda.device_array(params.img_dims, np.float32),
        input_depth=cuda.device_array(params.depth_dims, np.float32),
        seed_img=cuda.device_array((h0, w0), np.float32),
        scratch=tuple(scratch),
        gss=tuple(gss),
        dog=tuple(dog),
        gx=tuple(gx),
        gy=tuple(gy),
        depth=tuple(depth),
        extrema=create_extrema(params),
        keypoints=create_keypoints(params),
        keypoints_host=create_keypoints_host(params),
        tilt_map=cuda.device_array((params.max_extrema, 2, 2), np.float32),
    )


@cuda.jit(cache=True, fastmath=True)
def oversample_bilinear_kernel(src, dst, delta_min):
    j_out, i_out = cuda.grid(2)

    ho, wo = dst.shape
    if j_out >= wo or i_out >= ho:
        return

    hi, wi = src.shape

    x = numba.float32(i_out) * delta_min
    y = numba.float32(j_out) * delta_min

    im = int(x)
    jm = int(y)
    ip = im + 1
    jp = jm + 1

    if ip >= hi:
        ip = 2 * hi - 1 - ip
    if im >= hi:
        im = 2 * hi - 1 - im
    if jp >= wi:
        jp = 2 * wi - 1 - jp
    if jm >= wi:
        jm = 2 * wi - 1 - jm

    fx = numba.float32(x - ld.floorf(x))
    fy = numba.float32(y - ld.floorf(y))
    one = numba.float32(1.0)

    im_jm = src[im, jm]
    im_jp = src[im, jp]
    ip_jm = src[ip, jm]
    ip_jp = src[ip, jp]

    dst[i_out, j_out] = fx * (fy * ip_jp + (one - fy) * ip_jm) + (one - fx) * (
        fy * im_jp + (one - fy) * im_jm
    )


@cuda.jit(device=True, inline=True, cache=True, fastmath=True)
def mirror(i: int, n: int) -> int:
    if i < 0:
        i = -i - 1
    elif i >= n:
        i = (n << 1) - 1 - i
    return i


@cuda.jit(cache=True, fastmath=True)
def gauss_h(src, dst, g, radius):
    tile = cuda.shared.array(shape=GAUSS_HORZ_TILE_SIZE, dtype=numba.float32)
    g_sh = cuda.shared.array(shape=GAUSS_COEFF_TILE_SIZE, dtype=numba.float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    h, w_in = src.shape
    bs = cuda.blockDim.x

    # Load gaussian coefficients into shared memory once per block
    for i in range(tx, radius + 1, bs):
        g_sh[i] = g[i]
    cuda.syncthreads()

    tile_w = bs + 2 * radius
    base_x = cuda.blockIdx.x * bs - radius

    for i in range(tx, tile_w, bs):
        lx = base_x + i
        tile[i] = src[y, mirror(lx, w_in)]
    cuda.syncthreads()

    if x < w_in and y < h:
        center = tile[tx + radius]
        acc = center * g_sh[0]

        for k in range(1, MAX_GAUSS_RADIUS + 1):
            if k <= radius:
                left = tile[tx + radius - k]
                right = tile[tx + radius + k]
                acc += g_sh[k] * (left + right)

        dst[y, x] = acc


@cuda.jit(cache=True, fastmath=True)
def gauss_v(src, dst, g, radius):
    v_tile = cuda.shared.array(shape=GAUSS_VERT_TILE_SIZE, dtype=numba.float32)
    g_sh = cuda.shared.array(shape=GAUSS_COEFF_TILE_SIZE, dtype=numba.float32)

    h_in, w_in = src.shape

    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    base_x = cuda.blockIdx.x * cuda.blockDim.x
    base_y = cuda.blockIdx.y * cuda.blockDim.y - radius

    for i in range(tx, radius + 1, cuda.blockDim.x):
        g_sh[i] = g[i]
    cuda.syncthreads()

    tile_h = cuda.blockDim.y + 2 * radius

    i = ty
    while i < tile_h:
        ly = base_y + i
        src_y = mirror(ly, h_in)
        src_x = base_x + tx
        v_tile[i * TX + tx] = src[src_y, mirror(src_x, w_in)]
        i += cuda.blockDim.y
    cuda.syncthreads()

    if x < w_in and y < h_in:
        center = v_tile[(ty + radius) * TX + tx]
        acc = center * g_sh[0]

        for k in range(1, MAX_GAUSS_RADIUS + 1):
            if k <= radius:
                up = v_tile[(ty + radius - k) * TX + tx]
                down = v_tile[(ty + radius + k) * TX + tx]
                acc += g_sh[k] * (up + down)

        dst[y, x] = acc


@cuda.jit(cache=True, fastmath=True)
def downsample_kernel(src, dst):
    x, y = cuda.grid(2)
    h, w = dst.shape
    if x < w and y < h:
        dst[y, x] = src[y * 2, x * 2]


@cuda.jit(cache=True, fastmath=True)
def reciprocal_inplace_kernel(arr, eps):
    y, x = cuda.grid(2)
    h, w = arr.shape
    if x < w and y < h:
        v = arr[y, x]
        arr[y, x] = numba.float32(1.0) / v if ld.fabsf(v) > eps else numba.float32(0.0)


@cuda.jit(cache=True, fastmath=True)
def dog_diff_kernel(gss_in, dog_out):
    s = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ns, h, w = dog_out.shape
    if s < ns and y < h and x < w:
        dog_out[s, y, x] = gss_in[s + 1, y, x] - gss_in[s, y, x]


@cuda.jit(cache=True, fastmath=True)
def find_and_record_extrema_kernel(
    dog_oct,
    o,
    int_buf,
    float_buf,
    counter,
    max_extrema,
    sigma_min,
    n_spo,
    delta_min,
):
    s, y, x = cuda.grid(3)
    ns, h, w = dog_oct.shape
    if s <= 0 or s >= ns - 1 or y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
        return
    v = dog_oct[s, y, x]
    is_max = True
    is_min = True
    for ds in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if ds == 0 and dy == 0 and dx == 0:
                    continue
                n = dog_oct[s + ds, y + dy, x + dx]
                if n >= v:
                    is_max = False
                if n <= v:
                    is_min = False
                if not is_max and not is_min:
                    return

    idx = cuda.atomic.add(counter, 0, 1)
    if idx >= max_extrema:
        cuda.atomic.add(counter, 1, 1)
        return
    int_buf[idx, 0] = o
    int_buf[idx, 1] = s
    int_buf[idx, 2] = y
    int_buf[idx, 3] = x
    scale = numba.float32(delta_min) * numba.float32(1 << o)
    float_buf[idx, 0] = numba.float32(y) * scale
    float_buf[idx, 1] = numba.float32(x) * scale
    exp_arg = numba.float32(o) + (numba.float32(s) / numba.float32(n_spo))
    float_buf[idx, 2] = numba.float32(sigma_min) * ld.exp2f(exp_arg)
    float_buf[idx, 3] = v


@cuda.jit(device=True, inline=True, cache=True, fastmath=True)
def invert_3x3(H, Hi):
    det = (
        H[0, 0] * (H[1, 1] * H[2, 2] - H[2, 1] * H[1, 2])
        - H[0, 1] * (H[1, 0] * H[2, 2] - H[1, 2] * H[2, 0])
        + H[0, 2] * (H[1, 0] * H[2, 1] - H[1, 1] * H[2, 0])
    )
    k = 1.0 / det
    Hi[0, 0] = (H[1, 1] * H[2, 2] - H[2, 1] * H[1, 2]) * k
    Hi[0, 1] = (H[0, 2] * H[2, 1] - H[0, 1] * H[2, 2]) * k
    Hi[0, 2] = (H[0, 1] * H[1, 2] - H[0, 2] * H[1, 1]) * k
    Hi[1, 0] = (H[1, 2] * H[2, 0] - H[1, 0] * H[2, 2]) * k
    Hi[1, 1] = (H[0, 0] * H[2, 2] - H[0, 2] * H[2, 0]) * k
    Hi[1, 2] = (H[1, 0] * H[0, 2] - H[0, 0] * H[1, 2]) * k
    Hi[2, 0] = (H[1, 0] * H[2, 1] - H[2, 0] * H[1, 1]) * k
    Hi[2, 1] = (H[2, 0] * H[0, 1] - H[0, 0] * H[2, 1]) * k
    Hi[2, 2] = (H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1]) * k
    return True


@cuda.jit(device=True, inline=True, cache=True, fastmath=True)
def mat_vec_mul_3x1(M, v, out):
    out[0] = M[0, 0] * v[0] + M[0, 1] * v[1] + M[0, 2] * v[2]
    out[1] = M[1, 0] * v[0] + M[1, 1] * v[1] + M[1, 2] * v[2]
    out[2] = M[2, 0] * v[0] + M[2, 1] * v[1] + M[2, 2] * v[2]


@cuda.jit(cache=True, fastmath=True)
def refine_kernel(
    dog_oct, int_buf, float_buf, ext_count, n_spo, sigma_min, delta_min, oct_idx
):
    idx = cuda.grid(1)
    if idx >= ext_count[0]:
        return
    o = int_buf[idx, 0]
    if o != oct_idx:
        return
    s, y, x = int_buf[idx, 1], int_buf[idx, 2], int_buf[idx, 3]
    ns, h, w = dog_oct.shape
    g = cuda.local.array(3, dtype=numba.float32)
    Hm = cuda.local.array((3, 3), dtype=numba.float32)
    Hin = cuda.local.array((3, 3), dtype=numba.float32)
    off = cuda.local.array(3, dtype=numba.float32)
    valid = False
    for _ in range(5):
        in_bounds = 1 <= s < ns - 1 and 1 <= y < h - 1 and 1 <= x < w - 1
        if in_bounds:
            g[0] = 0.5 * (dog_oct[s + 1, y, x] - dog_oct[s - 1, y, x])
            g[1] = 0.5 * (dog_oct[s, y + 1, x] - dog_oct[s, y - 1, x])
            g[2] = 0.5 * (dog_oct[s, y, x + 1] - dog_oct[s, y, x - 1])
            Hm[0, 0] = (
                dog_oct[s + 1, y, x] + dog_oct[s - 1, y, x] - 2 * dog_oct[s, y, x]
            )
            Hm[1, 1] = (
                dog_oct[s, y + 1, x] + dog_oct[s, y - 1, x] - 2 * dog_oct[s, y, x]
            )
            Hm[2, 2] = (
                dog_oct[s, y, x + 1] + dog_oct[s, y, x - 1] - 2 * dog_oct[s, y, x]
            )
            Hm[0, 1] = Hm[1, 0] = 0.25 * (
                dog_oct[s + 1, y + 1, x]
                - dog_oct[s + 1, y - 1, x]
                - dog_oct[s - 1, y + 1, x]
                + dog_oct[s - 1, y - 1, x]
            )
            Hm[0, 2] = Hm[2, 0] = 0.25 * (
                dog_oct[s + 1, y, x + 1]
                - dog_oct[s + 1, y, x - 1]
                - dog_oct[s - 1, y, x + 1]
                + dog_oct[s - 1, y, x - 1]
            )
            Hm[1, 2] = Hm[2, 1] = 0.25 * (
                dog_oct[s, y + 1, x + 1]
                - dog_oct[s, y + 1, x - 1]
                - dog_oct[s, y - 1, x + 1]
                + dog_oct[s, y - 1, x - 1]
            )
            invert_3x3(Hm, Hin)
            mat_vec_mul_3x1(Hin, g, off)
            off[0], off[1], off[2] = -off[0], -off[1], -off[2]
        else:
            off[0] = numba.float32(5.0)
            off[1] = numba.float32(5.0)
            off[2] = numba.float32(5.0)
        if (
            ld.fabsf(off[0]) < numba.float32(0.6)
            and ld.fabsf(off[1]) < numba.float32(0.6)
            and ld.fabsf(off[2]) < numba.float32(0.6)
        ):
            valid = True
            break

        if off[1] > numba.float32(0.6) and (y + 1) < (h - 1):
            y += 1
        if off[1] < numba.float32(-0.6) and (y - 1) > 0:
            y -= 1
        if off[2] > numba.float32(0.6) and (x + 1) < (w - 1):
            x += 1
        if off[2] < numba.float32(-0.6) and (x - 1) > 0:
            x -= 1
        if off[0] > numba.float32(0.6) and (s + 1) < (ns - 1):
            s += 1
        if off[0] < numba.float32(-0.6) and (s - 1) > 0:
            s -= 1

    if not valid:
        int_buf[idx, 0] = -1
        return

    if not (1 <= s < ns - 1 and 1 <= y < h - 1 and 1 <= x < w - 1):
        int_buf[idx, 0] = -1
        return
    D_hat = numba.float32(dog_oct[s, y, x]) + numba.float32(0.5) * (
        g[0] * off[0] + g[1] * off[1] + g[2] * off[2]
    )
    int_buf[idx, 1], int_buf[idx, 2], int_buf[idx, 3] = s, y, x
    scale = numba.float32(delta_min) * numba.float32(1 << o)
    float_buf[idx, 0] = (numba.float32(y) + off[1]) * scale
    float_buf[idx, 1] = (numba.float32(x) + off[2]) * scale
    exp_arg = numba.float32(o) + (numba.float32(s) + off[0]) / numba.float32(n_spo)
    float_buf[idx, 2] = numba.float32(sigma_min) * ld.exp2f(exp_arg)
    float_buf[idx, 3] = D_hat


@cuda.jit(cache=True, fastmath=True)
def discard_with_low_response_kernel(int_buf, float_buf, ext_count, thresh, oct_idx):
    idx = cuda.grid(1)
    if idx >= ext_count[0]:
        return
    o = int_buf[idx, 0]
    if o != oct_idx:
        return
    v = ld.fabsf(float_buf[idx, 3])
    eps = numba.float32(1e-6)
    eff = thresh - eps
    if v <= eff:
        int_buf[idx, 0] = -1


@cuda.jit(device=True, inline=True, fastmath=True)
def _safe_get_depth(D, y, x, h, w):
    # mirror pad
    if y < 0:
        y = -y - 1
    elif y >= h:
        y = (h << 1) - 1 - y
    if x < 0:
        x = -x - 1
    elif x >= w:
        x = (w << 1) - 1 - x
    return D[y, x]


@cuda.jit(cache=True, fastmath=True)
def prepare_depth_geom_kernel(
    depth_oct,  # (H, W) float32
    int_buf,
    float_buf,  # extrema buffers
    ext_count,  # int32[2], uses ext_count[0]
    oct_idx,  # int
    fx,
    fy,
    cx,
    cy,  # full-res intrinsics
    delta_min,  # scalar
    Jt_out,  # (N,2,2)
    min_aspect,  # float, e.g., 0.20 -> reject if s2/s1 < min_aspect
    max_depth,  # float, discard if Z > max_depth
):
    k = cuda.grid(1)
    if k >= ext_count[0] or int_buf[k, 0] != oct_idx:
        return

    # Fallback first (classic): identity warp
    Jt_out[k, 0, 0] = 1.0
    Jt_out[k, 0, 1] = 0.0
    Jt_out[k, 1, 0] = 0.0
    Jt_out[k, 1, 1] = 1.0

    # Try depth-aware overwrite
    scale = delta_min * (1 << oct_idx)
    y0 = float_buf[k, 0] / scale
    x0 = float_buf[k, 1] / scale

    H, W = depth_oct.shape
    yi = int(ld.floorf(y0 + 0.5))
    xi = int(ld.floorf(x0 + 0.5))
    if yi < 0 or yi >= H or xi < 0 or xi >= W:
        return

    Zc = depth_oct[yi, xi]
    if not ld.finitef(Zc) or Zc <= 0.0 or Zc > max_depth:
        return

    fx_o = fx / scale
    fy_o = fy / scale
    cx_o = cx / scale
    cy_o = cy / scale

    # depth gradients (mirror padded)
    Zxm = _safe_get_depth(depth_oct, yi, xi + 1, H, W) - _safe_get_depth(
        depth_oct, yi, xi - 1, H, W
    )
    Zym = _safe_get_depth(depth_oct, yi + 1, xi, H, W) - _safe_get_depth(
        depth_oct, yi - 1, xi, H, W
    )
    Zx = 0.5 * Zxm
    Zy = 0.5 * Zym

    # back-projection partials
    Xn = (x0 - cx_o) / fx_o
    Yn = (y0 - cy_o) / fy_o

    dPdx_x = Zc / fx_o + Xn * Zx
    dPdx_y = Yn * Zx
    dPdx_z = Zx
    dPdy_x = Xn * Zy
    dPdy_y = Zc / fy_o + Yn * Zy
    dPdy_z = Zy

    # orthonormal tangent basis
    nx = ld.sqrtf(dPdx_x * dPdx_x + dPdx_y * dPdx_y + dPdx_z * dPdx_z)
    if nx < 1e-8:
        return
    e1x = dPdx_x / nx
    e1y = dPdx_y / nx
    e1z = dPdx_z / nx
    dot12 = e1x * dPdy_x + e1y * dPdy_y + e1z * dPdy_z
    v2x = dPdy_x - dot12 * e1x
    v2y = dPdy_y - dot12 * e1y
    v2z = dPdy_z - dot12 * e1z
    ny = ld.sqrtf(v2x * v2x + v2y * v2y + v2z * v2z)
    if ny < 1e-8:
        return
    e2x = v2x / ny
    e2y = v2y / ny
    e2z = v2z / ny

    # surface->image mapping M; then J = M^{-1}
    m00 = e1x * dPdy_x + e1y * dPdy_y + e1z * dPdy_z
    m01 = e1x * dPdx_x + e1y * dPdx_y + e1z * dPdx_z
    m10 = e2x * dPdy_x + e2y * dPdy_y + e2z * dPdy_z
    m11 = e2x * dPdx_x + e2y * dPdx_y + e2z * dPdx_z

    detM = m00 * m11 - m01 * m10
    if ld.fabsf(detM) < 1e-8:
        return

    inv_det = 1.0 / detM
    j00 = inv_det * m11
    j01 = -inv_det * m01
    j10 = -inv_det * m10
    j11 = inv_det * m00

    # singular values of J (closed form)
    a2c2 = j00 * j00 + j10 * j10
    b2d2 = j01 * j01 + j11 * j11
    tr = a2c2 + b2d2
    detJ = j00 * j11 - j01 * j10
    disc = tr * tr - 4.0 * (detJ * detJ)
    if disc < 0.0:
        disc = 0.0
    root = ld.sqrtf(disc)
    s1 = ld.sqrtf(0.5 * (tr + root))  # max sv
    s2 = ld.sqrtf(0.5 * (tr - root))  # min sv
    if s1 <= 0.0:
        return
    if (s2 / s1) < min_aspect:
        return

    # major-axis normalization (tilt-only): J_n = J / s1  => Jt_n = J^T / s1
    Jt_out[k, 0, 0] = j00 / s1
    Jt_out[k, 0, 1] = j10 / s1
    Jt_out[k, 1, 0] = j01 / s1
    Jt_out[k, 1, 1] = j11 / s1


def upscale(src, dst, delta_min, stream):
    assert delta_min <= 1.0
    hi, wi = src.shape
    ho = int(round(hi / float(delta_min)))
    wo = int(round(wi / float(delta_min)))
    if dst.shape != (ho, wo):
        raise ValueError(
            f"dst.shape must be {(ho, wo)} for delta_min={float(delta_min)}, got {dst.shape}"
        )

    grid = ((wo + TX - 1) // TX, (ho + TY - 1) // TY)
    oversample_bilinear_kernel[grid, (TX, TY), stream](
        src, dst, numba.float32(delta_min)
    )


def gaussian_symm_kernel(sigma: float) -> tuple[DeviceNDArray, int]:
    radius = int(math.ceil(4.0 * float(sigma)))
    radius = min(radius, MAX_GAUSS_RADIUS)
    g = np.empty(radius + 1, dtype=np.float32)
    g[0] = np.float32(1.0)

    if sigma > 0.0:
        sig32 = np.float32(sigma)
        sum32 = np.float32(1.0)
        for i in range(1, radius + 1):
            t32 = np.float32(-0.5) * np.float32(i) * np.float32(i) / sig32 / sig32
            val32 = np.float32(math.exp(float(t32)))
            g[i] = val32
            sum32 = np.float32(sum32 + np.float32(2.0) * val32)
        g /= sum32
    else:
        if radius > 0:
            g[1:] = np.float32(0.0)

    return cuda.to_device(g), radius


def gaussian_blur(img_in, img_out, scratch, stream, gauss_kernel, radius):
    th = BLUR_TH
    if radius > MAX_GAUSS_RADIUS:
        raise ValueError(
            f"Gaussian radius {radius} exceeds MAX_GAUSS_RADIUS={MAX_GAUSS_RADIUS}."
        )
    # Vertical pass: 2D blocks for coalesced loads along x
    v_grid = (
        (img_in.shape[1] + TX - 1) // TX,
        (img_in.shape[0] + TY - 1) // TY,
    )
    gauss_v[v_grid, (TX, TY), stream](img_in, scratch, gauss_kernel, radius)
    h_grid = ((img_in.shape[1] + th - 1) // th, img_in.shape[0])
    gauss_h[h_grid, (th,), stream](scratch, img_out, gauss_kernel, radius)


def gradient(img_in, gx_out, gy_out, stream):
    h, w = img_in.shape
    grid = ((w + TX - 1) // TX, (h + TY - 1) // TY)
    gradient_kernel[grid, (TX, TY), stream](img_in, gx_out, gy_out)


def compute_gss(
    data: SiftData,
    params: SiftParams,
    octave_index: int,
    stream,
    record: bool = False,
):
    gss = data.gss[octave_index]
    gx = data.gx[octave_index]
    gy = data.gy[octave_index]
    scratch = data.scratch[octave_index]
    num_scales_total = params.n_spo + 3
    gradient(gss[0], gx[0], gy[0], stream)
    for scale_index in range(1, num_scales_total):
        sigma = params.inc_sigmas[octave_index, scale_index]
        gauss_kernel, radius = params.gauss_kernels[sigma]
        gaussian_blur(
            gss[scale_index - 1],
            gss[scale_index],
            scratch,
            stream,
            gauss_kernel,
            radius,
        )

        gradient(gss[scale_index], gx[scale_index], gy[scale_index], stream)
    if record:
        return (
            gss.copy_to_host(stream=stream),
            gx.copy_to_host(stream=stream),
            gy.copy_to_host(stream=stream),
        )
    return None, None, None


def compute_depth(
    data: SiftData,
    params: SiftParams,
    octave_index: int,
    stream,
    record: bool = False,
):
    if octave_index == 0:
        in_h, in_w = data.input_depth.shape
        out_h, out_w = data.depth[0].shape
        delta_h = float(in_h) / float(out_h)
        delta_w = float(in_w) / float(out_w)
        if not (abs(delta_h - delta_w) <= 1e-6):
            raise ValueError(
                f"Depth upsample ratios mismatch (h: {delta_h}, w: {delta_w});"
                f" cannot upscale input depth of shape {(in_h, in_w)} to {(out_h, out_w)}"
            )
        # Disparity-based upsampling for better linearity
        threads = (TX, TY)
        grid_in = ((in_w + TX - 1) // TX, (in_h + TY - 1) // TY)
        reciprocal_inplace_kernel[grid_in, threads, stream](
            data.input_depth, numba.float32(1e-12)
        )

        upscale(data.input_depth, data.depth[0], float(delta_h), stream)

        grid_out = ((out_w + TX - 1) // TX, (out_h + TY - 1) // TY)
        reciprocal_inplace_kernel[grid_out, threads, stream](
            data.depth[0], numba.float32(1e-12)
        )

        # Restore input buffer to original depth values
        reciprocal_inplace_kernel[grid_in, threads, stream](
            data.input_depth, numba.float32(1e-12)
        )
        if record:
            return data.depth[0].copy_to_host(stream=stream)
        return None

    src = data.depth[octave_index - 1]
    dst = data.depth[octave_index]
    height, width = params.gss_shapes[octave_index]
    grid = ((width + TX - 1) // TX, (height + TY - 1) // TY)
    downsample_kernel[grid, (TX, TY), stream](src, dst)

    if record:
        return dst.copy_to_host(stream=stream)
    return None


def compute_dog(
    data: SiftData, params: SiftParams, octave_index: int, stream, record: bool = False
):
    gss, dog = data.gss[octave_index], data.dog[octave_index]
    height, width = params.gss_shapes[octave_index]
    num_scales = params.n_spo + 2
    threads = (16, 16, 4)
    grid = (
        (width + threads[0] - 1) // threads[0],
        (height + threads[1] - 1) // threads[1],
        (num_scales + threads[2] - 1) // threads[2],
    )
    dog_diff_kernel[grid, threads, stream](gss, dog)
    if record:
        return dog.copy_to_host(stream=stream)
    return None


def detect_extrema(
    data: SiftData, params: SiftParams, octave_index: int, stream, record: bool = False
):
    dog_octave = data.dog[octave_index]
    height, width = params.gss_shapes[octave_index]
    threads = (2, 8, 8)
    blocks = (
        (params.n_spo + 2 + threads[0] - 1) // threads[0],
        (height + threads[1] - 1) // threads[1],
        (width + threads[2] - 1) // threads[2],
    )
    find_and_record_extrema_kernel[blocks, threads, stream](
        dog_octave,
        octave_index,
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        params.max_extrema,
        params.sigma_min,
        params.n_spo,
        params.delta_min,
    )
    if record:
        n = int(data.extrema.counter.copy_to_host(stream=stream)[0])
        if n > 0:
            ib = data.extrema.int_buffer[:n].copy_to_host(stream=stream)
            fb = data.extrema.float_buffer[:n].copy_to_host(stream=stream)
            mask = ib[:, 0] == octave_index
            if mask.any():
                return (ib[mask], fb[mask])
        return None


def refine_extrema(
    data: SiftData, params: SiftParams, octave_index: int, stream, record: bool = False
):
    threads = 128
    blocks = (params.max_extrema + threads - 1) // threads
    refine_kernel[blocks, threads, stream](
        data.dog[octave_index],
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        params.n_spo,
        params.sigma_min,
        params.delta_min,
        octave_index,
    )
    if record:
        n = int(data.extrema.counter.copy_to_host(stream=stream)[0])
        if n > 0:
            ib = data.extrema.int_buffer[:n].copy_to_host(stream=stream)
            fb = data.extrema.float_buffer[:n].copy_to_host(stream=stream)
            mask = ib[:, 0] == octave_index
            return (ib[mask], fb[mask])
        return None


def discard_with_low_response(
    data: SiftData,
    params: SiftParams,
    multiplier: float,
    octave_index: int,
    stream,
    record: bool = False,
):
    thresh = numba.float32(float(params.C_dog) * float(multiplier))
    threads = 128
    blocks = (data.extrema.int_buffer.shape[0] + threads - 1) // threads
    discard_with_low_response_kernel[blocks, threads, stream](
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        thresh,
        octave_index,
    )
    if record:
        total = int(data.extrema.counter.copy_to_host(stream=stream)[0])
        ib_h = data.extrema.int_buffer[:total].copy_to_host(stream=stream)
        fb_h = data.extrema.float_buffer[:total].copy_to_host(stream=stream)
        mask = ib_h[:, 0] == octave_index
        return (ib_h[mask], fb_h[mask])
    return None


@cuda.jit(cache=True, fastmath=True)
def discard_on_edge_kernel(dog_oct, int_buf, ext_count, C_edge, oct_idx):
    idx = cuda.grid(1)
    if idx >= ext_count[0]:
        return
    o = int_buf[idx, 0]
    if o != oct_idx:
        return
    s = int_buf[idx, 1]
    i = int_buf[idx, 2]
    j = int_buf[idx, 3]
    ns, h, w = dog_oct.shape
    if not (1 <= s < ns - 1 and 1 <= i < h - 1 and 1 <= j < w - 1):
        int_buf[idx, 0] = -1
        return
    im = dog_oct[s]
    hXX = im[i - 1, j] + im[i + 1, j] - 2 * im[i, j]
    hYY = im[i, j + 1] + im[i, j - 1] - 2 * im[i, j]
    hXY = numba.float32(0.25) * (
        (im[i + 1, j + 1] - im[i + 1, j - 1]) - (im[i - 1, j + 1] - im[i - 1, j - 1])
    )
    det = hXX * hYY - hXY * hXY
    if det <= 0:
        int_buf[idx, 0] = -1
        return
    trace = hXX + hYY
    r = C_edge
    if (trace * trace) / det > ((r + 1.0) * (r + 1.0) / r):
        int_buf[idx, 0] = -1
        return


@cuda.jit(device=True, inline=True, cache=True, fastmath=True)
def wrap_angle(theta: numba.float32) -> numba.float32:
    return ld.fmodf(ld.fmodf(theta, TWO_PI) + TWO_PI, TWO_PI)


@cuda.jit(cache=True)
def gradient_kernel(img, gx_out, gy_out):
    tile = cuda.shared.array(shape=GRAD_TILE_SIZE, dtype=numba.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y

    x = bx * bdx + tx
    y = by * bdy + ty

    h, w = img.shape

    base_x = bx * bdx
    base_y = by * bdy

    tile_w = TX + 2
    tile_h = TY + 2
    for ly in range(ty, tile_h, bdy):
        gy = base_y + ly - 1
        if gy < 0:
            gy = 0
        elif gy > h - 1:
            gy = h - 1
        for lx in range(tx, tile_w, bdx):
            gx = base_x + lx - 1
            if gx < 0:
                gx = 0
            elif gx > w - 1:
                gx = w - 1
            tile[ly * tile_w + lx] = img[gy, gx]

    cuda.syncthreads()

    if x >= w or y >= h:
        return

    ltx = tx + 1
    lty = ty + 1

    fx = numba.float32(0.5 if (0 < x < w - 1) else 1.0)
    fy = numba.float32(0.5 if (0 < y < h - 1) else 1.0)

    base = lty * tile_w + ltx
    right = tile[base + 1]
    left = tile[base - 1]
    down = tile[base + tile_w]
    up = tile[base - tile_w]

    gx = fx * (right - left)
    gy = fy * (down - up)
    gx_out[y, x] = gx
    gy_out[y, x] = gy


@cuda.jit(cache=True, fastmath=True)
def orientation_kernel(
    gx,
    gy,
    int_buf,
    float_buf,
    n_extrema,
    key_float,
    key_int,
    kp_counter,
    oct_idx,
    lambda_ori,
    delta_min,
    Jt_all,
    tilt_map,
):
    kp_idx = cuda.blockIdx.x
    if kp_idx >= n_extrema[0] or int_buf[kp_idx, 0] != oct_idx:
        return

    s = int_buf[kp_idx, 1]
    scale = delta_min * (1 << oct_idx)
    y0 = float_buf[kp_idx, 0] / scale
    x0 = float_buf[kp_idx, 1] / scale
    sigmaw = float_buf[kp_idx, 2]
    sigmao = sigmaw / scale

    R = 3.0 * lambda_ori * sigmao
    radius = int(R + 0.5)
    radius = 1 if radius == 0 else radius
    h, w = gx.shape[1:]
    gsigma = lambda_ori * sigmao
    inv2s2 = 1.0 / (2.0 * gsigma * gsigma)
    binscl = numba.float32(ORI_BINS / TWO_PI)

    # Load J^T and form J and inv(J) (assume upstream ensured det != 0)
    jt00 = Jt_all[kp_idx, 0, 0]
    jt01 = Jt_all[kp_idx, 0, 1]
    jt10 = Jt_all[kp_idx, 1, 0]
    jt11 = Jt_all[kp_idx, 1, 1]
    j00 = jt00
    j01 = jt10
    j10 = jt01
    j11 = jt11
    det = j00 * j11 - j01 * j10
    den = det + ld.copysignf(numba.float32(1e-12), det)
    invd = 1.0 / den
    inv00 = invd * j11
    inv01 = -invd * j01
    inv10 = -invd * j10
    inv11 = invd * j00

    # histogram
    hist = cuda.shared.array(ORI_BINS, numba.float32)
    tflat = cuda.threadIdx.y * TX + cuda.threadIdx.x
    if tflat < ORI_BINS:
        hist[tflat] = 0.0
    cuda.syncthreads()

    # classical scan box (no circle gating)
    siMin = 0 if (y0 - R + 0.5) < 0.0 else int(y0 - R + 0.5)
    sjMin = 0 if (x0 - R + 0.5) < 0.0 else int(x0 - R + 0.5)
    siMax = h - 1 if (y0 + R + 0.5) > (h - 1) else int(y0 + R + 0.5)
    sjMax = w - 1 if (x0 + R + 0.5) > (w - 1) else int(x0 + R + 0.5)
    Hbox = siMax - siMin + 1
    Wbox = sjMax - sjMin + 1

    for dy in range(cuda.threadIdx.y, Hbox, TY):
        yy = siMin + dy
        dyf = numba.float32(yy) - numba.float32(y0)
        for dx in range(cuda.threadIdx.x, Wbox, TX):
            xx = sjMin + dx
            dxf = numba.float32(xx) - numba.float32(x0)

            # image -> surface coords
            u = inv00 * dyf + inv01 * dxf
            v = inv10 * dyf + inv11 * dxf

            # image gradients
            gyv = gy[s, yy, xx]
            gxv = gx[s, yy, xx]

            # rotate gradient to (u,v): g_uv = J^T g_yx  (y-first layout)
            gu = jt00 * gyv + jt01 * gxv
            gv = jt10 * gyv + jt11 * gxv

            m = ld.sqrtf(gu * gu + gv * gv)
            if m == 0.0:
                continue

            # angle in surface coords (NOTE: when Jt==I -> atan2(gx,gy), i.e. classical)
            a = wrap_angle(ld.atan2f(gv, gu))

            # Gaussian weight in surface coords (NOTE: when Jt==I -> dx,dy)
            wgt = m * ld.expf(-(u * u + v * v) * inv2s2)

            bin_f = a * binscl + numba.float32(0.5)
            bin_i = int(ld.floorf(bin_f)) % ORI_BINS
            if bin_i < 0:
                bin_i += ORI_BINS
            cuda.atomic.add(hist, bin_i, wgt)

    cuda.syncthreads()

    if tflat == 0:
        tmp = cuda.local.array(ORI_BINS, numba.float32)
        for _ in range(6):
            for i in range(ORI_BINS):
                tmp[i] = hist[i]
            for i in range(ORI_BINS):
                hist[i] = (
                    tmp[(i - 1) % ORI_BINS] + tmp[i] + tmp[(i + 1) % ORI_BINS]
                ) / 3.0

        vmax = numba.float32(0.0)
        for i in range(ORI_BINS):
            if hist[i] > vmax:
                vmax = hist[i]
        if vmax == 0.0:
            return

        thr = ORI_THRESHOLD * vmax
        for i in range(ORI_BINS):
            p = hist[(i - 1) % ORI_BINS]
            c = hist[i]
            n = hist[(i + 1) % ORI_BINS]
            if not (c > thr and c > p and c > n):
                continue
            denom = p - 2.0 * c + n
            off = (p - n) / (2.0 * denom)
            theta = wrap_angle((i + off + 0.5) * (TWO_PI / ORI_BINS))

            out = cuda.atomic.add(kp_counter, 0, 1)
            if out >= key_float.shape[0]:
                cuda.atomic.add(kp_counter, 0, -1)
                cuda.atomic.add(kp_counter, 2, 1)
                return

            key_float[out, 0] = float_buf[kp_idx, 0]
            key_float[out, 1] = float_buf[kp_idx, 1]
            key_float[out, 2] = sigmaw
            key_float[out, 3] = theta
            key_int[out, 0] = oct_idx
            key_int[out, 1] = s
            key_int[out, 2] = int_buf[kp_idx, 2]
            key_int[out, 3] = int_buf[kp_idx, 3]
            tilt_map[out, 0, 0] = jt00
            tilt_map[out, 0, 1] = jt01
            tilt_map[out, 1, 0] = jt10
            tilt_map[out, 1, 1] = jt11


def discard_on_edge(
    data: SiftData, params: SiftParams, octave_index: int, stream, record: bool = False
):
    threads = 128
    blocks = (data.extrema.int_buffer.shape[0] + threads - 1) // threads
    discard_on_edge_kernel[blocks, threads, stream](
        data.dog[octave_index],
        data.extrema.int_buffer,
        data.extrema.counter,
        numba.float32(params.C_edge),
        octave_index,
    )
    if record:
        total = int(data.extrema.counter.copy_to_host(stream=stream)[0])
        ib_h = data.extrema.int_buffer[:total].copy_to_host(stream=stream)
        fb_h = data.extrema.float_buffer[:total].copy_to_host(stream=stream)
        mask = ib_h[:, 0] == octave_index
        return (ib_h[mask], fb_h[mask])
    return None


def discard_near_the_border(
    data: SiftData, params: SiftParams, octave_index: int, stream, record: bool = False
):
    image_height, image_width = params.img_dims
    threads = 128
    blocks = (data.extrema.int_buffer.shape[0] + threads - 1) // threads
    discard_near_the_border_kernel[blocks, threads, stream](
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        int(octave_index),
        int(image_height),
        int(image_width),
    )
    if record:
        total = int(data.extrema.counter.copy_to_host()[0])
        ib_h = data.extrema.int_buffer[:total].copy_to_host()
        fb_h = data.extrema.float_buffer[:total].copy_to_host()
        mask = ib_h[:, 0] == octave_index
        return (ib_h[mask], fb_h[mask])
    return None


@cuda.jit(cache=True, fastmath=True)
def discard_near_the_border_kernel(
    int_buf, float_buf, ext_count, oct_idx, image_h, image_w
):
    idx = cuda.grid(1)
    if idx >= ext_count[0]:
        return
    o = int_buf[idx, 0]
    if o != oct_idx:
        return
    y = float_buf[idx, 0]
    x = float_buf[idx, 1]
    sigma = float_buf[idx, 2]
    if not (
        (y - sigma > numba.float32(0.0))
        and (y + sigma < numba.float32(image_h))
        and (x - sigma > numba.float32(0.0))
        and (x + sigma < numba.float32(image_w))
    ):
        int_buf[idx, 0] = -1


@cuda.jit(cache=True, fastmath=True)
def descriptor_kernel(
    gx,
    gy,
    key_float,
    key_int,
    kctr,
    desc,
    oct_idx,
    delta_min,
    Jt_kp,  # NEW: per-keypoint J^T (shape: [max_kp, 2, 2])
):
    kp_idx = cuda.blockIdx.x
    if kp_idx < kctr[1] or kp_idx >= kctr[0] or key_int[kp_idx, 0] != oct_idx:
        return

    s = key_int[kp_idx, 1]
    yw = key_float[kp_idx, 0]
    xw = key_float[kp_idx, 1]
    sigma = key_float[kp_idx, 2]  # full-res sigma (classic)
    theta0 = key_float[kp_idx, 3]  # dominant orientation IN SURFACE (from ori kernel)

    scale = delta_min * (1 << oct_idx)
    x0, y0 = xw / scale, yw / scale

    radiusF = LAMBDA_DESC * sigma / scale
    inv_2sig2 = 1.0 / (2.0 * radiusF * radiusF)
    bin_scale = NORIBIN / TWO_PI
    half_bins = (NHIST - 1.0) * 0.5
    inv_cell = NHIST / (2.0 * radiusF)

    R = (1.0 + 1.0 / NHIST) * radiusF
    Rp = ld.sqrtf(2.0) * R

    h, w = gx.shape[1:]
    siMin = 0 if (y0 - Rp + 0.5) < 0.0 else int(y0 - Rp + 0.5)
    sjMin = 0 if (x0 - Rp + 0.5) < 0.0 else int(x0 - Rp + 0.5)
    siMax_f = y0 + Rp + 0.5
    sjMax_f = x0 + Rp + 0.5
    siMax = h - 1 if siMax_f > (h - 1) else int(siMax_f)
    sjMax = w - 1 if sjMax_f > (w - 1) else int(sjMax_f)
    height = max(0, siMax - siMin)
    width = max(0, sjMax - sjMin)

    # --- NEW: load J^T for this keypoint and build inv(J) once per block
    jt00 = Jt_kp[kp_idx, 0, 0]
    jt01 = Jt_kp[kp_idx, 0, 1]
    jt10 = Jt_kp[kp_idx, 1, 0]
    jt11 = Jt_kp[kp_idx, 1, 1]
    # J = (Jt)^T
    j00 = jt00
    j01 = jt10
    j10 = jt01
    j11 = jt11
    detJ = j00 * j11 - j01 * j10
    inv00 = numba.float32(1.0)
    inv01 = numba.float32(0.0)
    inv10 = numba.float32(0.0)
    inv11 = numba.float32(1.0)
    if ld.fabsf(detJ) >= 1e-8:  # if singular, identity fallback
        inv_det = 1.0 / detJ
        inv00 = inv_det * j11
        inv01 = -inv_det * j01
        inv10 = -inv_det * j10
        inv11 = inv_det * j00

    # rotation terms (same as classic, but applied in SURFACE coords)
    c = ld.cosf(theta0)
    sn = ld.sinf(theta0)

    hist = cuda.shared.array(DESC_LEN, numba.float32)
    tflat = cuda.threadIdx.y * TX + cuda.threadIdx.x
    if tflat < DESC_LEN:
        hist[tflat] = 0.0
    cuda.syncthreads()

    for py in range(cuda.threadIdx.y, height, TY):
        yy = siMin + py
        dy0 = numba.float32(yy) - numba.float32(y0)
        for px in range(cuda.threadIdx.x, width, TX):
            xx = sjMin + px
            dx0 = numba.float32(xx) - numba.float32(x0)

            # --- NEW: image offset -> SURFACE coords (u,v) using inv(J)
            u = inv00 * dy0 + inv01 * dx0
            v = inv10 * dy0 + inv11 * dx0

            # --- CHANGED: rotate SURFACE coords by theta0 (same formulas as classic)
            # classic does:
            #   dx =  dy0*c + dx0*sn
            #   dy = -dy0*sn + dx0*c
            # here we replace (dy0,dx0) with (u,v)
            dx = u * c + v * sn
            dy = -u * sn + v * c

            if not (ld.fabsf(dx) < R and ld.fabsf(dy) < R):
                continue

            # gradients in IMAGE, then --- NEW: transform to SURFACE via J^T
            gyv = gy[s, yy, xx]
            gxv = gx[s, yy, xx]
            gu = jt00 * gyv + jt01 * gxv  # NEW
            gv = jt10 * gyv + jt11 * gxv  # NEW

            m = ld.sqrtf(gu * gu + gv * gv)  # magnitude in SURFACE  (NEW)
            if m == 0.0:
                continue

            # angle in SURFACE relative to theta0 (NEW)
            a = wrap_angle(ld.atan2f(gv, gu) - theta0)
            ob = a * bin_scale

            # same trilinear vote as classic, but weight uses SURFACE distance
            u0 = int(ld.floorf(dy * inv_cell + half_bins))
            du = (dy * inv_cell + half_bins) - u0
            v0 = int(ld.floorf(dx * inv_cell + half_bins))
            dv = (dx * inv_cell + half_bins) - v0
            o0 = int(ob)
            do = ob - o0

            wbase = m * ld.expf(-(dx * dx + dy * dy) * inv_2sig2)

            for iu in (0, 1):
                uu = u0 + iu
                if 0 <= uu < NHIST:
                    wu = (1 - du) if iu == 0 else du
                    for iv in (0, 1):
                        vv = v0 + iv
                        if 0 <= vv < NHIST:
                            wv = (1 - dv) if iv == 0 else dv
                            for io in (0, 1):
                                oo = (o0 + io) & (NORIBIN - 1)
                                wo = (1 - do) if io == 0 else do
                                hidx = ((uu * NHIST + vv) * NORIBIN) + oo
                                cuda.atomic.add(hist, hidx, wbase * wu * wv * wo)

    cuda.syncthreads()

    if tflat == 0:
        # normalization + clipping (unchanged)
        l2 = numba.float32(0.0)
        for i in range(DESC_LEN):
            l2 += hist[i] * hist[i]
        inv = 1.0 / (ld.sqrtf(l2) + 1e-12)

        l2p = numba.float32(0.0)
        for i in range(DESC_LEN):
            v = hist[i] * inv
            v = 0.2 if v > 0.2 else v
            hist[i] = v
            l2p += v * v
        inv2 = 1.0 / (ld.sqrtf(l2p) + 1e-12)

        for i in range(DESC_LEN):
            q = hist[i] * inv2 * 512.0
            desc[kp_idx, i] = numba.uint8(255 if q > 255 else int(q))


@cuda.jit(cache=True, fastmath=True)
def set_kp_start_from_count(kctr):
    if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0:
        kctr[1] = kctr[0]


def build_descriptors(
    data: SiftData, params: SiftParams, octave_index: int, stream, record: bool = False
):
    set_kp_start_from_count[1, 1, stream](data.keypoints.counter)

    orientation_kernel[(params.max_extrema,), (TX, TY), stream](
        data.gx[octave_index],
        data.gy[octave_index],
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        data.keypoints.float_buffer,
        data.keypoints.int_buffer,
        data.keypoints.counter,
        octave_index,
        params.lambda_ori,
        params.delta_min,
        data.tilt_map,
        data.keypoints.tilt_map,
    )

    descriptor_kernel[(params.max_keypoints,), (TX, TY), stream](
        data.gx[octave_index],
        data.gy[octave_index],
        data.keypoints.float_buffer,
        data.keypoints.int_buffer,
        data.keypoints.counter,
        data.keypoints.descriptors,
        octave_index,
        params.delta_min,
        data.keypoints.tilt_map,
    )

    if record:
        total = int(data.keypoints.counter.copy_to_host(stream=stream)[0])
        if total > 0:
            ib = data.keypoints.int_buffer[:total].copy_to_host(stream=stream)
            fb = data.keypoints.float_buffer[:total].copy_to_host(stream=stream)
            desc = data.keypoints.descriptors[:total].copy_to_host(stream=stream)
            mask = ib[:, 0] == octave_index
            ints = ib[mask]
            flts = fb[mask]
            desc_np = desc[mask]
            if ints.shape[0] > 0:
                return (ints, flts, desc_np)
        return None


@cuda.jit(cache=True, fastmath=True)
def set_identity_tilt_kernel(int_buf, ext_count, oct_idx, Jt_out):
    k = cuda.grid(1)
    if k >= ext_count[0] or int_buf[k, 0] != oct_idx:
        return
    Jt_out[k, 0, 0] = 1.0
    Jt_out[k, 0, 1] = 0.0
    Jt_out[k, 1, 0] = 0.0
    Jt_out[k, 1, 1] = 1.0


def prepare_depth_geoms(
    data: SiftData,
    params: SiftParams,
    octave_index: int,
    stream,
    K,
    use_depth: bool = True,  # <— NEW
    record: bool = False,
):
    threads = 128
    blocks = (params.max_extrema + threads - 1) // threads

    if not use_depth:
        # Classical mode: fill per-extremum J^T with identity
        set_identity_tilt_kernel[blocks, threads, stream](
            data.extrema.int_buffer,
            data.extrema.counter,
            int(octave_index),
            data.tilt_map,
        )
    else:
        # Depth-aware mode: compute J^T from depth
        prepare_depth_geom_kernel[blocks, threads, stream](
            data.depth[octave_index],
            data.extrema.int_buffer,
            data.extrema.float_buffer,
            data.extrema.counter,
            int(octave_index),
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
            numba.float32(params.delta_min),
            data.tilt_map,
            params.min_aspect,
            params.max_depth,
        )

    if record:
        total = int(data.extrema.counter.copy_to_host(stream=stream)[0])
        if total > 0:
            ib = data.extrema.int_buffer[:total].copy_to_host(stream=stream)
            mask = ib[:, 0] == octave_index
            if mask.any():
                return data.tilt_map[:total].copy_to_host(stream=stream)[mask]
        return None


def compute_octave(
    data: SiftData,
    params: SiftParams,
    octave_index: int,
    K,
    stream,
    use_depth: bool,
    record: bool = False,
) -> Optional[Dict[str, object]]:
    snapshot: dict[str, object] = {}

    data.extrema.counter.copy_to_device(np.array([0, 0], dtype=np.int32), stream)

    if octave_index == 0:
        set_seed(data, params, stream)
    else:
        set_first_scale(data, params, octave_index, stream)

    snapshot["gss"], snapshot["grad_x"], snapshot["grad_y"] = compute_gss(
        data, params, octave_index, stream, record
    )
    snapshot["depth"] = (
        compute_depth(data, params, octave_index, stream, record) if use_depth else None
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
    snapshot["tilt_map"] = prepare_depth_geoms(
        data, params, octave_index, stream, K, use_depth, record
    )
    snapshot["keys"] = build_descriptors(data, params, octave_index, stream, record)

    return snapshot


def set_seed(data: SiftData, params: SiftParams, stream):
    assert params.sigma_min >= params.sigma_in
    upscale(data.input_img, data.seed_img, params.delta_min, stream)
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


def set_first_scale(data: SiftData, params: SiftParams, octave_index: int, stream):
    src, dst = data.gss[octave_index - 1][params.n_spo], data.gss[octave_index][0]
    height, width = params.gss_shapes[octave_index]
    grid = ((width + TX - 1) // TX, (height + TY - 1) // TY)
    downsample_kernel[grid, (TX, TY), stream](src, dst)


def compute(
    data: SiftData,
    params: SiftParams,
    stream,
    img: np.ndarray,
    depth: Optional[np.ndarray],  # <— NEW
    K: np.ndarray,
    record: bool = False,
) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []

    data.input_img.copy_to_device(img.astype(np.float32), stream)

    use_depth = depth is not None
    if use_depth:
        data.input_depth.copy_to_device(depth.astype(np.float32), stream)

    data.extrema.counter.copy_to_device(np.array([0, 0], dtype=np.int32), stream)
    data.keypoints.counter.copy_to_device(np.array([0, 0, 0], dtype=np.int32), stream)

    for o in range(params.n_oct):
        snapshot = compute_octave(data, params, o, K, stream, use_depth, record)
        snapshots.append(snapshot)

    data.keypoints.int_buffer.copy_to_host(data.keypoints_host.int_buffer, stream)
    data.keypoints.float_buffer.copy_to_host(data.keypoints_host.float_buffer, stream)
    data.keypoints.descriptors.copy_to_host(data.keypoints_host.descriptors, stream)
    data.keypoints.tilt_map.copy_to_host(data.keypoints_host.tilt_map, stream)
    data.keypoints.counter.copy_to_host(data.keypoints_host.counter, stream)

    return snapshots


class Sift:
    def __init__(self, params: SiftParams):
        self.params = params
        self.data = create_sift_data(params)
        self._cp_stream = cp.cuda.Stream(non_blocking=True)
        self._stream = cuda.external_stream(self._cp_stream.ptr)

    def compute(
        self,
        img: np.ndarray,
        depth: Optional[np.ndarray],  # <— NEW
        K: Optional[np.ndarray],
        record: bool = False,
    ) -> tuple[KeypointsHost, list[dict[str, object]]]:
        assert img.shape == self.params.img_dims, (
            f"got {img.shape}, expected {self.params.img_dims}"
        )
        if depth is not None:
            assert depth.shape == self.params.depth_dims, (
                f"got depth {depth.shape}, expected {self.params.depth_dims}"
            )

        K = np.eye(3) if K is None else K

        snapshot = compute(
            self.data, self.params, self._stream, img, depth, K, record=record
        )
        return self.data.keypoints_host.copy(), snapshot


def draw_random_affine_keypoints(
    img: np.ndarray,
    key_int: np.ndarray,
    key_float: np.ndarray,
    tilt_map: np.ndarray,
    *,
    n_samples: int = 200,
    magnify: float = 3.0,
    delta_min: float = 0.5,
    n_pts: int = 64,
    color=(0, 0, 255),
    outline=(255, 255, 255),
    thick: int = 2,
    border: int = 1,
    seed: int | None = 0,
):
    import cv2
    import numpy as np

    assert key_int.ndim == 2 and key_int.shape[1] == 4
    assert key_float.ndim == 2 and key_float.shape[1] == 4
    assert tilt_map.ndim == 3 and tilt_map.shape[1:] == (2, 2)

    N = min(len(key_int), len(key_float), len(tilt_map))
    if N == 0:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    C = key_float[:N, :2]
    Jt = tilt_map[:N]
    valid = np.isfinite(C).all(axis=1) & np.isfinite(Jt.reshape(N, -1)).all(axis=1)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    rng = np.random.default_rng(seed)
    sel = rng.choice(idx, size=min(n_samples, idx.size), replace=False)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

    t = np.linspace(0.0, 2.0 * np.pi, int(max(8, n_pts)), endpoint=True).astype(
        np.float32
    )
    uv = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)
    t_outline = thick + 2 * border

    for i in sel:
        o = int(key_int[i, 0])
        cy, cx = map(float, C[i])
        ang = float(key_float[i, 3])
        s = float(delta_min) * float(1 << o) * float(magnify)
        J = Jt[i].T.astype(np.float32, copy=False)

        poly = (uv @ J.T) * s
        poly = np.stack([cx + poly[:, 1], cy + poly[:, 0]], axis=1)
        poly_i = np.round(poly).astype(np.int32).reshape(-1, 1, 2)

        udir = np.array([np.cos(ang), np.sin(ang)], np.float32)
        d = (J @ udir) * s
        ctr = (int(round(cx)), int(round(cy)))
        tip = (int(round(cx + float(d[1]))), int(round(cy + float(d[0]))))

        cv2.polylines(out, [poly_i], True, outline, t_outline, lineType=cv2.LINE_AA)
        cv2.line(out, ctr, tip, outline, t_outline, lineType=cv2.LINE_AA)
        cv2.circle(out, ctr, max(t_outline // 2, 1), outline, -1, lineType=cv2.LINE_AA)

        cv2.polylines(out, [poly_i], True, color, thick, lineType=cv2.LINE_AA)
        cv2.line(out, ctr, tip, color, thick, lineType=cv2.LINE_AA)
        cv2.circle(out, ctr, max(thick // 2, 1), color, -1, lineType=cv2.LINE_AA)

    return out


def match_symmetric_ratio_and_essential(
    kp1: KeypointsHost,
    kp2: KeypointsHost,
    K1: np.ndarray,
    K2: np.ndarray,
    ratio: float = 0.75,
    method: int = cv2.RANSAC,
    prob: float = 0.999,
    threshold: float = 1.0,
):
    """Match SIFT features with symmetric Lowe's ratio, then filter with E.

    Returns a dict containing:
      - pairs_idx: (M,2) int indices into kp1/kp2
      - pts1: (M,2) float32 pixel coords (x,y) in image1
      - pts2: (M,2) float32 pixel coords (x,y) in image2
      - E: (3,3) essential matrix or None
      - inlier_mask: (M,) uint8 mask from RANSAC (all ones here since already filtered)
    """
    n1 = int(kp1.counter[0]) if kp1.counter.size > 0 else len(kp1.descriptors)
    n2 = int(kp2.counter[0]) if kp2.counter.size > 0 else len(kp2.descriptors)
    if n1 == 0 or n2 == 0:
        return {
            "pairs_idx": np.empty((0, 2), dtype=np.int32),
            "pts1": np.empty((0, 2), dtype=np.float32),
            "pts2": np.empty((0, 2), dtype=np.float32),
            "E": None,
            "inlier_mask": np.empty((0,), dtype=np.uint8),
        }

    desc1 = kp1.descriptors[:n1].astype(np.float32, copy=False)
    desc2 = kp2.descriptors[:n2].astype(np.float32, copy=False)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Forward ratio
    fwd = {}
    for knn in bf.knnMatch(desc1, desc2, k=2):
        if len(knn) < 2:
            continue
        m, n = knn[0], knn[1]
        if m.distance < ratio * n.distance:
            fwd[m.queryIdx] = m.trainIdx

    # Backward ratio
    bwd = {}
    for knn in bf.knnMatch(desc2, desc1, k=2):
        if len(knn) < 2:
            continue
        m, n = knn[0], knn[1]
        if m.distance < ratio * n.distance:
            bwd[m.queryIdx] = m.trainIdx

    # Symmetric keep
    pairs = [(i, j) for i, j in fwd.items() if bwd.get(j, -1) == i]
    if len(pairs) == 0:
        return {
            "pairs_idx": np.empty((0, 2), dtype=np.int32),
            "pts1": np.empty((0, 2), dtype=np.float32),
            "pts2": np.empty((0, 2), dtype=np.float32),
            "E": None,
            "inlier_mask": np.empty((0,), dtype=np.uint8),
        }

    pairs_idx = np.asarray(pairs, dtype=np.int32)

    # Build pixel coords (OpenCV expects x,y)
    xy1 = np.stack(
        [kp1.float_buffer[pairs_idx[:, 0], 1], kp1.float_buffer[pairs_idx[:, 0], 0]],
        axis=1,
    ).astype(np.float32)
    xy2 = np.stack(
        [kp2.float_buffer[pairs_idx[:, 1], 1], kp2.float_buffer[pairs_idx[:, 1], 0]],
        axis=1,
    ).astype(np.float32)

    # Normalize by intrinsics to use the focal/pp variant robustly
    fx1, fy1 = float(K1[0, 0]), float(K1[1, 1])
    cx1, cy1 = float(K1[0, 2]), float(K1[1, 2])
    fx2, fy2 = float(K2[0, 0]), float(K2[1, 1])
    cx2, cy2 = float(K2[0, 2]), float(K2[1, 2])

    norm1 = np.column_stack(((xy1[:, 0] - cx1) / fx1, (xy1[:, 1] - cy1) / fy1)).astype(
        np.float32
    )
    norm2 = np.column_stack(((xy2[:, 0] - cx2) / fx2, (xy2[:, 1] - cy2) / fy2)).astype(
        np.float32
    )

    # Estimate E with RANSAC on normalized coords
    E, mask = cv2.findEssentialMat(
        norm1,
        norm2,
        1.0,
        (0.0, 0.0),
        method=method,
        prob=prob,
        threshold=threshold,
    )

    if E is None or mask is None:
        return {
            "pairs_idx": np.empty((0, 2), dtype=np.int32),
            "pts1": np.empty((0, 2), dtype=np.float32),
            "pts2": np.empty((0, 2), dtype=np.float32),
            "E": E,
            "inlier_mask": np.empty((0,), dtype=np.uint8),
        }

    inl = mask.ravel().astype(bool)
    pairs_idx = pairs_idx[inl]
    xy1 = xy1[inl]
    xy2 = xy2[inl]
    inlier_mask = (np.ones(len(xy1)) * 1).astype(np.uint8)

    return {
        "pairs_idx": pairs_idx,
        "pts1": xy1,
        "pts2": xy2,
        "E": E,
        "inlier_mask": inlier_mask,
    }


def draw_matches_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    *,
    max_draw: int = 1000,
    radius: int = 3,
    thickness: int = 1,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Draw side-by-side matches with connecting lines.

    img1, img2: grayscale or BGR images
    pts1, pts2: (N,2) float32 xy pixel coordinates
    inlier_mask: optional (N,) mask to pre-select inliers
    """
    if img1.ndim == 2:
        left = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        left = img1.copy()
    if img2.ndim == 2:
        right = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        right = img2.copy()

    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right

    N = len(pts1)
    if N == 0:
        return canvas
    mask = np.ones(N, dtype=bool) if inlier_mask is None else inlier_mask.astype(bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return canvas
    if max_draw is not None and idx.size > max_draw:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_draw, replace=False)

    offset_x = left.shape[1]
    for k in idx:
        x1, y1 = float(pts1[k, 0]), float(pts1[k, 1])
        x2, y2 = float(pts2[k, 0]) + float(offset_x), float(pts2[k, 1])
        c = (
            int(37 * (k % 7) + 80) % 255,
            int(53 * (k % 5) + 60) % 255,
            int(97 * (k % 9) + 40) % 255,
        )
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.circle(canvas, p1, radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p1, p2, c, thickness, lineType=cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    params = SiftParams(img_dims=(1440, 1920), depth_dims=(192, 256))
    sift = Sift(params)

    idx1 = 37
    K1 = np.load("data/sidewalk/intrinsics.npy")[idx1]
    img1 = read_gray_bt709(f"data/sidewalk/images/{idx1}.png")
    depth1 = np.load(f"data/sidewalk/depths/{idx1}.npy")

    res1, snapshot1 = sift.compute(img1, depth1, K1, record=True)

    idx2 = 159
    K2 = np.load("data/sidewalk/intrinsics.npy")[idx2]
    img2 = read_gray_bt709(f"data/sidewalk/images/{idx2}.png")
    depth2 = np.load(f"data/sidewalk/depths/{idx2}.npy")

    res2, snapshot2 = sift.compute(img2, depth2, K2, record=True)

    # Load the full-resolution image for visualization (grayscale)
    overlay = draw_random_affine_keypoints(
        cv2.imread(f"data/sidewalk/images/{idx1}.png", cv2.IMREAD_GRAYSCALE),
        res1.int_buffer,
        res1.float_buffer,
        res1.tilt_map,
        n_samples=1000,
        magnify=50,
    )
    cv2.imwrite("keypoints.png", overlay)
    # Match and draw
    matches = match_symmetric_ratio_and_essential(
        res1, res2, K1, K2, ratio=0.75, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    print(f"Num inlier matches: {matches['pts1'].shape[0]}")
    img1_vis = cv2.imread(f"data/sidewalk/images/{idx1}.png", cv2.IMREAD_GRAYSCALE)
    img2_vis = cv2.imread(f"data/sidewalk/images/{idx2}.png", cv2.IMREAD_GRAYSCALE)
    match_vis = draw_matches_side_by_side(
        img1_vis,
        img2_vis,
        matches["pts1"],
        matches["pts2"],
        matches["inlier_mask"],
        max_draw=1000,
        thickness=2,
    )
    cv2.imwrite("matches.png", match_vis)
