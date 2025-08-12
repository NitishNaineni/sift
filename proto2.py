from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable
import numpy as np
import cv2
import numba
from numba import cuda
from numba.cuda import libdevice as ld
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import math
import warnings
from numba.core.errors import NumbaPerformanceWarning
import os
import cupy as cp


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


def read_gray_bt709(path: str) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    return (im.astype(np.float32) * W709_BGR).sum(axis=2) / 256.0


@dataclass
class SiftParams:
    img_dims: tuple[int, int]
    n_oct: int = -1
    n_spo: int = 3
    sigma_in: float = 0.5
    delta_min: float = 0.5
    sigma_min: float = 0.8

    max_extrema: int = 10_000
    max_keypoints: int = 10_000

    C_dog: float = 0.013333333
    C_edge: float = 10.0

    lambda_ori: float = 1.5

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
    # keypoint count, overflow count
    counter: DeviceNDArray = field(
        default_factory=lambda: cuda.to_device(np.zeros(3, dtype=np.int32))
    )


@dataclass
class KeypointsHost:
    int_buffer: np.ndarray  # o, s, y_int, x_int
    float_buffer: np.ndarray  # y_world, x_world, sigma, orientation
    descriptors: np.ndarray  # 128-dim SIFT descriptor per keypoint (uint8)
    # keypoint count, overflow count
    counter: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.int32)
    )


@dataclass
class SiftData:
    input_img: DeviceNDArray
    seed_img: DeviceNDArray
    scratch: tuple[DeviceNDArray, ...]
    gss: tuple[DeviceNDArray, ...]
    dog: tuple[DeviceNDArray, ...]
    mag: tuple[DeviceNDArray, ...]
    ori: tuple[DeviceNDArray, ...]
    extrema: Extrema
    keypoints: Keypoints
    keypoints_host: KeypointsHost


def _alloc_octave_tensors(params: SiftParams, octave_index: int):
    height, width = params.gss_shapes[octave_index]
    num_gss_scales = params.n_spo + 3
    num_dog_scales = params.n_spo + 2

    gss = cuda.device_array((num_gss_scales, height, width), np.float32)
    dog = cuda.device_array((num_dog_scales, height, width), np.float32)
    scratch = cuda.device_array((height, width), np.float32)
    mag = cuda.device_array((num_gss_scales, height, width), np.float32)
    ori = cuda.device_array((num_gss_scales, height, width), np.float32)

    return gss, dog, scratch, mag, ori


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
    )

def create_keypoints_host(params: SiftParams) -> KeypointsHost:
    n = params.max_keypoints
    return KeypointsHost(
        int_buffer=np.empty((n, 4), dtype=np.int32),
        float_buffer=np.empty((n, 4), dtype=np.float32),
        descriptors=np.empty((n, 128), dtype=np.uint8),
    )


def create_sift_data(params: SiftParams) -> SiftData:
    gss, dog, scratch, mag, ori = zip(
        *(
            _alloc_octave_tensors(params, octave_index)
            for octave_index in range(params.n_oct)
        )
    )
    h0, w0 = params.gss_shapes[0]
    return SiftData(
        input_img=cuda.device_array(params.img_dims, np.float32),
        seed_img=cuda.device_array((h0, w0), np.float32),
        scratch=tuple(scratch),
        gss=tuple(gss),
        dog=tuple(dog),
        mag=tuple(mag),
        ori=tuple(ori),
        extrema=create_extrema(params),
        keypoints=create_keypoints(params),
        keypoints_host = create_keypoints_host(params)
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
    tile = cuda.shared.array(shape=0, dtype=numba.float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    h, w_in = src.shape
    bs = cuda.blockDim.x

    tile_w = bs + 2 * radius
    base_x = cuda.blockIdx.x * bs - radius

    for i in range(tx, tile_w, bs):
        lx = base_x + i
        tile[i] = src[y, mirror(lx, w_in)]
    cuda.syncthreads()

    if x < w_in and y < h:
        center = tile[tx + radius]
        acc = numba.float32(center * g[0])

        for k in range(1, radius + 1):
            left = tile[tx + radius - k]
            right = tile[tx + radius + k]
            acc += numba.float32(g[k] * (left + right))

        dst[y, x] = acc


@cuda.jit(cache=True, fastmath=True)
def gauss_v(src, dst, g, radius):
    tile = cuda.shared.array(shape=0, dtype=numba.float32)

    x, y = cuda.grid(2)
    ty = cuda.threadIdx.y
    h_in, w_in = src.shape
    bs = cuda.blockDim.y

    tile_h = bs + 2 * radius
    base_y = cuda.blockIdx.y * bs - radius

    for i in range(ty, tile_h, bs):
        ly = base_y + i
        tile[i] = src[mirror(ly, h_in), x]
    cuda.syncthreads()

    if x < w_in and y < h_in:
        center = tile[ty + radius]
        acc = numba.float32(center * g[0])

        for k in range(1, radius + 1):
            up = tile[ty + radius - k]
            down = tile[ty + radius + k]
            acc += numba.float32(g[k] * (up + down))

        dst[y, x] = acc


@cuda.jit(cache=True, fastmath=True)
def downsample_kernel(src, dst):
    x, y = cuda.grid(2)
    h, w = dst.shape
    if x < w and y < h:
        dst[y, x] = src[y * 2, x * 2]


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


def upscale(src, dst, delta_min, stream):
    assert delta_min <= 1.0
    hi, wi = src.shape
    ho = int(math.floor(hi / delta_min))
    wo = int(math.floor(wi / delta_min))
    if dst.shape != (ho, wo):
        raise ValueError(
            f"dst.shape must be {(ho, wo)} for delta_min={delta_min}, got {dst.shape}"
        )

    grid = ((wo + TX - 1) // TX, (ho + TY - 1) // TY)
    oversample_bilinear_kernel[grid, (TX, TY), stream](
        src, dst, numba.float32(delta_min)
    )


def gaussian_symm_kernel(sigma: float) -> tuple[DeviceNDArray, int]:
    radius = int(math.ceil(4.0 * float(sigma)))

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
    th = 128
    v_grid = (img_in.shape[1], (img_in.shape[0] + th - 1) // th)
    gauss_v[v_grid, (1, th), stream, (th + 2 * radius) * 4](
        img_in, scratch, gauss_kernel, radius
    )
    h_grid = ((img_in.shape[1] + th - 1) // th, img_in.shape[0])
    gauss_h[h_grid, (th,), stream, (th + 2 * radius) * 4](
        scratch, img_out, gauss_kernel, radius
    )


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
    mag = data.mag[octave_index]
    ori = data.ori[octave_index]
    scratch = data.scratch[octave_index]
    num_scales_total = params.n_spo + 3
    compute_mag_ori(gss[0], mag[0], ori[0], stream)
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
        compute_mag_ori(gss[scale_index], mag[scale_index], ori[scale_index], stream)
    if record:
        return gss.copy_to_host(stream=stream)
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
def gradient_kernel(img, mag, ori):
    tile = cuda.shared.array(shape=0, dtype=numba.float32)

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

    m = ld.sqrtf(gx * gx + gy * gy)
    mag[y, x] = m
    ori[y, x] = wrap_angle(ld.atan2f(gx, gy))

def compute_mag_ori(img, mag, ori, stream):
    h, w = img.shape
    grid = ((w + TX - 1) // TX, (h + TY - 1) // TY)
    gradient_kernel[grid, (TX, TY), stream, (TX + 2) * (TY + 2) * 4](img, mag, ori)




@cuda.jit(cache=True, fastmath=True)
def orientation_kernel(
    mag,
    ori,
    int_buf,
    float_buf,
    n_extrema,
    key_float,
    key_int,
    kp_counter,
    oct_idx,
    lambda_ori,
    delta_min,
):
    kp_idx = cuda.blockIdx.x
    if kp_idx >= n_extrema[0] or int_buf[kp_idx, 0] != oct_idx:
        return
    s = int_buf[kp_idx, 1]
    scale = delta_min * (1 << oct_idx)
    y0 = float_buf[kp_idx, 0] / scale
    x0 = float_buf[kp_idx, 1] / scale
    sigma_w = float_buf[kp_idx, 2]
    sigma_oct = sigma_w / scale
    R = 3.0 * lambda_ori * sigma_oct
    radius = int(R + 0.5)
    radius = 1 if radius == 0 else radius
    h, w = mag.shape[1:]
    g_sigma = lambda_ori * sigma_oct
    inv_2sig2 = 1.0 / (2.0 * g_sigma * g_sigma)
    bin_scale = numba.float32(ORI_BINS / TWO_PI)
    hist = cuda.shared.array(ORI_BINS, numba.float32)
    tflat = cuda.threadIdx.y * TX + cuda.threadIdx.x
    if tflat < ORI_BINS:
        hist[tflat] = 0.0
    cuda.syncthreads()
    siMin = 0 if (y0 - R + 0.5) < 0.0 else int(y0 - R + 0.5)
    sjMin = 0 if (x0 - R + 0.5) < 0.0 else int(x0 - R + 0.5)
    siMax_f = y0 + R + 0.5
    sjMax_f = x0 + R + 0.5
    siMax = h - 1 if siMax_f > (h - 1) else int(siMax_f)
    sjMax = w - 1 if sjMax_f > (w - 1) else int(sjMax_f)
    height = siMax - siMin + 1
    width = sjMax - sjMin + 1
    for dy in range(cuda.threadIdx.y, height, TY):
        yy = siMin + dy
        dyf = numba.float32(yy) - numba.float32(y0)
        for dx in range(cuda.threadIdx.x, width, TX):
            xx = sjMin + dx
            dxf = numba.float32(xx) - numba.float32(x0)
            m = mag[s, yy, xx]
            if m == 0.0:
                continue
            a = ori[s, yy, xx]
            wgt = m * ld.expf(-(dxf * dxf + dyf * dyf) * inv_2sig2)
            bin_f = a * bin_scale + numba.float32(0.5)
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
            vmax = vmax if vmax > hist[i] else hist[i]
        if vmax == 0.0:
            return
        thr = ORI_THRESHOLD * vmax
        for i in range(ORI_BINS):
            p, c, n = hist[(i - 1) % ORI_BINS], hist[i], hist[(i + 1) % ORI_BINS]
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
            key_float[out, 0] = float_buf[kp_idx, 0]  # y_world
            key_float[out, 1] = float_buf[kp_idx, 1]  # x_world
            key_float[out, 2] = sigma_w  # sigma
            key_float[out, 3] = theta  # orientation
            key_int[out, 0] = oct_idx  # o
            key_int[out, 1] = s  # s
            key_int[out, 2] = int_buf[kp_idx, 2]  # y_int
            key_int[out, 3] = int_buf[kp_idx, 3]  # x_int


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
    mag, ori, key_float, key_int, kctr, desc, oct_idx, delta_min
):
    kp_idx = cuda.blockIdx.x
    if kp_idx < kctr[1] or kp_idx >= kctr[0] or key_int[kp_idx, 0] != oct_idx:
        return
    s = key_int[kp_idx, 1]
    yw = key_float[kp_idx, 0]
    xw = key_float[kp_idx, 1]
    sigma = key_float[kp_idx, 2]
    theta0 = key_float[kp_idx, 3]
    scale = delta_min * (1 << oct_idx)
    x0, y0 = xw / scale, yw / scale
    radiusF = LAMBDA_DESC * sigma / scale
    inv_2sig2 = 1.0 / (2.0 * radiusF * radiusF)
    bin_scale = NORIBIN / TWO_PI
    half_bins = (NHIST - 1.0) * 0.5
    inv_cell = NHIST / (2.0 * radiusF)

    R = (1.0 + 1.0 / NHIST) * radiusF
    Rp = ld.sqrtf(2.0) * R

    h, w = mag.shape[1:]
    siMin = 0 if (y0 - Rp + 0.5) < 0.0 else int(y0 - Rp + 0.5)
    sjMin = 0 if (x0 - Rp + 0.5) < 0.0 else int(x0 - Rp + 0.5)
    siMax_f = y0 + Rp + 0.5
    sjMax_f = x0 + Rp + 0.5
    siMax = h - 1 if siMax_f > (h - 1) else int(siMax_f)
    sjMax = w - 1 if sjMax_f > (w - 1) else int(sjMax_f)
    height = max(0, siMax - siMin)
    width = max(0, sjMax - sjMin)

    c, snt = ld.cosf(theta0), ld.sinf(theta0)
    hist = cuda.shared.array(DESC_LEN, numba.float32)
    tflat = cuda.threadIdx.y * TX + cuda.threadIdx.x
    if tflat < DESC_LEN:
        hist[tflat] = 0.0
    cuda.syncthreads()
    for py in range(cuda.threadIdx.y, height, TY):
        yy = siMin + py
        dy0 = yy - y0
        for px in range(cuda.threadIdx.x, width, TX):
            xx = sjMin + px
            dx0 = xx - x0

            dx = dy0 * c + dx0 * snt
            dy = -dy0 * snt + dx0 * c
            u = dy * inv_cell + half_bins
            v = dx * inv_cell + half_bins
            if not (ld.fabsf(dx) < R and ld.fabsf(dy) < R):
                continue
            m = mag[s, yy, xx]
            if m == 0.0:
                continue
            a = wrap_angle(ori[s, yy, xx] - theta0)
            ob = a * bin_scale
            u0 = int(ld.floorf(u))
            du = u - u0
            v0 = int(ld.floorf(v))
            dv = v - v0
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
        l2 = numba.float32(0.0)
        for i in range(DESC_LEN):
            l2 += hist[i] * hist[i]
        norm = ld.sqrtf(l2) + 1e-12
        inv = 1.0 / norm

        l2p = numba.float32(0.0)
        for i in range(DESC_LEN):
            v = hist[i] * inv
            v = 0.2 if v > 0.2 else v
            hist[i] = v
            l2p += v * v
        norm2 = ld.sqrtf(l2p) + 1e-12
        inv2 = 1.0 / norm2

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
        data.mag[octave_index],
        data.ori[octave_index],
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        data.keypoints.float_buffer,
        data.keypoints.int_buffer,
        data.keypoints.counter,
        octave_index,
        params.lambda_ori,
        params.delta_min,
    )

    descriptor_kernel[(params.max_keypoints,), (TX, TY), stream](
        data.mag[octave_index],
        data.ori[octave_index],
        data.keypoints.float_buffer,
        data.keypoints.int_buffer,
        data.keypoints.counter,
        data.keypoints.descriptors,
        octave_index,
        params.delta_min,
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


def compute_octave(
    data: SiftData,
    params: SiftParams,
    octave_index: int,
    stream,
    record: bool = False,
) -> Optional[Dict[str, object]]:
    snapshot: dict[str, object] = {}

    data.extrema.counter.copy_to_device(np.array([0, 0], dtype=np.int32), stream)

    if octave_index == 0:
        set_seed(data, params, stream)
    else:
        set_first_scale(data, params, octave_index, stream)

    snapshot["gss"] = compute_gss(
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
    data: SiftData, params: SiftParams, stream, img, record: bool = False
) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []

    data.input_img.copy_to_device(img.astype(np.float32), stream)
    data.extrema.counter.copy_to_device(np.array([0, 0], dtype=np.int32), stream)
    data.keypoints.counter.copy_to_device(np.array([0, 0, 0], dtype=np.int32), stream)

    for o in range(params.n_oct):
        snapshot = compute_octave(data, params, o, stream, record)
        snapshots.append(snapshot)


    data.keypoints.int_buffer.copy_to_host(data.keypoints_host.int_buffer, stream)
    data.keypoints.float_buffer.copy_to_host(data.keypoints_host.float_buffer, stream)
    data.keypoints.descriptors.copy_to_host(data.keypoints_host.descriptors, stream)
    data.keypoints.counter.copy_to_host(data.keypoints_host.counter, stream)
    
    return snapshots


class Sift:
    def __init__(self, params: SiftParams):
        self.params = params
        self.data = create_sift_data(params)
        self._cp_stream = cp.cuda.Stream(non_blocking=True)
        self._stream = cuda.external_stream(self._cp_stream.ptr)

    def compute(self, img_path: str, record: bool = False) -> KeypointsHost:
        img = read_gray_bt709(img_path)
        assert img.shape == self.params.img_dims
        compute(self.data, self.params, self._stream, img, record=record)
        return self.data.keypoints_host

    def compute_many(self, img_paths: Iterable[str], record: bool = False):
        for p in img_paths:
            yield self.compute_from_path(p, record=record)



if __name__ == "__main__":

    params = SiftParams(img_dims = (640, 800))
    sift = Sift(params)

    img_paths = [f"data/oxford_affine/graf/img{i}.png" for i in range(1,7)]

    res1 = sift.compute(img_paths[0])
    res2 = sift.compute(img_paths[1])
