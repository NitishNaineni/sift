from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from math import ceil, floor, log2, sqrt
from pathlib import Path
from typing import Iterable, List, Tuple

import cupy as cp
import numba
import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda import libdevice as ld
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from PIL import Image
import cuda.bindings.runtime as rt

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


@cuda.jit(device=True, inline=True, cache=True)
def wrap_angle(theta: numba.float32) -> numba.float32:
    return ld.fmodf(ld.fmodf(theta, TWO_PI) + TWO_PI, TWO_PI)


def read_image(path: str | Path) -> np.ndarray:
    img = np.asarray(Image.open(path)).astype(np.float32) / 255.0
    return img @ np.array(
        [0.212639005871510, 0.715168678767756, 0.072192315360734], dtype=np.float32
    )


def _gauss_kernel(sigma: float, dtype=np.float32) -> Tuple[DeviceNDArray, int]:
    r = max(1, int(ceil(4.0 * sigma)))
    x = cp.arange(-r, r + 1, dtype=dtype)
    w = cp.exp(-(x * x) / (2.0 * sigma * sigma))
    w /= w.sum()
    return cuda.to_device(w), r


@dataclass
class SiftParams:
    img_dims: Tuple[int, int]
    n_oct: int = 8
    n_spo: int = 3
    sigma_in: float = 0.5
    delta_min: float = 0.5
    sigma_min: float = 0.8
    C_dog: float = 0.013333333
    C_edge: float = 10.0
    lambda_ori: float = 1.5
    max_extrema: int = 10_000
    max_keypoints: int = 10_000
    sigmas: np.ndarray | None = None
    gss_shapes: List[Tuple[int, int]] | None = None
    kernels: List[DeviceNDArray] | None = None
    radii: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._update_octave_count()
        self._scale_invariant_C_dog()
        self.sigmas = self._sigmas()
        self.gss_shapes = self._gss_shapes()
        self.kernels, self.radii = self._kernels_and_radii()
        self.seed_kernel, self.seed_radius = _gauss_kernel(
            sqrt(self.sigma_min**2 - self.sigma_in**2) / self.delta_min
        )

    def _scale_invariant_C_dog(self) -> None:
        kn = np.exp(np.log(2) / self.n_spo)
        k3 = np.exp(np.log(2) / 3.0)
        self.C_dog *= (kn - 1) / (k3 - 1)

    def _update_octave_count(self) -> None:
        max_n_oct = floor(log2(min(self.img_dims) / self.delta_min / 12)) + 1
        self.n_oct = max_n_oct if self.n_oct == -1 else min(max_n_oct, self.n_oct)

    def _sigmas(self) -> np.ndarray:
        s = np.arange(self.n_spo + 3, dtype=np.float32)[None, :]
        o = np.arange(self.n_oct, dtype=np.float32)[:, None]
        return self.sigma_min * np.power(2.0, o + s / self.n_spo)

    def _gss_shapes(self) -> List[Tuple[int, int]]:
        h = int(round(self.img_dims[0] / self.delta_min))
        w = int(round(self.img_dims[1] / self.delta_min))
        out = []
        for _ in range(self.n_oct):
            out.append((h, w))
            h, w = h // 2, w // 2
        return out

    def _kernels_and_radii(self) -> Tuple[List[DeviceNDArray], np.ndarray]:
        ns = self.n_spo + 3
        radii = np.zeros((self.n_oct, ns), dtype=np.int32)
        kernels: List[DeviceNDArray] = []
        for o in range(self.n_oct):
            for s in range(1, ns):
                inc = float(
                    sqrt(self.sigmas[o, s] ** 2 - self.sigmas[o, s - 1] ** 2)
                    / (self.delta_min * (1 << o))
                )
                k, r = _gauss_kernel(inc)
                kernels.append(k)
                radii[o, s] = r
        return kernels, radii


@dataclass
class Extrema:
    int_buffer: DeviceNDArray
    float_buffer: DeviceNDArray
    counter: DeviceNDArray = field(
        default_factory=lambda: cuda.device_array(2, dtype=np.int32)
    )


@dataclass
class Keypoints:
    positions: DeviceNDArray
    descriptors: DeviceNDArray
    scales: DeviceNDArray
    orientations: DeviceNDArray
    osl: DeviceNDArray
    counter: DeviceNDArray = field(
        default_factory=lambda: cuda.device_array(2, dtype=np.int32)
    )


@dataclass
class KeypointsHost:
    positions: np.ndarray
    descriptors: np.ndarray
    scales: np.ndarray
    orientations: np.ndarray
    osl: np.ndarray


@dataclass
class SiftData:
    input_img: DeviceNDArray
    seed_img: DeviceNDArray
    scratch: List[DeviceNDArray]
    gss: List[DeviceNDArray]
    dog: List[DeviceNDArray]
    grad_mag: List[DeviceNDArray]
    grad_ang: List[DeviceNDArray]
    extrema: Extrema
    keypoints: Keypoints
    keypoints_host: KeypointsHost


def _alloc_octave_tensors(p: SiftParams, o: int):
    h, w = p.gss_shapes[o]
    ng = p.n_spo + 3
    nd = p.n_spo + 2
    gss = cuda.device_array((ng, h, w), np.float32)
    dog = cuda.device_array((nd, h, w), np.float32)
    scratch = cuda.device_array((h, w), np.float32)
    grad_mag = cuda.device_array((ng, h, w), np.float32)
    grad_ang = cuda.device_array((ng, h, w), np.float32)
    return gss, dog, scratch, grad_mag, grad_ang


def create_extrema(p: SiftParams) -> Extrema:
    return Extrema(
        float_buffer=cuda.device_array((p.max_extrema, 4), np.float32),
        int_buffer=cuda.device_array((p.max_extrema, 4), np.int32),
    )


def create_keypoints(p: SiftParams) -> Keypoints:
    n = p.max_keypoints
    return Keypoints(
        positions=cuda.device_array((n, 2), np.float32),
        descriptors=cuda.device_array((n, 128), np.uint8),
        scales=cuda.device_array(n, np.float32),
        orientations=cuda.device_array(n, np.float32),
        osl=cuda.device_array((n, 2), np.int32),
    )


def create_keypoints_host(p: SiftParams) -> KeypointsHost:
    n = p.max_keypoints
    sz_f32, sz_u8, sz_i32 = (
        np.dtype(t).itemsize for t in (np.float32, np.uint8, np.int32)
    )
    total = n * (2 * sz_f32 + 128 * sz_u8 + sz_f32 + 2 * sz_i32)
    buf = cp.cuda.alloc_pinned_memory(total)
    off = 0
    pos = np.frombuffer(buf, dtype=np.float32, count=n * 2, offset=off).reshape(n, 2)
    off += n * 2 * sz_f32
    desc = np.frombuffer(buf, dtype=np.uint8, count=n * 128, offset=off).reshape(n, 128)
    off += n * 128 * sz_u8
    sca = np.frombuffer(buf, dtype=np.float32, count=n, offset=off)
    off += n * sz_f32
    ori = np.frombuffer(buf, dtype=np.float32, count=n, offset=off)
    off += n * sz_f32
    osl = np.frombuffer(buf, dtype=np.int32, count=n * 2, offset=off).reshape(n, 2)
    return KeypointsHost(pos, desc, sca, ori, osl)


def create_sift_data(p: SiftParams) -> SiftData:
    gss, dog, scratch, grad_mag, grad_ang = zip(
        *(_alloc_octave_tensors(p, o) for o in range(p.n_oct))
    )
    h0, w0 = p.gss_shapes[0]
    return SiftData(
        input_img=cuda.device_array(p.img_dims, np.float32),
        seed_img=cuda.device_array((h0, w0), np.float32),
        scratch=list(scratch),
        gss=list(gss),
        dog=list(dog),
        grad_mag=list(grad_mag),
        grad_ang=list(grad_ang),
        extrema=create_extrema(p),
        keypoints=create_keypoints(p),
        keypoints_host=create_keypoints_host(p),
    )


@cuda.jit(device=True, inline=True, cache=True)
def mirror(i: int, n: int) -> int:
    period = n << 1
    m = ((i % period) + period) % period
    return ld.min(m, period - 1 - m)


@cuda.jit(device=True, inline=True, cache=True)
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


@cuda.jit(device=True, inline=True, cache=True)
def mat_vec_mul_3x1(M, v, out):
    out[0] = M[0, 0] * v[0] + M[0, 1] * v[1] + M[0, 2] * v[2]
    out[1] = M[1, 0] * v[0] + M[1, 1] * v[1] + M[1, 2] * v[2]
    out[2] = M[2, 0] * v[0] + M[2, 1] * v[1] + M[2, 2] * v[2]


@cuda.jit(cache=True)
def gradient_kernel(img, mag, ang):
    x, y = cuda.grid(2)
    h, w = img.shape
    if 0 < x < w - 1 and 0 < y < h - 1:
        gx = 0.5 * (img[y, x + 1] - img[y, x - 1])
        gy = 0.5 * (img[y + 1, x] - img[y - 1, x])
        mag[y, x] = ld.sqrtf(gx * gx + gy * gy)
        ang[y, x] = wrap_angle(ld.atan2f(gy, gx))


@cuda.jit(cache=True)
def gauss_h(src, dst, w, radius):
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
        acc = numba.float32(0.0)
        for k in range(-radius, radius + 1):
            acc += tile[tx + radius + k] * w[k + radius]
        dst[y, x] = acc


@cuda.jit(cache=True)
def gauss_v(src, dst, w, radius):
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
        acc = numba.float32(0.0)
        for k in range(-radius, radius + 1):
            acc += tile[ty + radius + k] * w[k + radius]
        dst[y, x] = acc


@cuda.jit(cache=True)
def dog_diff_kernel(gss_in, dog_out):
    s = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ns, h, w = dog_out.shape
    if s < ns and y < h and x < w:
        dog_out[s, y, x] = gss_in[s + 1, y, x] - gss_in[s, y, x]


@cuda.jit(cache=True)
def upscale_kernel(src, dst, inv_y, inv_x):
    x_out, y_out = cuda.grid(2)
    h_out, w_out = dst.shape
    if x_out >= w_out or y_out >= h_out:
        return
    y_in = y_out * inv_y
    x_in = x_out * inv_x
    y0 = int(ld.floorf(y_in))
    x0 = int(ld.floorf(x_in))
    wy = numba.float32(y_in - y0)
    wx = numba.float32(x_in - x0)
    h_in, w_in = src.shape
    y0c, y1c = mirror(y0, h_in), mirror(y0 + 1, h_in)
    x0c, x1c = mirror(x0, w_in), mirror(x0 + 1, w_in)
    v00, v01 = src[y0c, x0c], src[y0c, x1c]
    v10, v11 = src[y1c, x0c], src[y1c, x1c]
    dst[y_out, x_out] = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * (
        (1 - wx) * v10 + wx * v11
    )


@cuda.jit(cache=True)
def downsample_kernel(src, dst):
    y, x = cuda.grid(2)
    h, w = dst.shape
    if y < h and x < w:
        dst[y, x] = src[y * 2, x * 2]


@cuda.jit(cache=True)
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
    scale = delta_min * (1 << o)
    float_buf[idx, 0] = y * scale
    float_buf[idx, 1] = x * scale
    float_buf[idx, 2] = sigma_min * (2.0 ** (o + s / n_spo))
    float_buf[idx, 3] = v


@cuda.jit(cache=True)
def refine_filter_kernel(
    dog_oct, int_buf, float_buf, ext_count, C_dog, C_edge, n_spo, sigma_min, delta_min
):
    idx = cuda.grid(1)
    if idx >= ext_count[0]:
        return
    o = int_buf[idx, 0]
    if o == -1:
        return
    s, y, x = int_buf[idx, 1], int_buf[idx, 2], int_buf[idx, 3]
    ns, h, w = dog_oct.shape
    g = cuda.local.array(3, dtype=numba.float32)
    Hm = cuda.local.array((3, 3), dtype=numba.float32)
    Hin = cuda.local.array((3, 3), dtype=numba.float32)
    off = cuda.local.array(3, dtype=numba.float32)
    valid = False
    for _ in range(5):
        if not (1 <= s < ns - 1 and 1 <= y < h - 1 and 1 <= x < w - 1):
            int_buf[idx, 0] = -1
            return
        g[0] = 0.5 * (dog_oct[s + 1, y, x] - dog_oct[s - 1, y, x])
        g[1] = 0.5 * (dog_oct[s, y + 1, x] - dog_oct[s, y - 1, x])
        g[2] = 0.5 * (dog_oct[s, y, x + 1] - dog_oct[s, y, x - 1])
        Hm[0, 0] = dog_oct[s + 1, y, x] + dog_oct[s - 1, y, x] - 2 * dog_oct[s, y, x]
        Hm[1, 1] = dog_oct[s, y + 1, x] + dog_oct[s, y - 1, x] - 2 * dog_oct[s, y, x]
        Hm[2, 2] = dog_oct[s, y, x + 1] + dog_oct[s, y, x - 1] - 2 * dog_oct[s, y, x]
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
        if ld.fabsf(off[0]) < 0.6 and ld.fabsf(off[1]) < 0.6 and ld.fabsf(off[2]) < 0.6:
            valid = True
            break
        s += (off[0] > 0.6) - (off[0] < -0.6)
        y += (off[1] > 0.6) - (off[1] < -0.6)
        x += (off[2] > 0.6) - (off[2] < -0.6)
        if not (1 <= s < ns - 1 and 1 <= y < h - 1 and 1 <= x < w - 1):
            break
    if not valid:
        int_buf[idx, 0] = -1
        return
    D_hat = dog_oct[s, y, x] + 0.5 * (g[0] * off[0] + g[1] * off[1] + g[2] * off[2])
    if ld.fabsf(D_hat) < C_dog:
        int_buf[idx, 0] = -1
        return
    Hxx, Hyy, Hxy = Hm[1, 1], Hm[2, 2], Hm[1, 2]
    det = Hxx * Hyy - Hxy * Hxy
    if det <= 0:
        int_buf[idx, 0] = -1
        return
    trace = Hxx + Hyy
    r = C_edge
    if (trace * trace) / det > ((r + 1) * (r + 1) / r):
        int_buf[idx, 0] = -1
        return
    int_buf[idx, 1], int_buf[idx, 2], int_buf[idx, 3] = s, y, x
    scale = delta_min * (1 << o)
    float_buf[idx, 0] = (y + off[1]) * scale
    float_buf[idx, 1] = (x + off[2]) * scale
    float_buf[idx, 2] = sigma_min * (2.0 ** (o + (s + off[0]) / n_spo))
    float_buf[idx, 3] = D_hat


@cuda.jit(cache=True)
def orientation_kernel_blockperkp(
    grad_mag,
    grad_ang,
    int_buf,
    float_buf,
    n_extrema,
    key_pos,
    key_scale,
    key_ori,
    key_osl,
    kp_counter,
    oct_idx,
    lambda_ori,
    delta_min,
):
    kp_idx = cuda.blockIdx.x
    if kp_idx >= n_extrema[0] or int_buf[kp_idx, 0] != oct_idx:
        return
    s = int_buf[kp_idx, 1]
    yy0 = int_buf[kp_idx, 2]
    xx0 = int_buf[kp_idx, 3]
    sigma_w = float_buf[kp_idx, 2]
    scale = delta_min * (1 << oct_idx)
    sigma_oct = sigma_w / scale
    radius = int(3.0 * lambda_ori * sigma_oct + 0.5)
    radius = 1 if radius == 0 else radius
    h, w = grad_mag.shape[1:]
    if (
        yy0 - radius < 1
        or yy0 + radius >= h - 1
        or xx0 - radius < 1
        or xx0 + radius >= w - 1
    ):
        return
    g_sigma = lambda_ori * sigma_oct
    inv_2sig2 = 1.0 / (2.0 * g_sigma * g_sigma)
    bin_scale = numba.float32(ORI_BINS / TWO_PI)
    hist = cuda.shared.array(ORI_BINS, numba.float32)
    tflat = cuda.threadIdx.y * TX + cuda.threadIdx.x
    if tflat < ORI_BINS:
        hist[tflat] = 0.0
    cuda.syncthreads()
    for dy in range(cuda.threadIdx.y, radius * 2 + 1, TY):
        yy = yy0 - radius + dy
        dyf = yy - yy0
        for dx in range(cuda.threadIdx.x, radius * 2 + 1, TX):
            xx = xx0 - radius + dx
            dxf = xx - xx0
            m = grad_mag[s, yy, xx]
            if m == 0.0:
                continue
            a = wrap_angle(grad_ang[s, yy, xx])
            wgt = m * ld.expf(-(dxf * dxf + dyf * dyf) * inv_2sig2)
            b = int(a * bin_scale + 0.5) % ORI_BINS
            cuda.atomic.add(hist, b, wgt)
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
            if c < thr or c <= p or c <= n:
                continue
            denom = p - 2.0 * c + n
            off = (p - n) / (2.0 * denom)
            theta = wrap_angle((i + off + 0.5) * (TWO_PI / ORI_BINS))
            out = cuda.atomic.add(kp_counter, 0, 1)
            if out >= key_pos.shape[0]:
                cuda.atomic.add(kp_counter, 0, -1)
                cuda.atomic.add(kp_counter, 1, 1)
                return
            key_pos[out, 0] = float_buf[kp_idx, 1]
            key_pos[out, 1] = float_buf[kp_idx, 0]
            key_scale[out] = sigma_w
            key_ori[out] = theta
            key_osl[out, 0] = oct_idx
            key_osl[out, 1] = s


@cuda.jit(cache=True)
def descriptor_kernel_blockperkp(
    grad_mag, grad_ang, kpos, kscale, kori, osl, kctr, desc, oct_idx, delta_min
):
    kp_idx = cuda.blockIdx.x
    if kp_idx >= kctr[0] or osl[kp_idx, 0] != oct_idx:
        return
    s = osl[kp_idx, 1]
    xw, yw = kpos[kp_idx]
    sigma = kscale[kp_idx]
    theta0 = kori[kp_idx]
    scale = delta_min * (1 << oct_idx)
    x0, y0 = xw / scale, yw / scale
    radiusF = LAMBDA_DESC * sigma / scale
    radius = max(1, int(radiusF + 0.5))
    dmin = ld.ceilf(ld.sqrtf(2.0) * radius)
    h, w = grad_mag.shape[1:]
    if x0 < dmin or x0 > w - 1 - dmin or y0 < dmin or y0 > h - 1 - dmin:
        return
    c, snt = ld.cosf(theta0), ld.sinf(theta0)
    inv_2sig2 = 1.0 / (2.0 * radiusF * radiusF)
    bin_scale = NORIBIN / TWO_PI
    patch = radius * 2 + 1
    hist = cuda.shared.array(DESC_LEN, numba.float32)
    tflat = cuda.threadIdx.y * TX + cuda.threadIdx.x
    if tflat < DESC_LEN:
        hist[tflat] = 0.0
    cuda.syncthreads()
    for py in range(cuda.threadIdx.y, patch, TY):
        yy = int(y0) - radius + py
        dy0 = yy - y0
        for px in range(cuda.threadIdx.x, patch, TX):
            xx = int(x0) - radius + px
            dx0 = xx - x0
            dx = dx0 * c + dy0 * snt
            dy = -dx0 * snt + dy0 * c
            u = dx / (2.0 * radiusF) * NHIST + (NHIST * 0.5 - 0.5)
            v = dy / (2.0 * radiusF) * NHIST + (NHIST * 0.5 - 0.5)
            R = (1.0 + 1.0 / NHIST) * radiusF
            if ld.fabsf(dx) >= R or ld.fabsf(dy) >= R:
                continue
            m = grad_mag[s, yy, xx]
            if m == 0.0:
                continue
            a = wrap_angle(grad_ang[s, yy, xx] - theta0)
            ob = a * bin_scale
            u0 = int(ld.floorf(u))
            du = u - u0
            v0 = int(ld.floorf(v))
            dv = v - v0
            o0 = int(ld.floorf(ob))
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
                                oo = (o0 + io) % NORIBIN
                                wo = (1 - do) if io == 0 else do
                                hidx = ((vv * NHIST + uu) * NORIBIN) + oo
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
            desc[kp_idx, i] = numba.uint8(255 if q > 255 else int(q + 0.5))


def _launch_gradient(img_in, mag_out, ang_out, stream):
    h, w = img_in.shape
    grid = ((w + TX - 1) // TX, (h + TY - 1) // TY)
    gradient_kernel[grid, (TX, TY), stream](img_in, mag_out, ang_out)


def gaussian_blur(img_in, img_out, scratch, weights, radius, stream):
    th = 128
    h_grid = ((img_in.shape[1] + th - 1) // th, img_in.shape[0])
    gauss_h[h_grid, (th,), stream, (th + 2 * radius) * 4](
        img_in, scratch, weights, radius
    )
    v_grid = (img_in.shape[1], (img_in.shape[0] + th - 1) // th)
    gauss_v[v_grid, (1, th), stream, (th + 2 * radius) * 4](
        scratch, img_out, weights, radius
    )


def compute_gss(data: SiftData, p: SiftParams, o: int, stream):
    gss, gm, ga, scratch = (
        data.gss[o],
        data.grad_mag[o],
        data.grad_ang[o],
        data.scratch[o],
    )
    ns = p.n_spo + 3
    base = o * (ns - 1)
    _launch_gradient(gss[0], gm[0], ga[0], stream)
    for s in range(1, ns):
        w = p.kernels[base + (s - 1)]
        r = p.radii[o, s]
        gaussian_blur(gss[s - 1], gss[s], scratch, w, r, stream)
        _launch_gradient(gss[s], gm[s], ga[s], stream)


def compute_dog(data: SiftData, p: SiftParams, o: int, stream):
    gss, dog = data.gss[o], data.dog[o]
    h, w = p.gss_shapes[o]
    ns = p.n_spo + 2
    threads = (4, 16, 4)
    grid = (
        (w + threads[2] - 1) // threads[2],
        (h + threads[1] - 1) // threads[1],
        (ns + threads[0] - 1) // threads[0],
    )
    dog_diff_kernel[grid, threads, stream](gss, dog)


def detect_extrema(data: SiftData, p: SiftParams, o: int, stream):
    cp.cuda.runtime.memsetAsync(
        int(data.extrema.counter.device_ctypes_pointer.value),
        0,
        data.extrema.counter.nbytes,
        stream.handle.value,
    )
    dog_o = data.dog[o]
    h, w = p.gss_shapes[o]
    threads = (2, 8, 8)
    blocks = (
        (p.n_spo + 2 + threads[0] - 1) // threads[0],
        (h + threads[1] - 1) // threads[1],
        (w + threads[2] - 1) // threads[2],
    )
    find_and_record_extrema_kernel[blocks, threads, stream](
        dog_o,
        o,
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        p.max_extrema,
        p.sigma_min,
        p.n_spo,
        p.delta_min,
    )


def refine_and_filter(data: SiftData, p: SiftParams, o: int, stream):
    threads = 128
    blocks = (p.max_extrema + threads - 1) // threads
    refine_filter_kernel[blocks, threads, stream](
        data.dog[o],
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        p.C_dog,
        p.C_edge,
        p.n_spo,
        p.sigma_min,
        p.delta_min,
    )


def assign_orientations(data: SiftData, p: SiftParams, o: int, stream):
    orientation_kernel_blockperkp[(p.max_extrema,), (TX, TY), stream](
        data.grad_mag[o],
        data.grad_ang[o],
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        data.keypoints.positions,
        data.keypoints.scales,
        data.keypoints.orientations,
        data.keypoints.osl,
        data.keypoints.counter,
        o,
        p.lambda_ori,
        p.delta_min,
    )


def build_descriptors(data: SiftData, p: SiftParams, o: int, stream):
    descriptor_kernel_blockperkp[(p.max_keypoints,), (TX, TY), stream](
        data.grad_mag[o],
        data.grad_ang[o],
        data.keypoints.positions,
        data.keypoints.scales,
        data.keypoints.orientations,
        data.keypoints.osl,
        data.keypoints.counter,
        data.keypoints.descriptors,
        o,
        p.delta_min,
    )


def upscale(src, dst, stream):
    hi, wi = src.shape
    ho, wo = dst.shape
    grid = ((wo + TX - 1) // TX, (ho + TY - 1) // TY)
    upscale_kernel[grid, (TX, TY), stream](src, dst, hi / ho, wi / wo)


def set_seed(data: SiftData, p: SiftParams, stream):
    assert p.sigma_min >= p.sigma_in
    upscale(data.input_img, data.seed_img, stream)
    gaussian_blur(
        data.seed_img,
        data.gss[0][0],
        data.scratch[0],
        p.seed_kernel,
        p.seed_radius,
        stream,
    )


def set_first_scale(data: SiftData, p: SiftParams, o: int, stream):
    src, dst = data.gss[o - 1][p.n_spo], data.gss[o][0]
    h, w = p.gss_shapes[o]
    grid = ((w + TX - 1) // TX, (h + TY - 1) // TY)
    downsample_kernel[grid, (TX, TY), stream](src, dst)


def _reset_counters(d: SiftData, stream):
    cp.cuda.runtime.memsetAsync(
        int(d.keypoints.counter.device_ctypes_pointer.value),
        0,
        d.keypoints.counter.nbytes,
        stream.handle.value,
    )
    cp.cuda.runtime.memsetAsync(
        int(d.extrema.counter.device_ctypes_pointer.value),
        0,
        d.extrema.counter.nbytes,
        stream.handle.value,
    )


def _compute_octave(d: SiftData, p: SiftParams, o: int, nb_stream):
    compute_gss(d, p, o, nb_stream)
    compute_dog(d, p, o, nb_stream)
    detect_extrema(d, p, o, nb_stream)
    refine_and_filter(d, p, o, nb_stream)
    assign_orientations(d, p, o, nb_stream)
    build_descriptors(d, p, o, nb_stream)


def _compute_all_octaves(d: SiftData, p: SiftParams, nb_stream):
    _reset_counters(d, nb_stream)
    set_seed(d, p, nb_stream)
    for o in range(p.n_oct):
        _compute_octave(d, p, o, nb_stream)
        if o < p.n_oct - 1:
            set_first_scale(d, p, o + 1, nb_stream)


class Sift:
    def __init__(self, params: SiftParams, use_graph: bool = True):
        self.params = params
        self.data = [create_sift_data(params), create_sift_data(params)]
        self.use_graph = use_graph
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.nb_stream = cuda.external_stream(self.stream.ptr)
        self.exec_graph = [None, None]
        self.graph = [None, None]
        if use_graph:
            for i in range(2):
                self._capture_graph(i)

    def _capture_graph(self, idx: int):
        with self.stream:
            rt.cudaStreamBeginCapture(
                self.stream.ptr, rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
            )
            _compute_all_octaves(self.data[idx], self.params, self.nb_stream)
            err, graph = rt.cudaStreamEndCapture(self.stream.ptr)
            _check_runtime_error(err)
            err, exec_graph = rt.cudaGraphInstantiate(graph, 0)
            _check_runtime_error(err)
            self.graph[idx] = graph
            self.exec_graph[idx] = exec_graph

    def __del__(self):
        if not self.use_graph:
            return
        destroy_exec = getattr(rt, "cudaGraphExecDestroy", None)
        destroy_graph = getattr(rt, "cudaGraphDestroy", None)
        if destroy_exec:
            for e in self.exec_graph:
                if e:
                    try:
                        destroy_exec(e)
                    except Exception:
                        pass
        if destroy_graph:
            for g in self.graph:
                if g:
                    try:
                        destroy_graph(g)
                    except Exception:
                        pass

    def process_images(self, paths: Iterable[str | Path]):
        paths = list(paths)
        if not paths:
            return
        h, w = self.params.img_dims
        img_bytes = h * w * 4
        h_in = [cp.cuda.alloc_pinned_memory(img_bytes) for _ in range(2)]
        up_ev = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
        comp_ev = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
        out_ev = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
        copy_stream = cp.cuda.Stream(non_blocking=True)
        nb_copy = cuda.external_stream(copy_stream.ptr)
        h_ctr_buf = [cp.cuda.alloc_pinned_memory(8) for _ in range(2)]
        h_ctr_view = [
            np.frombuffer(h_ctr_buf[i], dtype=np.int32, count=2) for i in range(2)
        ]

        def _h2d(idx: int, np_img: np.ndarray):
            np.asarray(h_in[idx]).view(np.float32)[: np_img.size] = np_img.ravel()
            cp.cuda.runtime.memcpyAsync(
                self.data[idx].input_img.device_ctypes_pointer.value,
                h_in[idx].ptr,
                np_img.nbytes,
                rt.cudaMemcpyKind.cudaMemcpyHostToDevice,
                copy_stream.ptr,
            )
            copy_stream.record(up_ev[idx])

        def _enqueue_d2h(idx: int):
            copy_stream.wait_event(comp_ev[idx])
            kd = self.data[idx].keypoints
            kh = self.data[idx].keypoints_host
            kd.counter.copy_to_host(h_ctr_view[idx], stream=nb_copy)
            kd.positions.copy_to_host(kh.positions, stream=nb_copy)
            kd.descriptors.copy_to_host(kh.descriptors, stream=nb_copy)
            kd.scales.copy_to_host(kh.scales, stream=nb_copy)
            kd.orientations.copy_to_host(kh.orientations, stream=nb_copy)
            kd.osl.copy_to_host(kh.osl, stream=nb_copy)
            copy_stream.record(out_ev[idx])

        np0 = read_image(paths[0])
        _h2d(0, np0)
        self.stream.wait_event(up_ev[0])
        self._launch_compute(0)
        self.stream.record(comp_ev[0])

        for i in range(1, len(paths) + 1):
            cur = i & 1
            prev = (i - 1) & 1
            _enqueue_d2h(prev)
            if i < len(paths):
                _h2d(cur, read_image(paths[i]))
                self.stream.wait_event(up_ev[cur])
                self._launch_compute(cur)
                self.stream.record(comp_ev[cur])
            out_ev[prev].synchronize()
            k = int(h_ctr_view[prev][0])
            yield _truncate_keypoints(self.data[prev].keypoints_host, k)

    def _launch_compute(self, idx: int):
        if self.use_graph:
            rt.cudaGraphLaunch(self.exec_graph[idx], self.stream.ptr)
        else:
            _compute_all_octaves(self.data[idx], self.params, self.nb_stream)


def _check_runtime_error(err):
    if err != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA runtime error: {rt.cudaGetErrorName(err)[1].decode()} – {rt.cudaGetErrorString(err)[1].decode()}"
        )


def _truncate_keypoints(kh: KeypointsHost, n: int) -> KeypointsHost:
    return KeypointsHost(
        kh.positions[:n],
        kh.descriptors[:n],
        kh.scales[:n],
        kh.orientations[:n],
        kh.osl[:n],
    )


if __name__ == "__main__":
    root = Path("data/oxford_affine/graf")
    images = sorted(root.glob("img*.png"))
    if not images:
        raise SystemExit("No images found")
    first = read_image(images[0])
    params = SiftParams(img_dims=first.shape)
    sift = Sift(params, use_graph=True)
    for path, kp in zip(images, sift.process_images(images)):
        print(f"{path.name}: {len(kp.positions)} keypoints")
