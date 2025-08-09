from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import cupy as cp
import numpy as np
import cv2
import numba
from numba import cuda
from numba.cuda import libdevice as ld
import math
import warnings
from numba.core.errors import NumbaPerformanceWarning
import os

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
)  # Cb,Cg,Cr


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

    C_dog: float = 0.013333333
    C_edge: float = 10.0

    # outputs
    sigmas: np.ndarray | None = None
    gss_shapes: np.ndarray | None = None
    inc_sigmas: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._update_octave_count()
        self.sigmas = self._make_sigmas()
        self.gss_shapes = self._make_gss_shapes()
        self.inc_sigmas = self._make_sigma_increments()

    def _update_octave_count(self) -> None:
        max_n_oct = math.floor(math.log2(min(self.img_dims) / self.delta_min / 12)) + 1
        self.n_oct = max_n_oct if self.n_oct == -1 else min(max_n_oct, self.n_oct)

    def _make_sigmas(self) -> np.ndarray:
        num_octaves = self.n_oct
        num_scales_total = self.n_spo + 3
        octave_indices = np.arange(num_octaves, dtype=np.float32)[:, None]
        scale_offsets = (np.arange(num_scales_total, dtype=np.float32) / self.n_spo)[
            None, :
        ]
        return (self.sigma_min * (2.0 ** (octave_indices + scale_offsets))).astype(
            np.float32
        )  # (num_octaves, num_scales_total)

    def _make_gss_shapes(self) -> np.ndarray:
        base = np.array(
            [
                int(self.img_dims[0] / self.delta_min),
                int(self.img_dims[1] / self.delta_min),
            ],
            dtype=np.int64,
        )
        hw = base // (1 << np.arange(self.n_oct, dtype=np.int64))[:, None]  # (O,2)
        return hw

    def _make_sigma_increments(self) -> np.ndarray:
        sig = self.sigmas.astype(np.float32)  # (num_octaves, num_scales_total)
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

        # C does sqrt(max(sig^2 - prev^2, 0)) / delta  in float
        diff2 = sig * sig - prev * prev
        np.maximum(diff2, 0.0, out=diff2, dtype=np.float32)
        np.sqrt(diff2, out=diff2)  # stays float32
        inc[:, :] = diff2 / deltas
        return inc

    def converted_dog_thresholds(self) -> tuple[float, float]:
        # Match C's convert_threshold: ((k_nspo-1)/(k_3-1)) * C_DoG
        k_nspo = math.exp(math.log(2.0) / float(self.n_spo))
        k_3 = math.exp(math.log(2.0) / 3.0)
        thresh = ((k_nspo - 1.0) / (k_3 - 1.0)) * float(self.C_dog)
        return 0.8 * thresh, thresh


@dataclass
class Extrema:
    int_buffer: cp.ndarray
    float_buffer: cp.ndarray
    counter: cp.ndarray = field(
        default_factory=lambda: cuda.device_array(2, dtype=np.int32)
    )


@dataclass
class SiftData:
    input_img: cp.ndarray
    seed_img: cp.ndarray
    scratch: list[cp.ndarray]
    gss: list[cp.ndarray]
    dog: list[cp.ndarray]
    extrema: Extrema


def _alloc_octave_tensors(params: SiftParams, o: int):
    h, w = params.gss_shapes[o]
    ng = params.n_spo + 3
    nd = params.n_spo + 2

    gss = cp.empty((ng, h, w), np.float32)
    dog = cp.empty((nd, h, w), np.float32)
    scratch = cp.empty((h, w), np.float32)

    return gss, dog, scratch


def create_extrema(params: SiftParams) -> Extrema:
    return Extrema(
        float_buffer=cp.empty((params.max_extrema, 4), np.float32),
        int_buffer=cp.empty((params.max_extrema, 4), np.int32),
    )


def create_sift_data(params: SiftParams) -> SiftData:
    gss, dog, scratch = zip(
        *(_alloc_octave_tensors(params, o) for o in range(params.n_oct))
    )
    h0, w0 = params.gss_shapes[0]
    return SiftData(
        input_img=cp.empty(params.img_dims, np.float32),
        seed_img=cp.empty((h0, w0), np.float32),
        scratch=tuple(scratch),
        gss=tuple(gss),
        dog=tuple(dog),
        extrema=create_extrema(params),
    )


@cuda.jit(cache=True)
def oversample_bilinear_kernel(src, dst, delta_min):
    j_out, i_out = cuda.grid(2)

    ho, wo = dst.shape
    if j_out >= wo or i_out >= ho:
        return

    hi, wi = src.shape

    # C: x = i*delta; y = j*delta
    x = numba.float32(i_out) * delta_min
    y = numba.float32(j_out) * delta_min

    # C: int cast (truncation)
    im = int(x)
    jm = int(y)
    ip = im + 1
    jp = jm + 1

    # symmetrization (exact same mapping)
    if ip >= hi:
        ip = 2 * hi - 1 - ip
    if im >= hi:
        im = 2 * hi - 1 - im
    if jp >= wi:
        jp = 2 * wi - 1 - jp
    if jm >= wi:
        jm = 2 * wi - 1 - jm

    # fractional parts use floorf in C
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


@cuda.jit(device=True, inline=True, cache=True)
def mirror(i: int, n: int) -> int:
    # C's symmetrized_coordinates:
    # ll = 2*n; i = (i+ll)%ll; if(i>n-1) i = ll-1-i;
    ll = n << 1
    i = (i + ll) % ll
    if i > n - 1:
        i = ll - 1 - i
    return i


@cuda.jit(cache=True)
def gauss_h(src, dst, g, radius):
    # dynamic shared tile (float32)
    tile = cuda.shared.array(shape=0, dtype=numba.float32)

    x, y = cuda.grid(2)  # x: column, y: row
    tx = cuda.threadIdx.x
    h, w_in = src.shape
    bs = cuda.blockDim.x

    tile_w = bs + 2 * radius
    base_x = cuda.blockIdx.x * bs - radius

    # cooperative load with mirroring
    for i in range(tx, tile_w, bs):
        lx = base_x + i
        tile[i] = src[y, mirror(lx, w_in)]
    cuda.syncthreads()

    if x < w_in and y < h:
        center = tile[tx + radius]
        acc = numba.float32(center * g[0])  # center term

        # pair-sum in the SAME order as C code
        for k in range(1, radius + 1):
            left = tile[tx + radius - k]
            right = tile[tx + radius + k]
            acc += numba.float32(g[k] * (left + right))

        dst[y, x] = acc


@cuda.jit(cache=True)
def gauss_v(src, dst, g, radius):
    tile = cuda.shared.array(shape=0, dtype=numba.float32)

    x, y = cuda.grid(2)  # x: column, y: row
    ty = cuda.threadIdx.y
    h_in, w_in = src.shape
    bs = cuda.blockDim.y

    tile_h = bs + 2 * radius
    base_y = cuda.blockIdx.y * bs - radius

    # cooperative load with mirroring
    for i in range(ty, tile_h, bs):
        ly = base_y + i
        tile[i] = src[mirror(ly, h_in), x]
    cuda.syncthreads()

    if x < w_in and y < h_in:
        center = tile[ty + radius]
        acc = numba.float32(center * g[0])  # center term

        # pair-sum in the SAME order as C code
        for k in range(1, radius + 1):
            up = tile[ty + radius - k]
            down = tile[ty + radius + k]
            acc += numba.float32(g[k] * (up + down))

        dst[y, x] = acc


@cuda.jit(cache=True)
def downsample_kernel(src, dst):
    x, y = cuda.grid(2)
    h, w = dst.shape
    if x < w and y < h:
        dst[y, x] = src[y * 2, x * 2]


@cuda.jit(cache=True)
def dog_diff_kernel(gss_in, dog_out):
    s = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ns, h, w = dog_out.shape
    if s < ns and y < h and x < w:
        dog_out[s, y, x] = gss_in[s + 1, y, x] - gss_in[s, y, x]


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


def upscale(src, dst, delta_min):
    assert delta_min <= 1.0
    hi, wi = src.shape
    ho = int(math.floor(hi / delta_min))
    wo = int(math.floor(wi / delta_min))
    if dst.shape != (ho, wo):
        raise ValueError(
            f"dst.shape must be {(ho, wo)} for delta_min={delta_min}, got {dst.shape}"
        )

    grid = ((wo + TX - 1) // TX, (ho + TY - 1) // TY)
    oversample_bilinear_kernel[grid, (TX, TY)](src, dst, numba.float32(delta_min))


def gaussian_symm_kernel(sigma: float) -> tuple[np.ndarray, int]:
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

    return cp.array(g), radius


def gaussian_blur(img_in, img_out, scratch, sigma):
    g_dev, r = gaussian_symm_kernel(sigma)
    th = 128
    v_grid = (img_in.shape[1], (img_in.shape[0] + th - 1) // th)
    gauss_v[v_grid, (1, th), 0, (th + 2 * r) * 4](img_in, scratch, g_dev, r)
    h_grid = ((img_in.shape[1] + th - 1) // th, img_in.shape[0])
    gauss_h[h_grid, (th,), 0, (th + 2 * r) * 4](scratch, img_out, g_dev, r)


def compute_gss(data: SiftData, params: SiftParams, o: int):
    gss = data.gss[o]
    scratch = data.scratch[o]
    ns = params.n_spo + 3
    for s in range(1, ns):
        gaussian_blur(gss[s - 1], gss[s], scratch, params.inc_sigmas[o, s])


def compute_dog(data: SiftData, p: SiftParams, o: int):
    gss, dog = data.gss[o], data.dog[o]
    h, w = p.gss_shapes[o]
    ns = p.n_spo + 2
    threads = (16, 16, 4)  # (x, y, z)
    grid = (
        (w + threads[0] - 1) // threads[0],
        (h + threads[1] - 1) // threads[1],
        (ns + threads[2] - 1) // threads[2],
    )
    dog_diff_kernel[grid, threads](gss, dog)


def detect_extrema(data: SiftData, params: SiftParams, o: int):
    data.extrema.counter[0] = 0
    dog_o = data.dog[o]
    h, w = params.gss_shapes[o]
    threads = (2, 8, 8)
    blocks = (
        (params.n_spo + 2 + threads[0] - 1) // threads[0],
        (h + threads[1] - 1) // threads[1],
        (w + threads[2] - 1) // threads[2],
    )
    find_and_record_extrema_kernel[blocks, threads](
        dog_o,
        o,
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        params.max_extrema,
        params.sigma_min,
        params.n_spo,
        params.delta_min,
    )


def compute_octave(data: SiftData, params: SiftParams, o: int):
    compute_gss(data, params, o)
    compute_dog(data, params, o)
    detect_extrema(data, params, o)


def set_seed(data: SiftData, params: SiftParams):
    assert params.sigma_min >= params.sigma_in
    upscale(data.input_img, data.seed_img, params.delta_min)
    gaussian_blur(
        data.seed_img, data.gss[0][0], data.scratch[0], params.inc_sigmas[0, 0]
    )


def set_first_scale(data: SiftData, params: SiftParams, o: int):
    src, dst = data.gss[o - 1][params.n_spo], data.gss[o][0]
    h, w = params.gss_shapes[o]
    grid = ((w + TX - 1) // TX, (h + TY - 1) // TY)
    downsample_kernel[grid, (TX, TY)](src, dst)


def compute(data: SiftData, params: SiftParams):
    set_seed(data, params)
    for o in range(params.n_oct):
        compute_octave(data, params, o)
        if o < params.n_oct - 1:
            set_first_scale(data, params, o + 1)


if __name__ == "__main__":
    img_path = Path("data/oxford_affine/graf/img1.png")
    img = read_gray_bt709(img_path)

    sift_params = SiftParams(img.shape)
    sift_data = create_sift_data(sift_params)

    # copy to GPU (float32)
    sift_data.input_img[...] = cp.asarray(img, dtype=cp.float32)

    compute(sift_data, sift_params)
