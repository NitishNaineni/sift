import warnings


import os
import numpy as np
from PIL import Image
import cuda.bindings.runtime as runtime

from dataclasses import dataclass, field
import cupy as cp
from math import floor, log2, sqrt, ceil


from numba import cuda
import numba
from numba.cuda import libdevice as ld
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda.cudadrv.devicearray import DeviceNDArray

os.environ["NUMBA_CUDA_ARRAY_INTERFACE_SYNC"] = "0"

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Explicitly enabling pynvjitlink is no longer necessary\.",
)

warnings.filterwarnings(
    "ignore",
    category=NumbaPerformanceWarning,
)


def read_img(path):
    img = np.asarray(Image.open(path)).astype(np.float32)
    img = img / 255
    img = img @ np.array(
        [0.212639005871510, 0.715168678767756, 0.072192315360734], dtype=np.float32
    )
    return img


def _gauss_kernel(sigma: float, dtype=np.float32):
    r = max(1, int(ceil(3.0 * sigma)))
    x = cp.arange(-r, r + 1, dtype=dtype)
    w = cp.exp(-(x * x) / (2.0 * sigma * sigma))
    w /= w.sum()
    return cuda.to_device(w), r


@dataclass
class SiftParams:
    img_dims: tuple[int, int]

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

    sigmas: np.ndarray = None
    gss_shapes: list[tuple[int, int]] = None
    kernels: list[np.ndarray] = None
    radii: np.ndarray = None

    def __post_init__(self):
        self._max_n_oct_inplace()
        self._convert_C_dog_inplace()
        self.sigmas = self._compute_sigmas()
        self.gss_shapes = self._compute_gss_shapes()
        self.kernels, self.radii = self._precompute_gauss_kernels()
        self.seed_kernel, self.seed_radius = _gauss_kernel(
            sqrt(self.sigma_min**2 - self.sigma_in**2) / self.delta_min
        )

    def _convert_C_dog_inplace(self):
        k_nspo = np.exp(np.log(2) / self.n_spo)
        k_3 = np.exp(np.log(2) / 3.0)
        self.C_dog = (k_nspo - 1) / (k_3 - 1) * self.C_dog

    def _max_n_oct_inplace(self):
        max_n_oct = floor(log2(min(self.img_dims) / self.delta_min / 12)) + 1
        self.n_oct = max_n_oct if self.n_oct == -1 else min(max_n_oct, self.n_oct)

    def _compute_sigmas(self):
        num_scales = self.n_spo + 3
        o = np.arange(self.n_oct, dtype=np.float32)[:, None]
        s = np.arange(num_scales, dtype=np.float32)[None, :]
        sigmas = self.sigma_min * np.power(2.0, o + s / self.n_spo)

        return sigmas

    def _compute_gss_shapes(self):
        M = int(round((self.img_dims[0] / self.delta_min)))
        N = int(round((self.img_dims[1] / self.delta_min)))

        shapes = []
        for _ in range(self.n_oct):
            shapes.append((M, N))
            M, N = M // 2, N // 2

        return shapes

    def _precompute_gauss_kernels(self):
        num_scales = self.n_spo + 3
        radii = np.zeros((self.n_oct, num_scales), dtype=np.int32)
        kernels = []

        for o in range(self.n_oct):
            for s in range(1, num_scales):
                sigma_inc = float(
                    sqrt(self.sigmas[o, s] ** 2 - self.sigmas[o, s - 1] ** 2)
                    / (self.delta_min * (1 << o))
                )
                k, r = _gauss_kernel(sigma_inc)
                kernels.append(k)
                radii[o, s] = r

        return kernels, radii


@dataclass
class Extrema:
    # o, s, y, x
    int_buffer: DeviceNDArray
    # y, x, sigma, dog
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
    scratch: list[DeviceNDArray]
    gss: list[DeviceNDArray]
    dog: list[DeviceNDArray]
    grad_mag: list[DeviceNDArray]
    grad_ang: list[DeviceNDArray]
    extrema: Extrema
    keypoints: Keypoints
    keypoints_host: KeypointsHost


def create_extrema(params):
    return Extrema(
        float_buffer=cuda.device_array((params.max_extrema, 4), dtype=np.float32),
        int_buffer=cuda.device_array((params.max_extrema, 4), dtype=np.int32),
    )


def create_keypoints(params):
    return Keypoints(
        positions=cuda.device_array((params.max_keypoints, 2), dtype=np.float32),
        descriptors=cuda.device_array((params.max_keypoints, 128), dtype=np.uint8),
        scales=cuda.device_array(params.max_keypoints, dtype=np.float32),
        orientations=cuda.device_array(params.max_keypoints, dtype=np.float32),
        osl=cuda.device_array((params.max_keypoints, 2), dtype=np.int32),
    )


def create_keypoints_host(params):
    positions_nbytes = params.max_keypoints * 2 * np.dtype(np.float32).itemsize
    descriptors_nbytes = params.max_keypoints * 128 * np.dtype(np.uint8).itemsize
    scales_nbytes = params.max_keypoints * np.dtype(np.float32).itemsize
    orientations_nbytes = params.max_keypoints * np.dtype(np.float32).itemsize
    osl_nbytes = params.max_keypoints * 2 * np.dtype(np.int32).itemsize

    total_bytes = (
        positions_nbytes
        + descriptors_nbytes
        + scales_nbytes
        + orientations_nbytes
        + osl_nbytes
    )
    buffer = cp.cuda.alloc_pinned_memory(total_bytes)

    offset = 0
    positions = np.frombuffer(
        buffer, dtype=np.float32, count=params.max_keypoints * 2, offset=offset
    ).reshape(params.max_keypoints, 2)
    offset += positions_nbytes

    descriptors = np.frombuffer(
        buffer, dtype=np.uint8, count=params.max_keypoints * 128, offset=offset
    ).reshape(params.max_keypoints, 128)
    offset += descriptors_nbytes

    scales = np.frombuffer(
        buffer, dtype=np.float32, count=params.max_keypoints, offset=offset
    )
    offset += scales_nbytes

    orientations = np.frombuffer(
        buffer, dtype=np.float32, count=params.max_keypoints, offset=offset
    )
    offset += orientations_nbytes

    osl = np.frombuffer(
        buffer, dtype=np.int32, count=params.max_keypoints * 2, offset=offset
    ).reshape(params.max_keypoints, 2)

    return KeypointsHost(
        positions=positions,
        descriptors=descriptors,
        scales=scales,
        orientations=orientations,
        osl=osl,
    )


def _alloc_octave_tensors(params: SiftParams, o: int):
    M, N = params.gss_shapes[o]
    n_g = params.n_spo + 3
    n_d = params.n_spo + 2
    gss = cuda.device_array((n_g, M, N), np.float32)
    dog = cuda.device_array((n_d, M, N), np.float32)
    scratch = cuda.device_array((M, N), np.float32)
    grad_mag = cuda.device_array((n_g, M, N), np.float32)
    grad_ang = cuda.device_array((n_g, M, N), np.float32)
    return gss, dog, scratch, grad_mag, grad_ang


def create_sift_data(params: SiftParams) -> SiftData:
    gss, dog, scratch, grad_mag, grad_ang = zip(
        *(_alloc_octave_tensors(params, o) for o in range(params.n_oct))
    )

    M0, N0 = params.gss_shapes[0]
    return SiftData(
        input_img=cuda.device_array(params.img_dims, np.float32),
        seed_img=cuda.device_array((M0, N0), np.float32),
        scratch=scratch,
        gss=gss,
        dog=dog,
        grad_mag=grad_mag,
        grad_ang=grad_ang,
        extrema=create_extrema(params),
        keypoints=create_keypoints(params),
        keypoints_host=create_keypoints_host(params),
    )


@cuda.jit(cache=True, fastmath=True)
def gradient_kernel(img, mag, ang):
    x, y = cuda.grid(2)
    H, W = img.shape
    if 0 < x < W - 1 and 0 < y < H - 1:
        gx = 0.5 * (img[y, x + 1] - img[y, x - 1])
        gy = 0.5 * (img[y + 1, x] - img[y - 1, x])
        mag[y, x] = ld.sqrtf(gx * gx + gy * gy)
        ang[y, x] = wrap_angle(ld.atan2f(gy, gx))


@cuda.jit(device=True, inline=True, cache=True)
def mirror(idx, size):
    period = size << 1
    i_mod = ((idx % period) + period) % period
    return ld.min(i_mod, period - 1 - i_mod)


@cuda.jit
def gauss_h(src, dst, w, radius):
    tile = cuda.shared.array(shape=0, dtype=numba.float32)
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    h, w_in = src.shape

    block_size = cuda.blockDim.x
    tile_w = block_size + 2 * radius

    base_x = cuda.blockIdx.x * block_size - radius
    for i in range(tx, tile_w, block_size):
        load_x = base_x + i
        tile[i] = src[y, mirror(load_x, w_in)]

    cuda.syncthreads()

    if x < w_in and y < h:
        acc = numba.float32(0.0)
        for k in range(-radius, radius + 1):
            acc += tile[tx + radius + k] * w[k + radius]
        dst[y, x] = acc


@cuda.jit
def gauss_v(src, dst, w, radius):
    tile = cuda.shared.array(shape=0, dtype=numba.float32)
    x, y = cuda.grid(2)
    ty = cuda.threadIdx.y
    h_in, w_in = src.shape

    block_size = cuda.blockDim.y
    tile_h = block_size + 2 * radius

    base_y = cuda.blockIdx.y * block_size - radius
    for i in range(ty, tile_h, block_size):
        load_y = base_y + i
        tile[i] = src[mirror(load_y, h_in), x]

    cuda.syncthreads()

    if x < w_in and y < h_in:
        acc = numba.float32(0.0)
        for k in range(-radius, radius + 1):
            acc += tile[ty + radius + k] * w[k + radius]
        dst[y, x] = acc


def gaussian_blur(img_in, img_out, scratch, weights, radius, stream):
    threads = 128
    h_block_dim = (threads,)
    h_grid_dim = (
        (img_in.shape[1] + h_block_dim[0] - 1) // h_block_dim[0],
        img_in.shape[0],
    )
    h_shared_mem = (h_block_dim[0] + 2 * radius) * 4
    gauss_h[h_grid_dim, h_block_dim, stream, h_shared_mem](
        img_in, scratch, weights, radius
    )

    v_block_dim = (1, threads)
    v_grid_dim = (
        img_in.shape[1],
        (img_in.shape[0] + v_block_dim[1] - 1) // v_block_dim[1],
    )
    v_shared_mem = (v_block_dim[1] + 2 * radius) * 4
    gauss_v[v_grid_dim, v_block_dim, stream, v_shared_mem](
        scratch, img_out, weights, radius
    )


def launch_gradient_kernel(src, dst_mag, dst_ang, stream):
    H, W = src.shape
    threads = (16, 16)  # 256 threads, *no* z-dim
    grid = ((W + 15) // 16, (H + 15) // 16)
    gradient_kernel[grid, threads, stream](src, dst_mag, dst_ang)


def compute_gss(sift_data, params, o, stream):
    gss = sift_data.gss[o]
    grad_mag = sift_data.grad_mag[o]
    grad_ang = sift_data.grad_ang[o]
    scratch = sift_data.scratch[o]

    num_scales = params.n_spo + 3
    kernels_per_o = num_scales - 1
    base_k_idx = o * kernels_per_o

    launch_gradient_kernel(gss[0], grad_mag[0], grad_ang[0], stream)

    for s in range(1, num_scales):
        k_idx = base_k_idx + (s - 1)
        w = params.kernels[k_idx]
        r = params.radii[o, s]

        gaussian_blur(gss[s - 1], gss[s], scratch, w, r, stream)

        launch_gradient_kernel(gss[s], grad_mag[s], grad_ang[s], stream)


@cuda.jit(cache=True)
def dog_diff_kernel(gss_in, dog_out):
    s_base = cuda.blockIdx.z * cuda.blockDim.z
    s_local = cuda.threadIdx.z
    s = s_base + s_local

    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    ns, H, W = dog_out.shape

    if s < ns and y < H and x < W:
        dog_out[s, y, x] = gss_in[s + 1, y, x] - gss_in[s, y, x]


def compute_dog(sift_data, params, o, stream):
    gss = sift_data.gss[o]
    dog = sift_data.dog[o]

    M, N = params.gss_shapes[o]

    num_dog_images = params.n_spo + 2
    threads = (4, 16, 4)

    grids = (
        (N + threads[2] - 1) // threads[2],  # Grid X
        (M + threads[1] - 1) // threads[1],  # Grid Y
        (num_dog_images + threads[0] - 1) // threads[0],  # Grid Z
    )

    dog_diff_kernel[grids, threads, stream](gss, dog)


@cuda.jit(cache=True)
def upscale_kernel(src, dst, inv_y, inv_x):
    x_out, y_out = cuda.grid(2)
    h_out, w_out = dst.shape
    if x_out >= w_out or y_out >= h_out:
        return
    y_in_f = y_out * inv_y
    x_in_f = x_out * inv_x
    y0 = int(ld.floorf(y_in_f))
    x0 = int(ld.floorf(x_in_f))
    wy = numba.float32(y_in_f - y0)
    wx = numba.float32(x_in_f - x0)
    h_in, w_in = src.shape
    y0c = mirror(y0, h_in)
    y1c = mirror(y0 + 1, h_in)
    x0c = mirror(x0, w_in)
    x1c = mirror(x0 + 1, w_in)
    v00 = src[y0c, x0c]
    v01 = src[y0c, x1c]
    v10 = src[y1c, x0c]
    v11 = src[y1c, x1c]
    dst[y_out, x_out] = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * (
        (1 - wx) * v10 + wx * v11
    )


def upscale(src, dst, stream):
    h_in, w_in = src.shape
    h_out, w_out = dst.shape
    threads = (16, 16)
    grid = (
        (w_out + threads[1] - 1) // threads[1],
        (h_out + threads[0] - 1) // threads[0],
    )
    upscale_kernel[grid, threads, stream](src, dst, h_in / h_out, w_in / w_out)


def set_seed(sift_data, params, stream):
    assert params.sigma_min >= params.sigma_in

    upscale(sift_data.input_img, sift_data.seed_img, stream)

    gaussian_blur(
        sift_data.seed_img,
        sift_data.gss[0][0],
        sift_data.scratch[0],
        params.seed_kernel,
        params.seed_radius,
        stream,
    )


@cuda.jit(cache=True)
def downsample_kernel(src, dst):
    y, x = cuda.grid(2)
    h_out, w_out = dst.shape
    if y < h_out and x < w_out:
        dst[y, x] = src[y * 2, x * 2]


def set_first_scale(sift_data, params, o, stream):
    src = sift_data.gss[o - 1][params.n_spo]
    dst = sift_data.gss[o][0]

    M, N = params.gss_shapes[o]

    threads = (16, 16)
    grid = ((N + threads[1] - 1) // threads[1], (M + threads[0] - 1) // threads[0])
    downsample_kernel[grid, threads, stream](src, dst)


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

    num_s, H, W = dog_oct.shape

    if s <= 0 or s >= num_s - 1 or y <= 0 or y >= H - 1 or x <= 0 or x >= W - 1:
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

                if n > v:
                    is_max = False
                if n < v:
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


def detect_extrema(sift_data, params, o, stream):
    cp.cuda.runtime.memsetAsync(
        int(sift_data.extrema.counter.device_ctypes_pointer.value),
        0,
        sift_data.extrema.counter.nbytes,
        stream.handle.value,
    )

    dog_o = sift_data.dog[o]
    M, N = params.gss_shapes[o]

    threads = (2, 8, 8)
    blocks = (
        (params.n_spo + 2 + threads[0] - 1) // threads[0],
        (M + threads[1] - 1) // threads[1],
        (N + threads[2] - 1) // threads[2],
    )

    find_and_record_extrema_kernel[blocks, threads, stream](
        dog_o,
        o,
        sift_data.extrema.int_buffer,
        sift_data.extrema.float_buffer,
        sift_data.extrema.counter,
        params.max_extrema,
        params.sigma_min,
        params.n_spo,
        params.delta_min,
    )


@cuda.jit(device=True, inline=True, cache=True)
def invert_3x3(H, H_inv):
    det = (
        H[0, 0] * (H[1, 1] * H[2, 2] - H[2, 1] * H[1, 2])
        - H[0, 1] * (H[1, 0] * H[2, 2] - H[1, 2] * H[2, 0])
        + H[0, 2] * (H[1, 0] * H[2, 1] - H[1, 1] * H[2, 0])
    )
    if ld.fabsf(det) < 1e-12:
        return False
    inv_det = 1.0 / det
    H_inv[0, 0] = (H[1, 1] * H[2, 2] - H[2, 1] * H[1, 2]) * inv_det
    H_inv[0, 1] = (H[0, 2] * H[2, 1] - H[0, 1] * H[2, 2]) * inv_det
    H_inv[0, 2] = (H[0, 1] * H[1, 2] - H[0, 2] * H[1, 1]) * inv_det
    H_inv[1, 0] = (H[1, 2] * H[2, 0] - H[1, 0] * H[2, 2]) * inv_det
    H_inv[1, 1] = (H[0, 0] * H[2, 2] - H[0, 2] * H[2, 0]) * inv_det
    H_inv[1, 2] = (H[1, 0] * H[0, 2] - H[0, 0] * H[1, 2]) * inv_det
    H_inv[2, 0] = (H[1, 0] * H[2, 1] - H[2, 0] * H[1, 1]) * inv_det
    H_inv[2, 1] = (H[2, 0] * H[0, 1] - H[0, 0] * H[2, 1]) * inv_det
    H_inv[2, 2] = (H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1]) * inv_det
    return True


@cuda.jit(device=True, inline=True, cache=True)
def mat_vec_mul_3x1(M, v, out):
    out[0] = M[0, 0] * v[0] + M[0, 1] * v[1] + M[0, 2] * v[2]
    out[1] = M[1, 0] * v[0] + M[1, 1] * v[1] + M[1, 2] * v[2]
    out[2] = M[2, 0] * v[0] + M[2, 1] * v[1] + M[2, 2] * v[2]


@cuda.jit(cache=True)
def refine_filter_kernel(
    dog_octave,
    int_buffer,
    float_buffer,
    extrema_count,
    C_dog,
    C_edge,
    n_spo,
    sigma_min,
    delta_min,
):
    idx = cuda.grid(1)
    if idx >= extrema_count[0]:
        return

    o = int_buffer[idx, 0]
    if o == -1:
        return

    s = int_buffer[idx, 1]
    y = int_buffer[idx, 2]
    x = int_buffer[idx, 3]

    num_s, H, W = dog_octave.shape

    g = cuda.local.array(3, dtype=numba.float32)
    Hm = cuda.local.array((3, 3), dtype=numba.float32)
    Hin = cuda.local.array((3, 3), dtype=numba.float32)
    off = cuda.local.array(3, dtype=numba.float32)

    valid = False
    for _ in range(5):
        if not (1 <= s < num_s - 1 and 1 <= y < H - 1 and 1 <= x < W - 1):
            int_buffer[idx, 0] = -1  # mark as rejected so later kernels skip it
            return

        g[0] = 0.5 * (dog_octave[s + 1, y, x] - dog_octave[s - 1, y, x])
        g[1] = 0.5 * (dog_octave[s, y + 1, x] - dog_octave[s, y - 1, x])
        g[2] = 0.5 * (dog_octave[s, y, x + 1] - dog_octave[s, y, x - 1])

        Hm[0, 0] = (
            dog_octave[s + 1, y, x] + dog_octave[s - 1, y, x] - 2 * dog_octave[s, y, x]
        )
        Hm[1, 1] = (
            dog_octave[s, y + 1, x] + dog_octave[s, y - 1, x] - 2 * dog_octave[s, y, x]
        )
        Hm[2, 2] = (
            dog_octave[s, y, x + 1] + dog_octave[s, y, x - 1] - 2 * dog_octave[s, y, x]
        )
        Hm[0, 1] = Hm[1, 0] = 0.25 * (
            dog_octave[s + 1, y + 1, x]
            - dog_octave[s + 1, y - 1, x]
            - dog_octave[s - 1, y + 1, x]
            + dog_octave[s - 1, y - 1, x]
        )
        Hm[0, 2] = Hm[2, 0] = 0.25 * (
            dog_octave[s + 1, y, x + 1]
            - dog_octave[s + 1, y, x - 1]
            - dog_octave[s - 1, y, x + 1]
            + dog_octave[s - 1, y, x - 1]
        )
        Hm[1, 2] = Hm[2, 1] = 0.25 * (
            dog_octave[s, y + 1, x + 1]
            - dog_octave[s, y + 1, x - 1]
            - dog_octave[s, y - 1, x + 1]
            + dog_octave[s, y - 1, x - 1]
        )

        if not invert_3x3(Hm, Hin):
            break

        mat_vec_mul_3x1(Hin, g, off)
        off[0] = -off[0]
        off[1] = -off[1]
        off[2] = -off[2]

        if ld.fabsf(off[0]) < 0.6 and ld.fabsf(off[1]) < 0.6 and ld.fabsf(off[2]) < 0.6:
            valid = True
            break

        s += (off[0] > 0.6) - (off[0] < -0.6)
        y += (off[1] > 0.6) - (off[1] < -0.6)
        x += (off[2] > 0.6) - (off[2] < -0.6)

        if not (1 <= s < num_s - 1 and 1 <= y < H - 1 and 1 <= x < W - 1):
            break

    if not valid:
        int_buffer[idx, 0] = -1
        return

    D_hat = dog_octave[s, y, x] + 0.5 * (g[0] * off[0] + g[1] * off[1] + g[2] * off[2])
    if ld.fabsf(D_hat) < C_dog:
        int_buffer[idx, 0] = -1
        return

    Hxx = Hm[1, 1]
    Hyy = Hm[2, 2]
    Hxy = Hm[1, 2]
    det = Hxx * Hyy - Hxy * Hxy
    if det <= 0:
        int_buffer[idx, 0] = -1
        return
    trace = Hxx + Hyy
    r = C_edge
    if (trace * trace) / det > ((r + 1) * (r + 1) / r):
        int_buffer[idx, 0] = -1
        return

    int_buffer[idx, 1] = s
    int_buffer[idx, 2] = y
    int_buffer[idx, 3] = x

    scale = delta_min * (1 << o)
    float_buffer[idx, 0] = (y + off[1]) * scale
    float_buffer[idx, 1] = (x + off[2]) * scale
    float_buffer[idx, 2] = sigma_min * (2.0 ** (o + (s + off[0]) / n_spo))
    float_buffer[idx, 3] = D_hat


def refine_and_filter(sift_data, params, o, stream):
    dog_oct = sift_data.dog[o]

    threads = 128
    blocks = (params.max_extrema + threads - 1) // threads

    refine_filter_kernel[blocks, threads, stream](
        dog_oct,
        sift_data.extrema.int_buffer,
        sift_data.extrema.float_buffer,
        sift_data.extrema.counter,
        params.C_dog,
        params.C_edge,
        params.n_spo,
        params.sigma_min,
        params.delta_min,
    )


TWO_PI = numba.float32(6.28318530718)


@cuda.jit(device=True, inline=True, cache=True)
def wrap_angle(theta: numba.float32) -> numba.float32:
    return ld.fmodf(ld.fmodf(theta, TWO_PI) + TWO_PI, TWO_PI)


TX, TY = 16, 16  # 256 threads / block


@cuda.jit(cache=True, fastmath=True)
def orientation_kernel_blockperkp(
    grad_mag,
    grad_ang,  # ← reuse gradients
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
    kp_idx = cuda.blockIdx.x  # one block ↔ one keypoint
    if kp_idx >= n_extrema[0]:
        return
    if int_buf[kp_idx, 0] != oct_idx:
        return

    # ------------- constants for this keypoint -------------
    s = int_buf[kp_idx, 1]
    yy0 = int_buf[kp_idx, 2]
    xx0 = int_buf[kp_idx, 3]

    sigma_w = float_buf[kp_idx, 2]
    scale = delta_min * (1 << oct_idx)
    sigma_oct = sigma_w / scale

    radius = int(3.0 * lambda_ori * sigma_oct + 0.5)
    radius = 1 if radius == 0 else radius

    H, W = grad_mag.shape[1:]
    if (
        yy0 - radius < 1
        or yy0 + radius >= H - 1
        or xx0 - radius < 1
        or xx0 + radius >= W - 1
    ):
        return

    g_sigma = lambda_ori * sigma_oct
    inv_2sig2 = 1.0 / (2.0 * g_sigma * g_sigma)
    bin_scale = numba.float32(36.0 / 6.28318530718)  # 36 / 2π

    # ------------- shared 36-bin histogram -------------
    hist = cuda.shared.array(36, numba.float32)
    tflat = cuda.threadIdx.y * TX + cuda.threadIdx.x
    if tflat < 36:
        hist[tflat] = 0.0
    cuda.syncthreads()

    # ------------- sweep patch -------------
    for dy in range(cuda.threadIdx.y, radius * 2 + 1, TY):
        yy = yy0 - radius + dy
        dyf = yy - yy0
        for dx in range(cuda.threadIdx.x, radius * 2 + 1, TX):
            xx = xx0 - radius + dx
            dxf = xx - xx0

            mag = grad_mag[s, yy, xx]
            if mag == 0.0:
                continue

            ang = wrap_angle(grad_ang[s, yy, xx])
            w = mag * ld.expf(-(dxf * dxf + dyf * dyf) * inv_2sig2)
            bin_idx = int(ang * bin_scale + 0.5) % 36
            cuda.atomic.add(hist, bin_idx, w)

    cuda.syncthreads()

    # ------------- smoothing + peak-picking -------------
    if tflat == 0:
        tmp = cuda.local.array(36, numba.float32)
        for _ in range(6):  # 6× circular box-smooth
            for i in range(36):
                tmp[i] = hist[i]
            for i in range(36):
                hist[i] = (tmp[(i - 1) % 36] + tmp[i] + tmp[(i + 1) % 36]) / 3.0

        # max + threshold
        vmax = numba.float32(0.0)
        for i in range(36):
            vmax = vmax if vmax > hist[i] else hist[i]
        if vmax == 0.0:
            return
        thresh = 0.8 * vmax
        eps = 1e-9 * vmax

        for i in range(36):
            prev, cur, nxt = hist[(i - 1) % 36], hist[i], hist[(i + 1) % 36]
            if cur < thresh or cur <= prev or cur <= nxt:
                continue
            denom = prev - 2.0 * cur + nxt
            if ld.fabsf(denom) < eps:
                continue
            offset = (prev - nxt) / (2.0 * denom)
            offset = 0.0 if ld.fabsf(offset) > 1.0 else offset
            theta = wrap_angle((i + offset) * (6.28318530718 / 36.0))

            out_idx = cuda.atomic.add(kp_counter, 0, 1)
            if out_idx >= key_pos.shape[0]:
                cuda.atomic.add(kp_counter, 0, -1)
                cuda.atomic.add(kp_counter, 1, 1)
                return

            key_pos[out_idx, 0] = float_buf[kp_idx, 1]  # x
            key_pos[out_idx, 1] = float_buf[kp_idx, 0]  # y
            key_scale[out_idx] = sigma_w
            key_ori[out_idx] = theta
            key_osl[out_idx, 0] = oct_idx
            key_osl[out_idx, 1] = s


def assign_orientations(sift_data, params, o, stream):
    blocks = (params.max_extrema,)  # one block per candidate
    threads = (TX, TY)

    orientation_kernel_blockperkp[blocks, threads, stream](
        sift_data.grad_mag[o],  # grad magnitude
        sift_data.grad_ang[o],  # grad angle
        sift_data.extrema.int_buffer,
        sift_data.extrema.float_buffer,
        sift_data.extrema.counter,
        sift_data.keypoints.positions,
        sift_data.keypoints.scales,
        sift_data.keypoints.orientations,
        sift_data.keypoints.osl,
        sift_data.keypoints.counter,
        o,
        params.lambda_ori,
        params.delta_min,
    )


LAMBDA_DESC = numba.float32(6.0)
NHIST = 4
NORIBIN = 8
NHIST2 = NHIST * NHIST
DESC_LEN = NHIST2 * NORIBIN


TX, TY = 16, 16  # 256 threads / block  → good occupancy


@cuda.jit(cache=True, fastmath=True)
def descriptor_kernel_blockperkp(
    grad_mag,
    grad_ang,
    kpos,
    kscale,
    kori,
    osl,
    kctr,  # same device counter array
    desc,
    oct_idx,
    delta_min,
):
    kp_idx = cuda.blockIdx.x  # one block == one keypoint
    if kp_idx >= kctr[0]:
        return
    if osl[kp_idx, 0] != oct_idx:
        return

    # ---------- per-keypoint constants ----------
    s = osl[kp_idx, 1]
    x_w, y_w = kpos[kp_idx]
    sigma = kscale[kp_idx]
    theta0 = kori[kp_idx]

    scale = delta_min * (1 << oct_idx)
    x0, y0 = x_w / scale, y_w / scale  # octave coords

    radiusF = LAMBDA_DESC * sigma / scale
    radius = int(radiusF + 0.5)
    radius = 1 if radius < 1 else radius
    dmin = ld.ceilf(ld.sqrtf(2.0) * radius)

    H, W = grad_mag.shape[1:]
    if x0 < dmin or x0 > W - 1 - dmin or y0 < dmin or y0 > H - 1 - dmin:
        return

    cos_t, sin_t = ld.cosf(theta0), ld.sinf(theta0)
    inv_2sig2 = 1.0 / (2.0 * radiusF * radiusF)
    bin_scale = NORIBIN / 6.28318530718  # 2π
    patch_size = radius * 2 + 1  # side length in px

    # ---------- shared histogram ----------
    hist = cuda.shared.array(DESC_LEN, numba.float32)

    # zero the shared array (first 128 threads of the block)
    tflat = cuda.threadIdx.y * TX + cuda.threadIdx.x
    if tflat < DESC_LEN:
        hist[tflat] = 0.0
    cuda.syncthreads()

    # ---------- sweep the rotated patch ----------
    # Each thread walks over the square in strides of blockDim.xy
    for py in range(cuda.threadIdx.y, patch_size, TY):
        yy = int(y0) - radius + py
        dy0 = yy - y0
        for px in range(cuda.threadIdx.x, patch_size, TX):
            xx = int(x0) - radius + px
            dx0 = xx - x0

            # rotate into keypoint frame
            dx = dx0 * cos_t + dy0 * sin_t
            dy = -dx0 * sin_t + dy0 * cos_t
            u = dx / (radius * 2) * NHIST + (NHIST * 0.5 - 0.5)
            v = dy / (radius * 2) * NHIST + (NHIST * 0.5 - 0.5)
            if u < -1 or u > NHIST or v < -1 or v > NHIST:
                continue

            mag = grad_mag[s, yy, xx]
            if mag == 0.0:
                continue

            ang = wrap_angle(grad_ang[s, yy, xx] - theta0)
            o_bin = ang * bin_scale
            u0 = int(ld.floorf(u))
            du = u - u0
            v0 = int(ld.floorf(v))
            dv = v - v0
            o0 = int(ld.floorf(o_bin))
            do = o_bin - o0
            w_base = mag * ld.expf(-(dx * dx + dy * dy) * inv_2sig2)

            # trilinear vote, atomics into shared memory
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
                                hidx = ((vv * NHIST + uu) * NORIBIN) + oo
                                cuda.atomic.add(hist, hidx, w_base * wu * wv * wo)

    cuda.syncthreads()

    # ---------- normalise, clamp, quantise ----------
    if tflat == 0:  # single thread does the maths
        l2 = numba.float32(0.0)
        for i in range(DESC_LEN):
            l2 += hist[i] * hist[i]
        l2 = ld.rsqrtf(l2 + 1e-7)

        l2p = numba.float32(0.0)
        for i in range(DESC_LEN):
            v = hist[i] * l2
            if v > 0.2:
                v = 0.2
            hist[i] = v
            l2p += v * v
        l2p = ld.rsqrtf(l2p + 1e-7)

        for i in range(DESC_LEN):
            q = hist[i] * l2p * 512.0
            desc[kp_idx, i] = numba.uint8(255 if q > 255 else int(q + 0.5))


def build_descriptors(sift_data, params, o, stream):
    threads = (TX, TY)
    # launch as many blocks as the *maximum* possible keypoints; extra
    # blocks exit immediately if kp_idx ≥ counter[0]
    blocks = (params.max_keypoints,)
    descriptor_kernel_blockperkp[blocks, threads, stream](
        sift_data.grad_mag[o],
        sift_data.grad_ang[o],
        sift_data.keypoints.positions,
        sift_data.keypoints.scales,
        sift_data.keypoints.orientations,
        sift_data.keypoints.osl,
        sift_data.keypoints.counter,
        sift_data.keypoints.descriptors,
        o,
        params.delta_min,
    )


def compute(sift_data, params, stream):
    cp.cuda.runtime.memsetAsync(
        int(sift_data.keypoints.counter.device_ctypes_pointer.value),
        0,
        sift_data.keypoints.counter.nbytes,
        stream.handle.value,
    )
    cp.cuda.runtime.memsetAsync(
        int(sift_data.extrema.counter.device_ctypes_pointer.value),
        0,
        sift_data.extrema.counter.nbytes,
        stream.handle.value,
    )

    set_seed(sift_data, params, stream)

    for o in range(params.n_oct):
        compute_gss(sift_data, params, o, stream)
        compute_dog(sift_data, params, o, stream)
        detect_extrema(sift_data, params, o, stream)
        refine_and_filter(sift_data, params, o, stream)
        assign_orientations(sift_data, params, o, stream)
        build_descriptors(sift_data, params, o, stream)

        if o < params.n_oct - 1:
            set_first_scale(sift_data, params, o + 1, stream)


def check_runtime_error(err):
    if err != runtime.cudaError_t.cudaSuccess:
        err_name = runtime.cudaGetErrorName(err)[1].decode()
        err_str = runtime.cudaGetErrorString(err)[1].decode()
        raise RuntimeError(f"Runtime error: {err_name} ({err_str})")


class Sift:
    def __init__(self, params: SiftParams, use_graph: bool = True):
        self.params = params
        self.use_graph = use_graph
        self.sift_data = create_sift_data(self.params)

        self.exec_stream = cp.cuda.Stream(non_blocking=True)

        if self.use_graph:
            numba_strm = cuda.external_stream(self.exec_stream.ptr)

            with self.exec_stream:
                (err,) = runtime.cudaStreamBeginCapture(
                    self.exec_stream.ptr,
                    runtime.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal,
                )
                check_runtime_error(err)

                compute(self.sift_data, self.params, numba_strm)

                err, self.graph = runtime.cudaStreamEndCapture(self.exec_stream.ptr)
            check_runtime_error(err)
            err, self.exec_graph = runtime.cudaGraphInstantiate(self.graph, 0)
            check_runtime_error(err)

            # Warm-up compile to move JIT & cudaMalloc cost out of measured region
            with self.exec_stream:
                compute(self.sift_data, self.params, numba_strm)
            cp.cuda.get_current_stream().synchronize()
        else:
            self.numba_exec_stream = cuda.external_stream(self.exec_stream.ptr)

    def __del__(self):
        if runtime and self.use_graph:
            runtime.cudaGraphExecDestroy(self.exec_graph)
            runtime.cudaGraphDestroy(self.graph)

    def _get_results(self, numba_copy_stream, copy_stream):
        k = int(
            self.sift_data.keypoints.counter.copy_to_host(stream=numba_copy_stream)[0]
        )
        self.sift_data.keypoints.positions.copy_to_host(
            self.sift_data.keypoints_host.positions, stream=numba_copy_stream
        )
        self.sift_data.keypoints.descriptors.copy_to_host(
            self.sift_data.keypoints_host.descriptors, stream=numba_copy_stream
        )
        self.sift_data.keypoints.scales.copy_to_host(
            self.sift_data.keypoints_host.scales, stream=numba_copy_stream
        )
        self.sift_data.keypoints.orientations.copy_to_host(
            self.sift_data.keypoints_host.orientations, stream=numba_copy_stream
        )
        self.sift_data.keypoints.osl.copy_to_host(
            self.sift_data.keypoints_host.osl, stream=numba_copy_stream
        )
        copy_stream.synchronize()
        return truncate_keypoints(self.sift_data.keypoints_host, k)

    def process_images(self, image_paths):
        image_paths = list(image_paths)
        num_images = len(image_paths)
        if num_images == 0:
            return

        copy_stream = cp.cuda.Stream(non_blocking=True)
        numba_copy_stream = cuda.external_stream(copy_stream.ptr)
        img_bytes = (
            self.params.img_dims[0]
            * self.params.img_dims[1]
            * np.dtype(np.float32).itemsize
        )
        h_bufs = [cp.cuda.alloc_pinned_memory(img_bytes) for _ in range(2)]
        events = [cp.cuda.Event() for _ in range(2)]

        # First image
        host_img = read_img(image_paths[0]).astype(np.float32)
        np.asarray(h_bufs[0]).view(np.float32)[: host_img.size] = host_img.ravel()
        cp.cuda.runtime.memcpyAsync(
            self.sift_data.input_img.device_ctypes_pointer.value,
            h_bufs[0].ptr,
            host_img.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            copy_stream.ptr,
        )
        copy_stream.record(events[0])
        self.exec_stream.wait_event(events[0])
        if self.use_graph:
            runtime.cudaGraphLaunch(self.exec_graph, self.exec_stream.ptr)
        else:
            compute(self.sift_data, self.params, self.numba_exec_stream)
        self.exec_stream.record(events[0])

        for i in range(1, num_images):
            current_buffer_idx = i % 2
            prev_buffer_idx = (i - 1) % 2

            next_host_img = read_img(image_paths[i]).astype(np.float32)

            # Remove explicit sync - let kernels of img i run while we copy img i+1
            # events[prev_buffer_idx].synchronize()

            # *** NEW *** - make sure all work for image i-1 is finished on the GPU
            copy_stream.wait_event(events[prev_buffer_idx])

            yield self._get_results(numba_copy_stream, copy_stream)

            np.asarray(h_bufs[current_buffer_idx]).view(np.float32)[
                : next_host_img.size
            ] = next_host_img.ravel()
            cp.cuda.runtime.memcpyAsync(
                self.sift_data.input_img.device_ctypes_pointer.value,
                h_bufs[current_buffer_idx].ptr,
                next_host_img.nbytes,
                runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                copy_stream.ptr,
            )
            copy_stream.record(events[current_buffer_idx])
            self.exec_stream.wait_event(events[current_buffer_idx])
            if self.use_graph:
                runtime.cudaGraphLaunch(self.exec_graph, self.exec_stream.ptr)
            else:
                compute(self.sift_data, self.params, self.numba_exec_stream)
            self.exec_stream.record(events[current_buffer_idx])

        # Final image
        final_buffer_idx = (num_images - 1) % 2
        events[final_buffer_idx].synchronize()
        yield self._get_results(numba_copy_stream, copy_stream)


def truncate_keypoints(kp_host: KeypointsHost, num_keypoints: int):
    return KeypointsHost(
        positions=kp_host.positions[:num_keypoints],
        descriptors=kp_host.descriptors[:num_keypoints],
        scales=kp_host.scales[:num_keypoints],
        orientations=kp_host.orientations[:num_keypoints],
        osl=kp_host.osl[:num_keypoints],
    )


if __name__ == "__main__":
    root = "data/oxford_affine/graf"
    image_paths = [f"{root}/img{i}.png" for i in range(1, 7)]

    # --- 1. Initial Setup ---
    first_img_for_setup = read_img(image_paths[0])
    params = SiftParams(img_dims=first_img_for_setup.shape)
    sift = Sift(params, use_graph=True)  # Re-enable CUDA graphs

    for i, keypoints in enumerate(sift.process_images(image_paths)):
        print(f"{image_paths[i]}: {len(keypoints.positions)} keypoints")
