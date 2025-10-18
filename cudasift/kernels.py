"""CUDA kernels for SIFT."""
# pyright: basic

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba
import numpy as np
from numba import cuda
from numba.cuda import libdevice as ld
from numba.cuda.cudadrv.devicearray import DeviceNDArray

if TYPE_CHECKING:
    from .types import SiftData, SiftParams

# Constants used by CUDA kernels
TX, TY = 16, 16
TWO_PI = np.float32(6.28318530718)
ORI_BINS = 36
NHIST, NORIBIN = 4, 8
NHIST2 = NHIST * NHIST
DESC_LEN = NHIST2 * NORIBIN
LAMBDA_DESC = numba.float32(6.0)
ORI_THRESHOLD = numba.float32(0.8)
BLUR_TH = 256
MAX_GAUSS_RADIUS = 16
GRAD_TILE_SIZE = (TX + 2) * (TY + 2)
GAUSS_HORZ_TILE_SIZE = BLUR_TH + 2 * MAX_GAUSS_RADIUS
GAUSS_COEFF_TILE_SIZE = MAX_GAUSS_RADIUS + 1
GAUSS_VERT_TILE_H = TY + 2 * MAX_GAUSS_RADIUS
GAUSS_VERT_TILE_SIZE = GAUSS_VERT_TILE_H * TX


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
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


@cuda.jit(cache=True)  # type: ignore[misc]
def reset_counters_kernel(ext_counter, key_counter):
    if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0:
        ext_counter[0] = 0
        ext_counter[1] = 0
        key_counter[0] = 0
        key_counter[1] = 0
        key_counter[2] = 0


@cuda.jit(device=True, inline=True, cache=True, fastmath=True)  # type: ignore[misc]
def mirror(i: int, n: int) -> int:
    if i < 0:
        i = -i - 1
    elif i >= n:
        i = (n << 1) - 1 - i
    return i


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
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


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
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


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
def downsample_kernel(src, dst):
    x, y = cuda.grid(2)
    h, w = dst.shape
    if x < w and y < h:
        dst[y, x] = src[y * 2, x * 2]


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
def dog_diff_kernel(gss_in, dog_out):
    s = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ns, h, w = dog_out.shape
    if s < ns and y < h and x < w:
        dog_out[s, y, x] = gss_in[s + 1, y, x] - gss_in[s, y, x]


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
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


@cuda.jit(device=True, inline=True, cache=True, fastmath=True)  # type: ignore[misc]
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


@cuda.jit(device=True, inline=True, cache=True, fastmath=True)  # type: ignore[misc]
def mat_vec_mul_3x1(M, v, out):
    out[0] = M[0, 0] * v[0] + M[0, 1] * v[1] + M[0, 2] * v[2]
    out[1] = M[1, 0] * v[0] + M[1, 1] * v[1] + M[1, 2] * v[2]
    out[2] = M[2, 0] * v[0] + M[2, 1] * v[1] + M[2, 2] * v[2]


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
def refine_kernel(dog_oct, int_buf, float_buf, ext_count, n_spo, sigma_min, delta_min, oct_idx):
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


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
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
    ho = int(round(hi / float(delta_min)))
    wo = int(round(wi / float(delta_min)))
    if dst.shape != (ho, wo):
        raise ValueError(
            f"dst.shape must be {(ho, wo)} for delta_min={float(delta_min)}, got {dst.shape}"
        )

    grid = ((wo + TX - 1) // TX, (ho + TY - 1) // TY)
    oversample_bilinear_kernel[grid, (TX, TY), stream](src, dst, numba.float32(delta_min))


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
    th = BLUR_TH
    if radius > MAX_GAUSS_RADIUS:
        raise ValueError(f"Gaussian radius {radius} exceeds MAX_GAUSS_RADIUS={MAX_GAUSS_RADIUS}.")
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
    threads = 256
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
    threads = 256
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


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
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
    if det <= numba.float32(0):
        int_buf[idx, 0] = -1
        return
    trace = hXX + hYY
    r = C_edge
    if (trace * trace) / det > ((r + numba.float32(1.0)) * (r + numba.float32(1.0)) / r):
        int_buf[idx, 0] = -1
        return


@cuda.jit(device=True, inline=True, cache=True, fastmath=True)  # type: ignore[misc]
def wrap_angle(theta: numba.float32) -> numba.float32:
    return ld.fmodf(ld.fmodf(theta, TWO_PI) + TWO_PI, TWO_PI)


@cuda.jit(cache=True)  # type: ignore[misc]
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


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
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
    h, w = gx.shape[1:]
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
            gxv = gx[s, yy, xx]
            gyv = gy[s, yy, xx]
            m = ld.sqrtf(gxv * gxv + gyv * gyv)
            if m == 0.0:
                continue
            a = wrap_angle(ld.atan2f(gxv, gyv))
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
                hist[i] = (tmp[(i - 1) % ORI_BINS] + tmp[i] + tmp[(i + 1) % ORI_BINS]) / 3.0
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
    threads = 256
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
    threads = 256
    blocks = (data.extrema.int_buffer.shape[0] + threads - 1) // threads
    discard_near_the_border_kernel[blocks, threads, stream](
        data.extrema.int_buffer,
        data.extrema.float_buffer,
        data.extrema.counter,
        int(octave_index),
        int(image_height),
        int(image_width),
        numba.float32(1.0),
    )
    if record:
        total = int(data.extrema.counter.copy_to_host(stream=stream)[0])
        ib_h = data.extrema.int_buffer[:total].copy_to_host(stream=stream)
        fb_h = data.extrema.float_buffer[:total].copy_to_host(stream=stream)
        mask = ib_h[:, 0] == octave_index
        return (ib_h[mask], fb_h[mask])
    return None


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
def discard_near_the_border_kernel(
    int_buf, float_buf, ext_count, oct_idx, image_h, image_w, lambda_border
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
        (y - lambda_border * sigma > numba.float32(0.0))
        and (y + lambda_border * sigma < numba.float32(image_h))
        and (x - lambda_border * sigma > numba.float32(0.0))
        and (x + lambda_border * sigma < numba.float32(image_w))
    ):
        int_buf[idx, 0] = -1


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
def descriptor_kernel(gx, gy, key_float, key_int, kctr, desc, oct_idx, delta_min):
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

    h, w = gx.shape[1:]
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
            gxv = gx[s, yy, xx]
            gyv = gy[s, yy, xx]
            m = ld.sqrtf(gxv * gxv + gyv * gyv)
            if m == 0.0:
                continue
            a = wrap_angle(ld.atan2f(gxv, gyv) - theta0)
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


@cuda.jit(cache=True, fastmath=True)  # type: ignore[misc]
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


# ============================================================================
# End of CUDA Kernels Section - Re-enable type checking
# ============================================================================
# pyright: reportGeneralTypeIssues=true, reportOptionalSubscript=true, reportOperatorIssue=true
