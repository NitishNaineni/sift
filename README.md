# CUDA SIFT

A high-performance SIFT (Scale-Invariant Feature Transform) implementation using custom CUDA kernels written in Python via Numba. Achieves C-level performance with zero external CUDA dependencies.

An initial prototype was built in WebGPU with Python bindings before being rewritten in Numba CUDA for performance.

## What this does

Detects and describes keypoints in images — the full SIFT pipeline:

1. Gaussian scale-space construction (separable blur with shared memory tiling)
2. Difference of Gaussians (DoG)
3. Extrema detection with sub-pixel refinement
4. Orientation assignment
5. 128-dimensional descriptor extraction

## GPU techniques used

- **Shared memory tiling with halo regions** — Gaussian blur kernels load image tiles + border pixels into shared memory for fast neighbor access (`gauss_h`, `gauss_v`, `gradient_kernel`)
- **Shared memory histograms with atomic accumulation** — Orientation and descriptor kernels build histograms in shared memory using `cuda.atomic.add`, then normalize in-place
- **Multi-stream concurrent execution** — One CUDA stream per octave with event-based synchronization. Octave N signals readiness via `cuda.event()`, octave N+1 waits before starting
- **Pinned host memory** — Host-side buffers use `cuda.pinned_array` for async device-to-host transfers
- **Dynamic shared memory allocation** — Blur kernels size their shared memory tiles at launch based on the Gaussian radius
- **Per-kernel thread block tuning** — Different block configurations for different workloads: 16x16 for spatial kernels, 256 for 1D kernels, (2,8,8) for 3D extrema search, (16,16,4) for DoG
- **Pre-allocated GPU memory** — All buffers allocated once upfront, reused across frames
- **Profiled with Nsight Systems and Nsight Compute** to identify and fix occupancy and memory throughput bottlenecks

## Correctness validation

Every intermediate output is validated against a [reference C implementation](sift_anatomy/) (from IPOL's "Anatomy of the SIFT Method"):

- Tolerance: 5×10⁻⁶ for array comparisons (Gaussian scale-space, DoG, gradients)
- Hamming distance matching for descriptors
- Set parity checks for keypoint counts at every pipeline stage
- Tested across 100+ images from the DIV2K validation dataset
- Tests automatically build and run the C reference binary for comparison

Run tests:
```
uv run python -m pytest tests/ -v
```

## Structure

```
cudasift/
  kernels.py   — All CUDA kernels (blur, gradient, extrema, orientation, descriptors)
  core.py      — Pipeline orchestration, stream management
  types.py     — GPU/host data structures, parameter computation
  utils.py     — Memory allocation, pinned buffers, image I/O
tests/
  test_sift_unittest.py — Stage-by-stage validation against C reference
sift_anatomy/
  src/         — Reference C implementation (IPOL)
```

## Usage

```python
from cudasift import SiftDetector

detector = SiftDetector(img_dims=(height, width))
keypoints, descriptors = detector.detect("image.png")

# keypoints: (N, 4) float32 — x, y, scale, orientation
# descriptors: (N, 128) uint8
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- numba, numpy, opencv-python

## Branches

- `classic` — Clean, modular implementation with full test suite
- `depth` — Experimental: depth-aware SIFT using RGB-D data with camera intrinsics and Jacobian-based keypoint normalization. Multi-stream pipeline. Results didn't match expectations.
- `review-branch` — RootSIFT descriptor variant, double-buffered pinned memory, refactored kernels
