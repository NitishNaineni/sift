#!/usr/bin/env python3
"""
Final comprehensive benchmark comparing before and after optimizations.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from numba import cuda

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudasift


def benchmark_image(img_path, num_runs=30):
    """Benchmark SIFT on a single image."""

    if not Path(img_path).exists():
        print(f"Image not found: {img_path}")
        return None

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load: {img_path}")
        return None

    h, w = img.shape

    detector = cudasift.SiftDetector(
        img_dims=(h, w),
        max_keypoints=100_000,
        record=False,
    )

    # Warm-up
    for _ in range(3):
        detector.detect(img)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = cuda.event()
        end = cuda.event()

        start.record()
        keypoints, _ = detector.detect(img)
        end.record()
        end.synchronize()

        times.append(cuda.event_elapsed_time(start, end))

    times = np.array(times)

    return {
        "resolution": f"{h}x{w}",
        "megapixels": (h * w) / 1e6,
        "keypoints": len(keypoints),
        "mean_ms": times.mean(),
        "median_ms": np.median(times),
        "std_ms": times.std(),
        "min_ms": times.min(),
        "max_ms": times.max(),
        "throughput_kps": len(keypoints) / (times.mean() / 1000),
        "throughput_mps": ((h * w) / 1e6) / (times.mean() / 1000),
    }


def main():
    print("=" * 70)
    print("CUDA SIFT - Final Performance Benchmark")
    print("=" * 70)

    test_images = [
        "data/oxford_affine/graf/img1.png",
        "data/oxford_affine/boat/img1.png",
        "data/oxford_affine/bark/img1.png",
        "data/oxford_affine/leuven/img1.png",
    ]

    results = []

    for img_path in test_images:
        print(f"\nBenchmarking: {img_path}")
        result = benchmark_image(img_path, num_runs=30)
        if result:
            results.append(result)
            print(f"  Resolution: {result['resolution']}")
            print(f"  Keypoints: {result['keypoints']}")
            print(f"  Mean time: {result['mean_ms']:.2f} ms")
            print(
                f"  Throughput: {result['throughput_kps']:.0f} kp/s, {result['throughput_mps']:.1f} MP/s"
            )

    if results:
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(
            f"{'Image':<30} {'Resolution':<12} {'Keypoints':<10} {'Mean (ms)':<12} {'Throughput':<20}"
        )
        print("-" * 70)

        for i, r in enumerate(results):
            img_name = Path(test_images[i]).parent.name
            print(
                f"{img_name:<30} {r['resolution']:<12} {r['keypoints']:<10} {r['mean_ms']:<12.2f} {r['throughput_mps']:<8.1f} MP/s"
            )

        avg_throughput = np.mean([r["throughput_mps"] for r in results])
        print("-" * 70)
        print(f"Average throughput: {avg_throughput:.1f} MP/s")

        print("\n" + "=" * 70)
        print("OPTIMIZATIONS APPLIED:")
        print("=" * 70)
        print("  1. Changed thread blocks from 16x16 to 32x8 (warp-aligned)")
        print("  2. Added opt=True to all kernels for compiler optimizations")
        print("  3. Replaced branches with clamping (max/min)")
        print("  4. Padded shared memory arrays to avoid bank conflicts")
        print("  5. Improved memory coalescing in gradient and DoG kernels")
        print("  6. Optimized Gaussian blur coefficient loading")
        print("  7. Better loop iteration patterns for cache efficiency")


if __name__ == "__main__":
    main()
