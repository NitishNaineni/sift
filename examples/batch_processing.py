"""
Batch SIFT feature extraction example.

This script demonstrates how to efficiently process multiple images.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

import cudasift


def main():
    # Check if directory path is provided
    if len(sys.argv) < 2:
        print("Usage: python batch_processing.py <image_directory>")
        print("Example: python batch_processing.py ../data/oxford_affine/graf/")
        sys.exit(1)

    img_dir = Path(sys.argv[1])

    if not img_dir.exists() or not img_dir.is_dir():
        print(f"Error: Directory not found: {img_dir}")
        sys.exit(1)

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions]

    if not image_paths:
        print(f"Error: No images found in {img_dir}")
        sys.exit(1)

    image_paths.sort()
    print(f"Found {len(image_paths)} images")

    # Get dimensions from first image
    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print(f"Error: Could not load image: {image_paths[0]}")
        sys.exit(1)

    height, width = first_img.shape
    print(f"Image dimensions: {height}x{width}")

    # Create SIFT detector
    print("Initializing SIFT detector...")
    detector = cudasift.SiftDetector(img_dims=(height, width))

    # Process images
    print(f"\nProcessing {len(image_paths)} images...")
    start_time = time.time()

    results = []
    for i, img_path in enumerate(image_paths, 1):
        # Check dimensions
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img.shape != (height, width):
            print(f"Skipping {img_path.name}: wrong dimensions {img.shape}")
            continue

        keypoints, descriptors = detector.detect(str(img_path))
        results.append((img_path.name, len(keypoints)))

        if i % 10 == 0 or i == len(image_paths):
            print(f"  Processed {i}/{len(image_paths)} images...")

    elapsed = time.time() - start_time

    # Display results
    print(f"\n✓ Processed {len(results)} images in {elapsed:.2f}s")
    print(f"  Average: {elapsed / len(results):.3f}s per image")
    print(f"  Throughput: {len(results) / elapsed:.1f} fps")

    print("\nKeypoints per image:")
    for name, num_kpts in results:
        print(f"  {name:30s} {num_kpts:6d} keypoints")

    # Statistics
    keypoint_counts = [count for _, count in results]
    print("\nStatistics:")
    print(f"  Min keypoints:  {min(keypoint_counts)}")
    print(f"  Max keypoints:  {max(keypoint_counts)}")
    print(f"  Mean keypoints: {np.mean(keypoint_counts):.1f}")
    print(f"  Std keypoints:  {np.std(keypoint_counts):.1f}")


if __name__ == "__main__":
    main()
