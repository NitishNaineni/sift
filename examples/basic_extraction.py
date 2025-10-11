"""
Basic SIFT feature extraction example.

This script demonstrates the simplest usage of CUDA SIFT
to extract features from a single image.
"""

import sys
from pathlib import Path

import cudasift


def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python basic_extraction.py <image_path>")
        print("Example: python basic_extraction.py ../data/oxford_affine/graf/img1.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Load image to get dimensions
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    height, width = img.shape
    print(f"Image dimensions: {height}x{width}")

    # Create SIFT detector
    print("Initializing SIFT detector...")
    detector = cudasift.SiftDetector(img_dims=(height, width))

    # Detect features
    print("Detecting SIFT features...")
    keypoints, descriptors = detector.detect(str(image_path))

    # Display results
    print(f"\n✓ Found {len(keypoints)} keypoints")
    print(f"  Keypoints shape: {keypoints.shape}")
    print(f"  Descriptors shape: {descriptors.shape}")

    # Show some example keypoints
    if len(keypoints) > 0:
        print("\nFirst 5 keypoints:")
        print(f"  {'X':>8} {'Y':>8} {'Scale':>8} {'Orientation':>12}")
        print(f"  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12}")
        for i in range(min(5, len(keypoints))):
            x, y, scale, ori = keypoints[i]
            print(f"  {x:8.2f} {y:8.2f} {scale:8.3f} {ori:12.3f}")


if __name__ == "__main__":
    main()
