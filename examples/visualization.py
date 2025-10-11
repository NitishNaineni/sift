"""
SIFT feature visualization example.

This script demonstrates how to visualize detected SIFT features.
Requires matplotlib for display.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

import cudasift

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ImportError:
    print("Error: This example requires matplotlib")
    print("Install with: pip install matplotlib")
    sys.exit(1)


def draw_keypoints(ax, image, keypoints, max_display=100):
    """
    Draw keypoints on an image.

    Args:
        ax: Matplotlib axes
        image: Grayscale image
        keypoints: Nx4 array of keypoints [x, y, scale, orientation]
        max_display: Maximum number of keypoints to display
    """
    ax.imshow(image, cmap="gray")

    # Limit number of keypoints for clarity
    if len(keypoints) > max_display:
        # Sample uniformly
        indices = np.linspace(0, len(keypoints) - 1, max_display, dtype=int)
        keypoints = keypoints[indices]

    for x, y, scale, orientation in keypoints:
        # Draw circle at keypoint location (size based on scale)
        circle = Circle((x, y), scale * 3, fill=False, color="red", linewidth=1)
        ax.add_patch(circle)

        # Draw orientation arrow
        arrow_len = scale * 5
        dx = arrow_len * np.cos(orientation)
        dy = arrow_len * np.sin(orientation)
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=scale * 2,
            head_length=scale * 2,
            fc="yellow",
            ec="yellow",
            linewidth=1,
            alpha=0.7,
        )

    ax.set_title(f"SIFT Features ({len(keypoints)} keypoints shown)")
    ax.axis("off")


def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <image_path>")
        print("Example: python visualization.py ../data/oxford_affine/graf/img1.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    height, width = img.shape
    print(f"Image: {image_path.name}")
    print(f"Dimensions: {height}x{width}")

    # Create SIFT detector
    print("Detecting SIFT features...")
    detector = cudasift.SiftDetector(img_dims=(height, width))

    # Detect features
    keypoints, descriptors = detector.detect(str(image_path))

    print(f"Found {len(keypoints)} keypoints")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Original image
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Image with keypoints
    draw_keypoints(axes[1], img, keypoints, max_display=200)

    plt.tight_layout()
    plt.suptitle(f"SIFT Feature Detection: {image_path.name}", y=1.02, fontsize=14)
    plt.show()

    # Optional: Show scale distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scale distribution
    axes[0, 0].hist(keypoints[:, 2], bins=50, edgecolor="black")
    axes[0, 0].set_xlabel("Scale (sigma)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Scale Distribution")
    axes[0, 0].grid(True, alpha=0.3)

    # Orientation distribution
    axes[0, 1].hist(keypoints[:, 3], bins=36, edgecolor="black")
    axes[0, 1].set_xlabel("Orientation (radians)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Orientation Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # Spatial distribution
    axes[1, 0].scatter(keypoints[:, 0], keypoints[:, 1], s=1, alpha=0.5)
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    axes[1, 0].set_title("Spatial Distribution")
    axes[1, 0].set_xlim(0, width)
    axes[1, 0].set_ylim(height, 0)  # Invert y-axis
    axes[1, 0].grid(True, alpha=0.3)

    # Scale vs position
    scatter = axes[1, 1].scatter(
        keypoints[:, 0], keypoints[:, 1], c=keypoints[:, 2], s=5, alpha=0.6, cmap="viridis"
    )
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")
    axes[1, 1].set_title("Keypoint Scale (color)")
    axes[1, 1].set_xlim(0, width)
    axes[1, 1].set_ylim(height, 0)  # Invert y-axis
    plt.colorbar(scatter, ax=axes[1, 1], label="Scale")

    plt.tight_layout()
    plt.suptitle(f"SIFT Statistics: {image_path.name}", y=1.02, fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
