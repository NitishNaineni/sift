"""
Test SIFT implementation using Oxford affine dataset.
"""

import cv2

from cudasift import Sift, SiftParams

# Load a test image from Oxford dataset
image_path = "data/oxford_affine/graf/img1.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load {image_path}. Run get_oxford_affine.py first.")

print(f"Loaded image: {image.shape}")
h, w = image.shape[:2]

# Initialize SIFT with image dimensions
print(f"Initializing SIFT with dimensions: ({h}, {w})")
params = SiftParams(img_dims=(h, w))
sift = Sift(params)

# Run SIFT extraction
print("Running SIFT extraction...")
keypoints_host, snapshot = sift.compute(image_path)

num_keypoints = int(keypoints_host.counter[0])
print(f"Number of keypoints detected: {num_keypoints}")
print(f"Keypoints counter: {keypoints_host.counter}")

# Basic validation
assert num_keypoints > 0, "Should detect at least some keypoints"
assert keypoints_host.descriptors.shape[1] == 128, "SIFT descriptors should be 128-dimensional"

# Test with another image from the sequence
image2_path = "data/oxford_affine/graf/img2.png"
image2 = cv2.imread(image2_path)

if image2 is None:
    print(f"Warning: Could not load {image2_path}, skipping second image test")
else:
    h2, w2 = image2.shape[:2]
    if (h2, w2) == (h, w):
        print(f"\nTesting with second image: {image2.shape}")
        keypoints_host2, snapshot2 = sift.compute(image2_path)
        num_keypoints2 = int(keypoints_host2.counter[0])
        print(f"Number of keypoints detected: {num_keypoints2}")
        assert num_keypoints2 > 0, "Should detect keypoints in second image too"
    else:
        print(f"\nSkipping second image: different dimensions ({h2}, {w2}) vs ({h}, {w})")
        print("Testing with first image again...")
        keypoints_host2, snapshot2 = sift.compute(image_path)
        num_keypoints2 = int(keypoints_host2.counter[0])
        print(f"Number of keypoints detected: {num_keypoints2}")
        assert num_keypoints2 > 0, "Should detect keypoints consistently"

print("\n✓ All tests passed!")
