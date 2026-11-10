
"""
Purpose:
    Create a reusable test for the NeuroMask MRI preprocessing module.

Working:
    1. Locate the project root and backend directory.
    2. Import the preprocess_mri function.
    3. Load one real Glioma MRI image.
    4. Verify tensor shape, data type and normalized pixel range.
    5. Check that all tensor values are finite.
    6. Raise an error if any validation fails.

Expected Output:
    All preprocessing checks should pass and display
    PREPROCESSING TEST PASSED.

Usage:
    Run this file whenever MRI preprocessing needs to be tested.
"""

import sys
from pathlib import Path

import torch

# Locate the project root and backend directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from preprocessing.image_preprocessor import preprocess_mri


# Select one real MRI image for testing.
IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "Glioma"
    / "enh_1849.png"
)


def test_preprocessing():
    print("Starting MRI preprocessing test...")

    # Ensure the test image exists.
    assert IMAGE_PATH.exists(), f"Image not found: {IMAGE_PATH}"

    # Generate the preprocessed MRI tensor.
    image_tensor = preprocess_mri(str(IMAGE_PATH))

    # Validate tensor shape.
    assert image_tensor.shape == (1, 1, 512, 512)

    # Validate tensor data type.
    assert image_tensor.dtype == torch.float32

    # Validate normalized pixel range.
    assert image_tensor.min().item() >= 0.0
    assert image_tensor.max().item() <= 1.0

    # Validate numerical stability.
    assert torch.isfinite(image_tensor).all().item()

    print("Tensor shape:", tuple(image_tensor.shape))
    print("Data type:", image_tensor.dtype)
    print("Minimum:", image_tensor.min().item())
    print("Maximum:", image_tensor.max().item())
    print("All values finite:", torch.isfinite(image_tensor).all().item())

    print("\nPREPROCESSING TEST PASSED!")


if __name__ == "__main__":
    test_preprocessing()
