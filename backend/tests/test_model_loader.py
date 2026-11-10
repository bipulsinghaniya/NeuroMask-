
"""
Purpose:
    Test whether the previously trained NeuroMask U-Net model
    can be restored and perform inference successfully.

Working:
    1. Locate the backend directory.
    2. Import the model-loading function.
    3. Load the existing trained checkpoint.
    4. Verify evaluation mode.
    5. Run a dummy MRI tensor through the model.
    6. Validate output shape and numerical stability.

Expected Output:
    Successful model loading and finite segmentation output.

Important:
    This test does not train or modify the model.
"""

import sys
from pathlib import Path

import torch

# Locate project directories.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from model.model_loader import load_model


def test_model_loader():
    print("Starting U-Net model loading test...")

    # Restore the trained model.
    model, device = load_model()

    # Verify evaluation mode.
    assert model.training is False

    # Create a dummy grayscale MRI tensor.
    dummy_image = torch.zeros(
        (1, 1, 128, 128),
        dtype=torch.float32,
        device=device
    )

    # Perform inference without calculating gradients.
    with torch.no_grad():
        prediction = model(dummy_image)

    # Validate prediction shape.
    assert prediction.shape == (1, 1, 128, 128)

    # Validate numerical stability.
    assert torch.isfinite(prediction).all().item()

    print("Device:", device)
    print("Model mode: Evaluation")
    print("Input shape:", tuple(dummy_image.shape))
    print("Output shape:", tuple(prediction.shape))
    print(
        "All output values finite:",
        torch.isfinite(prediction).all().item()
    )

    print("\nMODEL LOADER TEST PASSED!")


if __name__ == "__main__":
    test_model_loader()
