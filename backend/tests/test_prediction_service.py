
"""
Purpose:
    Create a reusable test for the NeuroMask prediction service.

Working:
    1. Locate the project root and backend directory.
    2. Import the predict_mri function.
    3. Select one real Glioma MRI image.
    4. Generate its predicted segmentation mask.
    5. Validate mask shape, data type and pixel values.
    6. Save the predicted mask for visual inspection.

Expected Output:
    A 512 x 512 binary mask containing only 0 and 255.
    Prediction test should pass successfully.

Important:
    This test performs inference using the existing trained model.
    It does not train or modify the model.
"""

import sys
from pathlib import Path

import numpy as np

# Locate project directories.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from services.prediction_service import predict_mri


# Select one real MRI image.
IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "Glioma"
    / "enh_1849.png"
)

# Output location for the predicted mask.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_PATH = OUTPUT_DIR / "test_prediction.png"


def test_prediction_service():
    print("Starting NeuroMask prediction test...")

    # Generate the segmentation mask.
    prediction = predict_mri(str(IMAGE_PATH))

    # Validate mask shape.
    assert prediction.shape == (512, 512), (
        f"Unexpected shape: {prediction.shape}"
    )

    # Validate data type.
    assert prediction.dtype == np.uint8

    # Validate binary pixel values.
    unique_values = np.unique(prediction)
    assert set(unique_values.tolist()).issubset({0, 255}), (
        f"Unexpected pixel values: {unique_values}"
    )

    # Validate numerical stability.
    assert np.isfinite(prediction).all()

    # Save prediction for visual inspection.
    from PIL import Image

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction).save(OUTPUT_PATH)

    print("Mask shape:", prediction.shape)
    print("Data type:", prediction.dtype)
    print("Unique values:", unique_values.tolist())
    print("Tumor pixels:", int(np.count_nonzero(prediction)))
    print("Saved mask:", OUTPUT_PATH)

    print("\nPREDICTION SERVICE TEST PASSED!")


if __name__ == "__main__":
    test_prediction_service()
