
"""
Purpose:
    Load the previously trained NeuroMask U-Net checkpoint.

Working:
    1. Detects the project root directory.
    2. Locates the saved best model checkpoint.
    3. Selects CPU or CUDA automatically.
    4. Reconstructs the U-Net architecture.
    5. Loads trained model weights with strict validation.
    6. Sets the model to evaluation mode.

Expected Output:
    A restored U-Net model ready for MRI segmentation inference.

Important:
    This code loads existing trained weights.
    It does NOT train the model again.
"""

from pathlib import Path
import torch

from model.unet import UNet

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Saved checkpoint location
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "neuromask_best_model.pth"
)


def load_model():

    # Select available computation device.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Verify checkpoint existence.
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Reconstruct the trained U-Net architecture.
    model = UNet()

    # Load checkpoint safely.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    # Restore trained model parameters.
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    # Set inference mode.
    model.to(device)
    model.eval()

    return model, device
