
"""
Purpose:
    Download and load the trained NeuroMask U-Net checkpoint
    from Hugging Face when it is not available locally.
"""

from pathlib import Path
import os
import urllib.request
import torch
import tempfile
from model.unet import UNet


# Hugging Face model URL
# MODEL_URL = os.getenv(
#     "HF_MODEL_URL",
#     "https://huggingface.co/bipul08/neuromask-brain-tumor-segmentation/resolve/main/neuromask_best_model.pth"
# )
MODEL_URL = os.getenv(
    "HF_MODEL_URL",
    "https://huggingface.co/bipul08/neuromask-brain-tumor-segmentation/resolve/main/neuromask_inference.pth"
)
# Store downloaded model in temporary storage
# CHECKPOINT_PATH = Path("/tmp/neuromask_best_model.pth")
# CHECKPOINT_PATH = Path("/tmp/neuromask_inference.pth")
CHECKPOINT_PATH = Path(tempfile.gettempdir()) / "neuromask_inference.pth"


def download_model():

    if CHECKPOINT_PATH.exists():
        print("Model checkpoint already exists.")
        return

    print("Downloading NeuroMask model from Hugging Face...")

    temp_path = CHECKPOINT_PATH.with_suffix(".download")

    try:
        urllib.request.urlretrieve(MODEL_URL, temp_path)
        temp_path.replace(CHECKPOINT_PATH)
        print("Model downloaded successfully.")

    except Exception as error:
        if temp_path.exists():
            temp_path.unlink()

        raise RuntimeError(
            f"Failed to download model checkpoint: {error}"
        )


def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Download checkpoint if it is missing.
    download_model()

    # Reconstruct U-Net architecture.
    model = UNet()

    # Load trained weights.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(device)
    model.eval()

    print(f"NeuroMask model loaded successfully on {device}.")

    return model, device
















# """
# Purpose:
#     Load the previously trained NeuroMask U-Net checkpoint.

# Working:
#     1. Detects the project root directory.
#     2. Locates the saved best model checkpoint.
#     3. Selects CPU or CUDA automatically.
#     4. Reconstructs the U-Net architecture.
#     5. Loads trained model weights with strict validation.
#     6. Sets the model to evaluation mode.

# Expected Output:
#     A restored U-Net model ready for MRI segmentation inference.

# Important:
#     This code loads existing trained weights.
#     It does NOT train the model again.
# """

# from pathlib import Path
# import torch

# from model.unet import UNet

# # Project root directory
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# # Saved checkpoint location
# CHECKPOINT_PATH = (
#     PROJECT_ROOT
#     / "outputs"
#     / "checkpoints"
#     / "neuromask_best_model.pth"
# )


# def load_model():

#     # Select available computation device.
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     # Verify checkpoint existence.
#     if not CHECKPOINT_PATH.exists():
#         raise FileNotFoundError(
#             f"Model checkpoint not found: {CHECKPOINT_PATH}"
#         )

#     # Reconstruct the trained U-Net architecture.
#     model = UNet()

#     # Load checkpoint safely.
#     checkpoint = torch.load(
#         CHECKPOINT_PATH,
#         map_location=device,
#         weights_only=True
#     )

#     # Restore trained model parameters.
#     model.load_state_dict(
#         checkpoint["model_state_dict"],
#         strict=True
#     )

#     # Set inference mode.
#     model.to(device)
#     model.eval()

#     return model, device






















