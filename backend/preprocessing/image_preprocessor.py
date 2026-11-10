"""
Purpose:
    Preprocess an uploaded brain MRI image before U-Net inference.

Working:
    1. Open the uploaded MRI image.
    2. Convert it into grayscale.
    3. Resize it to 512 x 512 using bilinear interpolation.
    4. Apply per-image Min-Max normalization.
    5. Convert the image into a PyTorch tensor.
    6. Add batch and channel dimensions.

Expected Output:
    A float32 tensor with shape [1, 1, 512, 512]
    and normalized pixel values between 0 and 1.

Important:
    The preprocessing should remain consistent with the
    preprocessing used during U-Net training.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def preprocess_mri(image_path: str) -> torch.Tensor:
    """
    Convert an MRI image into a normalized U-Net input tensor.

    Args:
        image_path: Path to the uploaded MRI image.

    Returns:
        Normalized float32 tensor of shape [1, 1, 512, 512].
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"MRI image not found: {image_path}")

    # Read the image and convert it to grayscale.
    image = Image.open(image_path).convert("L")

    # Resize using bilinear interpolation, matching image preprocessing.
    image = image.resize((512, 512), Image.Resampling.BILINEAR)

    # Convert the image into a NumPy float32 array.
    image_array = np.asarray(image, dtype=np.float32)

    # Apply per-image Min-Max normalization.
    min_value = image_array.min()
    max_value = image_array.max()

    if max_value > min_value:
        image_array = (image_array - min_value) / (max_value - min_value)
    else:
        image_array = np.zeros_like(image_array, dtype=np.float32)

    # Convert NumPy array into a PyTorch tensor.
    image_tensor = torch.from_numpy(image_array).float()

    # Add channel and batch dimensions.
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    return image_tensor