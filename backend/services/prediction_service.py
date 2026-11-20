
"""
Purpose:
    Create a reusable prediction service for NeuroMask.

Working:
    1. Load the previously trained U-Net model.
    2. Set the model to evaluation mode.
    3. Preprocess an uploaded MRI image.
    4. Generate segmentation logits using the model.
    5. Convert logits into probabilities using sigmoid.
    6. Apply a 0.5 threshold to generate a binary mask.
    7. Return the mask as a NumPy array containing 0 and 255.

Expected Output:
    A 512 x 512 binary segmentation mask.

Important:
    The trained model is loaded once when this module is imported.
    This service performs inference only; it does not retrain the model.
"""

import numpy as np
import torch

from model.model_loader import load_model
from preprocessing.image_preprocessor import preprocess_mri


# Load the trained model once.
MODEL, DEVICE = load_model()
MODEL.eval()


def predict_mri(image_path: str) -> np.ndarray:
    """
    Generate a binary tumor segmentation mask for an MRI image.

    Args:
        image_path: Path to the input MRI image.

    Returns:
        A NumPy array of shape (512, 512) with values 0 and 255.
    """

    # Preprocess the input MRI image.
    image_tensor = preprocess_mri(image_path).to(DEVICE)

    # Generate segmentation predictions.
    with torch.inference_mode():
        logits = MODEL(image_tensor)

        # Convert logits into probabilities.
        probabilities = torch.sigmoid(logits)

        # Convert probabilities into a binary mask.
        binary_mask = (probabilities >= 0.5).to(torch.uint8)

    # Convert the tensor into a NumPy image.
    prediction_mask = (
        binary_mask.squeeze().cpu().numpy() * 255
    ).astype(np.uint8)

    return prediction_mask
