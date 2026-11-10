
# """
# Purpose:
#     Create a FastAPI endpoint for NeuroMask MRI tumor segmentation.

# Working:
#     1. Accept an MRI image through an HTTP POST request.
#     2. Validate that an image file was uploaded.
#     3. Save the uploaded image temporarily.
#     4. Call the prediction service to generate a tumor mask.
#     5. Convert the predicted NumPy mask into a PNG image.
#     6. Return the PNG mask as an HTTP response.
#     7. Delete the temporary uploaded file.

# Expected Output:
#     A PNG segmentation mask with dimensions 512 x 512
#     and pixel values 0 (background) or 255 (tumor).

# Important:
#     This endpoint performs inference only.
#     It does not train or modify the trained U-Net model.
# """

# from pathlib import Path
# from tempfile import NamedTemporaryFile
# from io import BytesIO

# from fastapi import APIRouter, UploadFile, File, HTTPException
# from fastapi.responses import StreamingResponse
# from PIL import Image

# from services.prediction_service import predict_mri


# # Create an API router for prediction-related endpoints.
# router = APIRouter()


# @router.post("/predict")
# async def predict(image: UploadFile = File(...)):
#     """
#     Accept an MRI image and return its predicted tumor segmentation mask.
#     """

#     # Validate the uploaded file.
#     if not image.content_type or not image.content_type.startswith("image/"):
#         raise HTTPException(
#             status_code=400,
#             detail="Please upload a valid image file."
#         )

#     temp_path = None

#     try:
#         # Save the uploaded MRI image temporarily.
#         suffix = Path(image.filename or "upload.png").suffix or ".png"

#         with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
#             temp_file.write(await image.read())
#             temp_path = temp_file.name

#         # Generate the tumor segmentation mask.
#         prediction_mask = predict_mri(temp_path)

#         # Convert the NumPy mask into a PNG image in memory.
#         output_buffer = BytesIO()
#         Image.fromarray(prediction_mask).save(output_buffer, format="PNG")
#         output_buffer.seek(0)

#         # Return the generated segmentation mask.
#         return StreamingResponse(
#             output_buffer,
#             media_type="image/png",
#             headers={
#                 "Content-Disposition": 'inline; filename="prediction_mask.png"'
#             }
#         )

#     except HTTPException:
#         raise

#     except Exception as error:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Prediction failed: {str(error)}"
#         )

#     finally:
#         # Remove the temporary uploaded image.
#         if temp_path and Path(temp_path).exists():
#             Path(temp_path).unlink()

#         await image.close()




"""
Purpose:
    Create a secure FastAPI endpoint for NeuroMask MRI tumor segmentation.

Working:
    1. Accept an uploaded MRI image through an HTTP POST request.
    2. Validate the uploaded file size and actual image format.
    3. Reject empty, corrupted, or unsupported image files.
    4. Save the validated MRI image temporarily.
    5. Call the trained U-Net prediction service.
    6. Convert the predicted NumPy mask into a PNG image.
    7. Return the segmentation mask as an HTTP response.
    8. Delete the temporary uploaded file safely.

Expected Output:
    A PNG segmentation mask with dimensions 512 x 512
    and pixel values 0 (background) or 255 (predicted tumor).

Important:
    This endpoint performs inference only.
    It does not train or modify the trained U-Net model.
    Maximum upload size is limited to 10 MB.
"""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from services.prediction_service import predict_mri


# Configure logging for prediction errors.
logger = logging.getLogger(__name__)

# Create an API router for prediction-related endpoints.
router = APIRouter()

# Set the maximum permitted upload size to 10 MB.
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allow only PNG and JPEG MRI images.
ALLOWED_FORMATS = {
    "PNG": ".png",
    "JPEG": ".jpg"
}


@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Validate an uploaded MRI image and return its tumor segmentation mask.
    """

    temp_path = None

    try:
        # Read the uploaded file into memory.
        file_content = await image.read()

        # Reject empty uploads.
        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # Validate the uploaded file size.
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB."
            )

        # Verify the actual image format and file integrity.
        try:
            with Image.open(BytesIO(file_content)) as uploaded_image:
                image_format = uploaded_image.format
                uploaded_image.verify()

            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail="Only PNG and JPEG images are supported."
                )

        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid or supported image."
            )

        # Save the validated MRI image temporarily.
        suffix = ALLOWED_FORMATS[image_format]

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        # Generate the tumor segmentation mask.
        prediction_mask = predict_mri(temp_path)

        # Convert the NumPy mask into a PNG image in memory.
        output_buffer = BytesIO()
        Image.fromarray(prediction_mask).save(output_buffer, format="PNG")
        output_buffer.seek(0)

        # Return the generated segmentation mask.
        return StreamingResponse(
            output_buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": 'inline; filename="prediction_mask.png"'
            }
        )

    except HTTPException:
        raise

    except Exception:
        # Log the internal error without exposing details to API users.
        logger.exception("NeuroMask prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error."
        )

    finally:
        # Safely remove the temporary uploaded image.
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except OSError:
                logger.exception("Could not remove temporary MRI file.")

        # Close the uploaded file.
        await image.close()
