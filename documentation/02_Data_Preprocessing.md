# Phase 2: Data Preprocessing

## 1. Objective

The objective of this phase is to transform the raw Brain MRI images and their corresponding segmentation masks into a clean, standardized, and model-ready dataset for brain tumor segmentation.

The original dataset will remain unchanged. All processed files will be stored separately.

## 2. Dataset Information

- Total MRI Images: 2,192
- Total Segmentation Masks: 2,192
- Glioma: 554 pairs
- Meningioma: 708 pairs
- Pituitary Tumor: 930 pairs
- Original Image Resolution: 512 × 512
- Image Format: PNG
- Mask Format: PNG

## 3. Preprocessing Tasks

### Step 1: Mask Binarization

Convert grayscale segmentation masks into binary masks using the verified thresholding strategy.

Proposed mapping:

- Pixel values below 128 → Background (0)
- Pixel values greater than or equal to 128 → Tumor (1)

The original masks will not be modified.

### Step 2: Dataset Splitting

Divide the dataset into training, validation, and testing subsets.

The split must prevent data leakage by grouping related images at the patient level wherever patient identifiers are available.

### Step 3: MRI Image Normalization

Normalize MRI image intensities into a consistent numerical range suitable for deep learning.

### Step 4: Image and Mask Alignment

Ensure that every MRI image corresponds to its correct binary segmentation mask.

### Step 5: Data Augmentation

Apply suitable geometric and intensity transformations.

Spatial transformations must be applied consistently to MRI images and their masks. Nearest-neighbor interpolation must be used for categorical masks.

### Step 6: PyTorch Dataset and DataLoader

Create a custom PyTorch Dataset and DataLoader to load MRI images and masks in batches.

### Step 7: Final Preprocessing Validation

Verify:

- Image and mask dimensions
- Binary mask values
- Image-mask pairing
- Missing or corrupted files
- Dataset split integrity
- Sample batch shapes and data types

## 4. Output Directory Structure

```text
dataset/
├── raw/
│   ├── Glioma/
│   ├── Meningioma/
│   └── Pituitary tumor/
│
└── processed/
    ├── Glioma/
    ├── Meningioma/
    └── Pituitary tumor/
```

## 5. Important Rules

- Never modify original raw images or masks.
- Preserve image-mask correspondence.
- Avoid data leakage between dataset splits.
- Use nearest-neighbor interpolation for masks.
- Validate every preprocessing operation.
- Document actual results after each completed step.

## 6. Phase Status

Status: In Progress

Completed:
- Dataset EDA
- Mask encoding investigation
- Identification of suspicious mask values
- Proposed binary thresholding strategy

Pending:
- Mask binarization implementation
- Dataset splitting
- MRI normalization
- Data augmentation
- PyTorch Dataset and DataLoader
- Final preprocessing validation

## 7. Final Outcome

At the end of this phase, we aim to obtain a validated, standardized dataset ready for baseline U-Net development and training.