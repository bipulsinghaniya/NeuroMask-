# 01. Exploratory Data Analysis (EDA)

## 1. Project Overview

NeuroMask is a brain MRI tumor segmentation project focused on identifying tumor regions from MRI images using deep learning.

The dataset was obtained from Kaggle:

- **Dataset:** Brain Tumor Dataset: Segmentation & Classification
- **Uploader:** Indrakumar K
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification)

## 2. Dataset Overview

The dataset contains three tumor categories.

| Category | MRI Images | Masks |
|---|---:|---:|
| Glioma | 554 | 554 |
| Meningioma | 708 | 708 |
| Pituitary Tumor | 930 | 930 |
| **Total** | **2,192** | **2,192** |

Each MRI image has a corresponding segmentation mask.

All images and masks have dimensions of 512 × 512 pixels.

## 3. Dataset Integrity Analysis

### 3.1 Image-Mask Pair Verification

- Total MRI images: 2,192
- Total masks: 2,192
- Successfully matched pairs: 2,192

All image-mask pairs were matched using their filenames.

### 3.2 Corrupted Image Detection

A total of 4,384 PNG files were checked.

- Corrupted files: 0
- Valid files: 4,384

All files passed the image integrity verification.

### 3.3 Duplicate Image Detection

Exact duplicate MRI images were checked using MD5 hashing.

- Total MRI images: 2,192
- Unique images: 2,192
- Exact duplicates: 0

This check identifies exact file duplicates and does not rule out visually similar images with different encodings.

## 4. MRI Image Analysis

### 4.1 Image Dimensions

All MRI images and their corresponding masks have dimensions of 512 × 512 pixels.

No image-mask dimension mismatches were identified.

### 4.2 MRI Pixel Intensity Analysis

The average MRI pixel intensity by category was:

| Category | Mean Intensity |
|---|---:|
| Glioma | 49.70 |
| Meningioma | 55.76 |
| Pituitary Tumor | 69.97 |

These values describe the observed grayscale intensity distribution.

## 5. Segmentation Mask Analysis

### 5.1 Mask Image Modes

All 2,192 masks were found to use PIL image mode L (8-bit grayscale).

| Category | Mask Count | Mode |
|---|---:|---|
| Glioma | 554 | L |
| Meningioma | 708 | L |
| Pituitary Tumor | 930 | L |

### 5.2 Mask Pixel Value Analysis

The masks contain multiple pixel intensities.

The dominant observed pixel value is 3, while 255 is another frequently occurring value.

Additional values, including 2 and several other grayscale intensities, were also identified.

The exact semantic meaning of these pixel values has not yet been fully established.

### 5.3 Suspicious Mask Detection

A total of 31 masks were identified with values outside the exploratory expected-value set {3, 255}.

| Category | Suspicious Masks |
|---|---:|
| Glioma | 0 |
| Meningioma | 19 |
| Pituitary Tumor | 12 |
| **Total** | **31** |

The 19 Meningioma masks and 7 Pituitary masks contain pixel value 2.

Five additional Pituitary masks contain multiple grayscale values:

- enh_995_mask.png
- enh_996_mask.png
- enh_997_mask.png
- enh_998_mask.png
- enh_999_mask.png

### 5.4 Unexpected Pixel Analysis

A total of 5,946 unexpected pixels were identified across the 31 suspicious masks.

| Type | Pixel Count |
|---|---:|
| Pixel value 2 | 3,481 |
| Other grayscale values | 2,465 |
| **Total** | **5,946** |

### 5.5 Boundary Proximity Investigation

The five Pituitary masks containing multiple grayscale values were further investigated.

All 1,126 unusual pixels were found within the approximately 2-pixel expanded neighborhood of the candidate tumor region.

This indicates a spatial association between unusual pixel values and the candidate tumor boundary.

The exact origin and semantic meaning of these values remain unconfirmed.

## 6. Visual Inspection

The following visual inspections were performed:

- MRI images from all three categories.
- Original segmentation masks.
- MRI-mask overlays.
- Color-coded mask visualizations.
- Suspicious masks with additional grayscale values.
- Unusual pixel locations overlaid on MRI images.

These inspections helped identify potential annotation and encoding inconsistencies.

## 7. Key Findings

1. The dataset contains 2,192 MRI images and 2,192 corresponding masks.
2. All 4,384 PNG files passed integrity checks.
3. No exact duplicate MRI images were detected.
4. All images and masks have dimensions of 512 × 512.
5. All segmentation masks use grayscale image mode L.
6. Thirty-one masks contain unexpected pixel values under the exploratory {3, 255} assumption.
7. Five Pituitary masks contain multiple grayscale values concentrated near the candidate tumor boundary.

## 8. Limitations and Pending Verification

- The exact semantic meaning of mask pixel values must be verified before binary conversion.
- The reason for additional grayscale values has not been conclusively established.
- Filename matching and visual inspection do not guarantee annotation correctness.
- No original MRI images or segmentation masks were modified during EDA.

## 9. Conclusion

The exploratory analysis established the dataset structure, image integrity, image dimensions, intensity characteristics, and segmentation mask encoding patterns.

The dataset is suitable for proceeding to the data preprocessing investigation, subject to verification of the segmentation annotation convention.

The next phase will focus on preparing MRI images and segmentation masks for PyTorch-based deep learning while preserving the original dataset.