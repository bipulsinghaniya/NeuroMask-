# Phase 3: U-Net Model Development

## 1. Objective

The objective of this phase is to design and implement a U-Net-based deep learning model for brain tumor segmentation using preprocessed MRI images.

The model will learn to identify and segment tumor regions from MRI scans using corresponding binary segmentation masks.

## 2. Selected Model Architecture

**Model: U-Net**

U-Net is an encoder-decoder convolutional neural network architecture widely used for biomedical image segmentation.

Its main components are:

- Encoder (Contracting Path)
- Bottleneck
- Decoder (Expanding Path)
- Skip Connections
- Output Segmentation Layer

## 3. Input and Output Specifications

| Parameter | Configuration |
|---|---|
| Input image | Brain MRI |
| Input channels | 1 (Grayscale) |
| Input dimensions | 512 × 512 |
| Output channels | 1 |
| Output dimensions | 512 × 512 |
| Segmentation type | Binary |
| Background label | 0 |
| Tumor label | 1 |

## 4. Planned Implementation Steps

1. Understand the U-Net architecture.
2. Design the encoder blocks.
3. Implement the bottleneck.
4. Design the decoder blocks.
5. Implement skip connections.
6. Create the final segmentation output layer.
7. Define the model configuration.
8. Test the model using a sample batch.
9. Verify output dimensions and parameter count.
10. Prepare the model for Google Colab GPU training.

## 5. Loss Function and Evaluation Metrics

Planned loss functions:

- Binary Cross-Entropy (BCE)
- Dice Loss
- Combined BCE + Dice Loss

Planned evaluation metrics:

- Dice Score
- Intersection over Union (IoU)
- Precision
- Recall

Final choices and configurations will be documented after implementation and experimentation.

## 6. Training Environment

- Development and debugging: Local Antigravity / VS Code.
- Actual model training: Google Colab.
- Training hardware: GPU available in Google Colab.
- Framework: PyTorch.

## 7. Important Implementation Rules

- Original MRI images and masks will remain unchanged.
- Training augmentation will be applied only to the training dataset.
- Validation and testing datasets will remain unaugmented.
- Model training and experimental results will be recorded separately.
- Actual experimental results will be documented using Google Colab outputs.

## 8. Expected Outcome

A working U-Net segmentation model that accepts a grayscale MRI image and produces a binary tumor segmentation prediction of the same spatial dimensions.

**Status:** In Progress