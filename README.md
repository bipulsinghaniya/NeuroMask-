# 🧠 NeuroMask

> **AI-Powered Brain MRI Tumor Segmentation** — Upload a brain MRI scan and receive a pixel-accurate tumor segmentation mask in seconds, powered by a trained U-Net deep learning model.

---
    
## ✨ What It Does

NeuroMask is an end-to-end full-stack machine learning application for **brain tumor segmentation from MRI scans**. It combines a PyTorch U-Net model trained on 2,192 annotated MRI images with a FastAPI backend and a React frontend — letting you upload any brain MRI and instantly get a predicted binary tumor mask.

---

## 📸 Demo

| Step | Description |
|------|-------------|
| **1. Upload** | Select a grayscale brain MRI (PNG or JPEG, ≤ 10 MB) |
| **2. Predict** | Click **Generate Tumor Mask** — the model runs inference |
| **3. Download** | View the side-by-side result and download the segmentation mask |

---

## 🏆 Model Performance

The U-Net was evaluated on **329 held-out test images** across three tumor categories:

### Overall Metrics

| Metric | Score |
|--------|------:|
| **Dice Score** | 0.7642 |
| **IoU (Jaccard)** | 0.6184 |
| **Precision** | 0.7660 |
| **Recall** | 0.7623 |

### Category-wise Breakdown

| Category | Test Images | Dice | IoU | Precision | Recall |
|----------|------------:|-----:|----:|----------:|-------:|
| Glioma | 83 | 0.749 | 0.5988 | 0.7667 | 0.7321 |
| Meningioma | 107 | 0.7403 | 0.5876 | 0.7422 | 0.7384 |
| **Pituitary Tumor** | 139 | **0.8118** | **0.6833** | **0.7969** | **0.8273** |

---

## 🗂️ Project Structure

```text
NeuroMask/
│
├── backend/                        # FastAPI + PyTorch inference server
│   ├── app/
│   │   ├── main.py                 # FastAPI app entry point & CORS setup
│   │   ├── api/
│   │   │   └── prediction.py       # POST /api/predict — MRI upload endpoint
│   │   ├── models/
│   │   │   └── unet.py             # U-Net architecture (app copy)
│   │   ├── schemas/
│   │   │   └── prediction.py       # Pydantic request/response schemas
│   │   ├── services/
│   │   │   └── prediction_service.py
│   │   └── utils/
│   ├── model/
│   │   ├── unet.py                 # U-Net encoder–bottleneck–decoder
│   │   └── model_loader.py         # Loads neuromask_best_model.pth
│   ├── preprocessing/
│   │   └── image_preprocessor.py  # Grayscale → resize → min-max normalize →  tensor
│   ├── services/
│   │   └── prediction_service.py  # Runs inference, returns uint8 mask
│   ├── training/                   # Training scripts (Colab)
│   ├── weights/                    # (Legacy) model weight storage
│   ├── tests/
│   │   ├── test_model_loader.py
│   │   ├── test_prediction_service.py
│   │   └── test_preprocessing.py
│   ├── check_dataset.py
│   └── requirements.txt
│
├── frontend/                       # React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx                 # Main upload → predict → result UI
│   │   ├── App.css
│   │   ├── main.jsx
│   │   └── assets/
│   ├── package.json
│   └── vite.config.js
│
├── dataset/
│   ├── raw/                        # Original Kaggle MRI images & masks
│   ├── processed/                  # Augmented / preprocessed images
│   │   ├── Glioma/
│   │   ├── Meningioma/
│   │   └── Pituitary/
│   ├── normalized_images/          # Min-max normalized images
│   └── splits/                     # Train / val / test split indices
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA notebook
│   ├── 02_data_preprocessing.ipynb # Preprocessing pipeline notebook
│   └── 03_UNet_Model_Development.md
│
├── outputs/
│   ├── checkpoints/
│   │   └── neuromask_best_model.pth  # 🏋️ Trained model weights
│   ├── graphs/
│   │   └── neuromask_category_comparison.png
│   ├── metrics/
│   │   ├── overall_test_metrics.csv
│   │   ├── category_wise_metrics.csv
│   │   └── prediction_manifest.csv
│   └── predictions/                # Sample inference results
│       ├── Glioma_1.png / Glioma_2.png
│       ├── Meningioma_1.png / Meningioma_2.png
│       └── Pituitary_tumor_1.png / Pituitary_tumor_2.png
│
├── documentation/                  # Additional project docs
├── 01_EDA.md                       # Exploratory Data Analysis report
├── venv/                           # Python virtual environment
└── README.md
```

---

## 🧬 Model Architecture — U-Net

The model is a custom U-Net with:

- **Input:** `[B, 1, 512, 512]` — single-channel grayscale MRI
- **Output:** `[B, 1, 512, 512]` — binary tumor probability map

```
Input (1×512×512)
     │
 DoubleConv → 64              ← initial_conv
     │
  MaxPool + DoubleConv → 128  ← down1
     │
  MaxPool + DoubleConv → 256  ← down2
     │
  MaxPool + DoubleConv → 512  ← down3
     │
  MaxPool + DoubleConv → 1024 ← bottleneck (Down block)
     │
  ConvTranspose + DoubleConv → 512  ← up1  (+ skip from down3)
     │
  ConvTranspose + DoubleConv → 256  ← up2  (+ skip from down2)
     │
  ConvTranspose + DoubleConv → 128  ← up3  (+ skip from down1)
     │
  ConvTranspose + DoubleConv → 64   ← up4  (+ skip from initial_conv)
     │
  Conv2d(1×1) → 1                   ← output_layer
     │
Sigmoid → threshold @ 0.5 → Binary Mask (0 / 255)
```

Each `DoubleConv` block: `Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU` (all with `bias=False`).

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | PyTorch |
| **Model** | Custom U-Net |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React 19 + Vite 8 |
| **Image Processing** | Pillow, NumPy |
| **Training Environment** | Google Colab (GPU) |
| **Package Manager** | pip (backend), npm (frontend) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- The trained checkpoint at `outputs/checkpoints/neuromask_best_model.pth`

---

### 1 — Backend Setup

```bash
# Create and activate the virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # macOS / Linux

# Install backend dependencies
pip install -r backend/requirements.txt
```

**Start the FastAPI server:**

```bash
# From the project root
python -m uvicorn app.main:app --app-dir backend --reload
```

The API will be available at **http://127.0.0.1:8000**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Verify the server is running |
| `/api/predict` | POST | Upload MRI → returns segmentation mask (PNG) |
| `/docs` | GET | Interactive Swagger UI |

---

### 2 — Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The dashboard will be available at **http://localhost:5173**

---

### 3 — Full Stack (Both at Once)

Open two terminals:

```bash
# Terminal 1 — Backend
python -m uvicorn app.main:app --app-dir backend --reload

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Then open **http://localhost:5173** in your browser.

---

## 🔌 API Usage

### `POST /api/predict`

Upload a brain MRI image and receive its tumor segmentation mask.

**Request:** `multipart/form-data`

| Field | Type | Constraint |
|-------|------|-----------|
| `image` | file | PNG or JPEG, max 10 MB |

**Response:** PNG image (`image/png`) — 512 × 512 binary mask  
- Pixel value `0` → background  
- Pixel value `255` → predicted tumor region

**Example with curl:**

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "image=@/path/to/mri.png" \
  --output prediction_mask.png
```

**Error responses:**

| Code | Reason |
|------|--------|
| `400` | Empty file, unsupported format, or corrupted image |
| `413` | File exceeds 10 MB |
| `500` | Internal inference error |

---

## 🗃️ Dataset

**Source:** [Brain Tumor Dataset: Segmentation & Classification](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification) — Kaggle (Uploader: Indrakumar K)

| Category | MRI Images | Masks |
|----------|----------:|------:|
| Glioma | 554 | 554 |
| Meningioma | 708 | 708 |
| Pituitary Tumor | 930 | 930 |
| **Total** | **2,192** | **2,192** |

- All images: **512 × 512 pixels**, grayscale
- All 4,384 PNG files passed integrity checks (zero corrupted, zero duplicates)
- Train / Validation / Test split applied prior to any augmentation

---

## 🔄 Inference Pipeline

```
MRI File (PNG / JPEG)
        │
        ▼
  PIL Image.open()
        │
  Convert to Grayscale (mode "L")
        │
  Resize → 512 × 512 (bilinear)
        │
  Min-Max Normalize → [0, 1]
        │
  torch.Tensor [1, 1, 512, 512]
        │
        ▼
    U-Net Forward Pass
        │
  torch.sigmoid(logits)
        │
  Threshold ≥ 0.5 → binary mask
        │
  × 255 → uint8 NumPy array
        │
        ▼
  PNG Response (512 × 512)
```

---

## 📊 Notebooks & Reports

| File | Description |
|------|-------------|
| [`01_EDA.md`](01_EDA.md) | Dataset integrity, pixel intensity, and mask analysis |
| [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | Interactive EDA notebook |
| [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb) | Preprocessing pipeline |
| [`notebooks/03_UNet_Model_Development.md`](notebooks/03_UNet_Model_Development.md) | U-Net architecture design notes |

---

## 🧪 Running Tests

```bash
# From the project root with venv activated
python -m pytest backend/tests/ -v
```

Test coverage includes:
- `test_model_loader.py` — checkpoint loading and device selection
- `test_preprocessing.py` — MRI tensor shape and normalization range
- `test_prediction_service.py` — end-to-end mask output validation

---

## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| [`backend/app/main.py`](backend/app/main.py) | FastAPI app, CORS, router registration |
| [`backend/app/api/prediction.py`](backend/app/api/prediction.py) | `POST /api/predict` endpoint |
| [`backend/model/unet.py`](backend/model/unet.py) | U-Net architecture definition |
| [`backend/model/model_loader.py`](backend/model/model_loader.py) | Checkpoint loader |
| [`backend/preprocessing/image_preprocessor.py`](backend/preprocessing/image_preprocessor.py) | MRI preprocessing pipeline |
| [`backend/services/prediction_service.py`](backend/services/prediction_service.py) | Inference orchestration |
| [`frontend/src/App.jsx`](frontend/src/App.jsx) | React dashboard UI |
| [`outputs/checkpoints/neuromask_best_model.pth`](outputs/checkpoints/neuromask_best_model.pth) | Trained model weights |
| [`outputs/metrics/overall_test_metrics.csv`](outputs/metrics/overall_test_metrics.csv) | Aggregated test results |
| [`outputs/metrics/category_wise_metrics.csv`](outputs/metrics/category_wise_metrics.csv) | Per-category test results |

---

## 🛣️ Roadmap

- [x] Exploratory Data Analysis
- [x] Data preprocessing pipeline
- [x] U-Net architecture implementation
- [x] Model training (Google Colab GPU)
- [x] FastAPI inference server
- [x] React frontend dashboard
- [ ] Overlay visualization (MRI + mask blended)
- [ ] Confidence score / probability heatmap output
- [ ] Docker Compose deployment

---

## 📄 License

This project is for research and educational purposes.  
Dataset attribution: Indrakumar K via [Kaggle](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification).
