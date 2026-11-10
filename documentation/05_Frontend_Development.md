# Phase 5 — Frontend Development

## 1. Objective

Develop a modern, responsive web interface for NeuroMask, an AI-powered Brain MRI Tumor Segmentation application.

The frontend will allow users to upload brain MRI images, request tumor segmentation from the trained U-Net model through the FastAPI backend, and visualize the generated segmentation mask.

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| React.js | Frontend component development |
| Vite | Development server and build tool |
| Tailwind CSS | Responsive UI styling |
| Axios | Backend API communication |
| Lucide React | Interface icons |

## 3. Planned Features

1. Professional NeuroMask dashboard.
2. MRI image upload interface.
3. Uploaded image preview.
4. Image validation.
5. Prediction loading indicator.
6. Integration with FastAPI prediction endpoint.
7. Predicted segmentation mask visualization.
8. Original MRI and predicted mask comparison.
9. Download predicted segmentation mask.
10. Error handling and responsive design.

## 4. Backend Integration

The frontend will communicate with the existing FastAPI backend.

**API endpoint:**

`POST http://127.0.0.1:8000/api/predict`

**Request format:** multipart/form-data

**Request field:** image

**Successful response:** PNG segmentation mask.

## 5. Development Progress

- [ ] Frontend project setup
- [ ] Tailwind CSS configuration
- [ ] Dashboard layout
- [ ] MRI upload component
- [ ] Image preview
- [ ] Prediction integration
- [ ] Segmentation result visualization
- [ ] Download functionality
- [ ] Responsive design
- [ ] Frontend testing

## 6. Expected Outcome

A responsive NeuroMask web application where users can upload a brain MRI image, generate a tumor segmentation mask using the trained U-Net model, visualize the result, and download the predicted mask.

**Current Status:** Frontend development started.



## Step 1 — React + Vite Frontend Setup

### Purpose
Initialize the NeuroMask frontend using React and Vite.

### Implementation
Created the frontend project inside the root `frontend/` directory using:

```bash
npm create vite@latest frontend -- --template react
```

### Project Structure
- `frontend/src/` — React application source code.
- `frontend/public/` — Static assets.
- `frontend/package.json` — Project configuration and dependencies.
- `frontend/vite.config.js` — Vite configuration.
- `frontend/node_modules/` — Installed dependencies.

### Verification
The Vite development server started successfully.

- Local URL: `http://localhost:5173/`
- Frontend project creation: Successful.

**Status:** React + Vite frontend initialized successfully.