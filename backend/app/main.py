
"""
Purpose:
    Initialize the NeuroMask FastAPI application and register its routes.

Working:
    1. Create the FastAPI application.
    2. Register the health-check endpoint.
    3. Import and register the MRI prediction router.
    4. Expose prediction functionality under the /api prefix.

Expected Output:
    /health       -> API health status
    /api/predict  -> MRI tumor segmentation endpoint

Important:
    Registering the prediction router connects the API endpoint.
    The trained model performs inference through prediction_service.py.
"""

from fastapi import FastAPI

from app.api.prediction import router as prediction_router
from fastapi.middleware.cors import CORSMiddleware

# Initialize the NeuroMask FastAPI application.
app = FastAPI(
    title="NeuroMask API",
    description="Brain MRI Tumor Segmentation Backend",
    version="1.0.0"
)

# Allow the React frontend to communicate with the FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health-check endpoint to verify that the backend is running.
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "NeuroMask API is running successfully."
    }


# Register the MRI prediction router.
app.include_router(
    prediction_router,
    prefix="/api",
    tags=["Prediction"]
)
