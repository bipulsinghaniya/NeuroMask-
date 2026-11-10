
/*
Purpose:
  Build the main NeuroMask MRI segmentation dashboard.

Working:
  1. Allow users to upload a brain MRI image.
  2. Display the uploaded MRI preview.
  3. Send the MRI to the FastAPI prediction endpoint.
  4. Receive and display the predicted segmentation mask.
  5. Allow users to download the generated mask.
  6. Handle loading and error states.

Expected Output:
  A functional MRI upload and segmentation interface
  connected to the NeuroMask backend.

Important:
  The backend must be running at http://127.0.0.1:8000.
  Prediction uses the previously trained U-Net model.
*/

import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api/predict";

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [predictionUrl, setPredictionUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!image) {
      setImagePreview(null);
      return;
    }

    const url = URL.createObjectURL(image);
    setImagePreview(url);

    return () => URL.revokeObjectURL(url);
  }, [image]);

  useEffect(() => {
    return () => {
      if (predictionUrl) URL.revokeObjectURL(predictionUrl);
    };
  }, [predictionUrl]);

  const handleImageChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setImage(file);
    setPredictionUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return null;
    });
    setError("");
  };

  const handlePrediction = async () => {
    if (!image) {
      setError("Please upload an MRI image first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", image);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Prediction failed.");
      }

      const resultBlob = await response.blob();

      setPredictionUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return URL.createObjectURL(resultBlob);
      });
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "Cannot connect to backend. Check that FastAPI is running and CORS is configured."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setImagePreview(null);
    setPredictionUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return null;
    });
    setError("");
  };

  return (
    <div className="app-container">
      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">N</div>
          <span>NeuroMask</span>
        </div>
        <span className="nav-status">
          <span className="status-dot"></span>
          AI Segmentation Platform
        </span>
      </header>

      <main className="main-content">
        <section className="hero">
          <span className="eyebrow">AI-POWERED MEDICAL IMAGING</span>
          <h1>Brain MRI Tumor Segmentation</h1>
          <p>
            Upload a brain MRI scan and generate a tumor segmentation
            mask using our trained U-Net deep learning model.
          </p>
        </section>

        <section className="workspace">
          <div className="upload-card">
            <div className="section-heading">
              <h2>Upload MRI Scan</h2>
              <span className="step-label">STEP 01</span>
            </div>

            <label className="upload-area">
              <div className="upload-icon">↑</div>
              <h3>Select MRI Image</h3>
              <p>Choose a brain MRI image from your device</p>
              <span className="browse-button">Browse Files</span>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />
            </label>

            <p className="file-note">
              Supported formats: PNG, JPG, JPEG
            </p>

            {image && (
              <div className="selected-file">
                <span>Selected: {image.name}</span>
                <button onClick={handleReset}>Remove</button>
              </div>
            )}

            <button
              className="predict-button"
              onClick={handlePrediction}
              disabled={!image || loading}
            >
              {loading ? "Generating Segmentation..." : "Generate Tumor Mask"}
            </button>

            {error && <p className="error-message">{error}</p>}
          </div>

          <div className="result-card">
            <div className="section-heading">
              <h2>Segmentation Results</h2>
              <span className="step-label">STEP 02</span>
            </div>

            <div className="result-grid">
              <div className="image-panel">
                <h3>Original MRI</h3>
                {imagePreview ? (
                  <img src={imagePreview} alt="Original MRI scan" />
                ) : (
                  <div className="empty-preview">
                    Your MRI preview will appear here
                  </div>
                )}
              </div>

              <div className="image-panel">
                <h3>Predicted Tumor Mask</h3>
                {predictionUrl ? (
                  <img src={predictionUrl} alt="Predicted segmentation mask" />
                ) : (
                  <div className="empty-preview">
                    Your segmentation result will appear here
                  </div>
                )}
              </div>
            </div>

            {predictionUrl && (
              <div className="success-area">
                <p>Segmentation generated successfully.</p>
                <a
                  className="download-button"
                  href={predictionUrl}
                  download="neuromask_prediction.png"
                >
                  Download Segmentation Mask ↓
                </a>
              </div>
            )}
          </div>
        </section>

        <footer>
          <p>NeuroMask · AI-Powered Brain MRI Tumor Segmentation</p>
          <span>Research & Development Project</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
