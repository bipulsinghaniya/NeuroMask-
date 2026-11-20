import { useEffect, useRef, useState } from "react";
import "./App.css";

// const API_URL = "http://127.0.0.1:8000/api/predict";
const API_URL = "https://neuromask-backend.onrender.com/api/predict";

const SAMPLES = [
  { id:1, label:"Glioma · Case 1",     category:"Glioma",      mri:"/samples/Glioma_1_mri.png", mask:"/samples/Glioma_1_mask.png" },
  { id:2, label:"Glioma · Case 2",     category:"Glioma",      mri:"/samples/Glioma_2_mri.png", mask:"/samples/Glioma_2_mask.png" },
  { id:3, label:"Meningioma · Case 1", category:"Meningioma",  mri:"/samples/Meningioma_1_mri.png", mask:"/samples/Meningioma_1_mask.png" },
  { id:4, label:"Meningioma · Case 2", category:"Meningioma",  mri:"/samples/Meningioma_2_mri.png", mask:"/samples/Meningioma_2_mask.png" },
  { id:5, label:"Pituitary · Case 1",  category:"Pituitary",   mri:"/samples/Pituitary_1_mri.png", mask:"/samples/Pituitary_1_mask.png" },
  { id:6, label:"Pituitary · Case 2",  category:"Pituitary",   mri:"/samples/Pituitary_2_mri.png", mask:"/samples/Pituitary_2_mask.png" },
];

const TRY_SAMPLES = [
  { id:1, label:"Sample 1", category:"Glioma",     filename:"sample_1_glioma.png",    src:"/sample-images/sample_1_glioma.png" },
  { id:2, label:"Sample 2", category:"Meningioma", filename:"sample_2_meningioma.png",src:"/sample-images/sample_2_meningioma.png" },
  { id:3, label:"Sample 3", category:"Pituitary",  filename:"sample_3_pituitary.png", src:"/sample-images/sample_3_pituitary.png" },
  { id:4, label:"Sample 4", category:"Pituitary",  filename:"sample_4_pituitary.png", src:"/sample-images/sample_4_pituitary.png" },
];

const BADGE = { Glioma:"badge-glioma", Meningioma:"badge-meningioma", Pituitary:"badge-pituitary" };

export default function App() {
  /* ── State ── */
  const [tab,        setTab]        = useState("upload");
  const [file,       setFile]       = useState(null);
  const [mriUrl,     setMriUrl]     = useState(null);
  const [maskUrl,    setMaskUrl]    = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState("");
  const [progress,   setProgress]   = useState(0);
  const [logs,       setLogs]       = useState(["[SYSTEM] NeuroMask API ready. Waiting for MRI input..."]);
  const [galleryFilter, setGalleryFilter] = useState("all");
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [modal,      setModal]      = useState(null);  // sample obj
  const [toast,      setToast]      = useState({ show:false, msg:"", type:"success" });
  const logsRef = useRef(null);
  const chartRef = useRef(null);
  const chartInst = useRef(null);
  const dragRef  = useRef(false);

  /* ── Logs auto-scroll ── */
  useEffect(() => { if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight; }, [logs]);

  /* ── Cleanup object URLs ── */
  useEffect(() => () => { if (mriUrl) URL.revokeObjectURL(mriUrl); }, [mriUrl]);
  useEffect(() => () => { if (maskUrl) URL.revokeObjectURL(maskUrl); }, [maskUrl]);

  /* ── Chart.js ── */
  useEffect(() => {
    if (!chartRef.current) return;
    if (chartInst.current) chartInst.current.destroy();
    chartInst.current = new window.Chart(chartRef.current, {
      type: "bar",
      data: {
        labels: ["Glioma", "Meningioma", "Pituitary"],
        datasets: [
          { label:"Dice",      data:[0.749,0.7403,0.8118], backgroundColor:"rgba(13,148,136,.7)",  borderColor:"#0d9488", borderWidth:1 },
          { label:"IoU",       data:[0.5988,0.5876,0.6833],backgroundColor:"rgba(14,165,233,.7)", borderColor:"#0ea5e9", borderWidth:1 },
          { label:"Precision", data:[0.7667,0.7422,0.7969],backgroundColor:"rgba(16,185,129,.7)", borderColor:"#10b981", borderWidth:1 },
          { label:"Recall",    data:[0.7321,0.7384,0.8273],backgroundColor:"rgba(167,139,250,.7)",borderColor:"#a78bfa", borderWidth:1 },
        ]
      },
      options: {
        responsive:true, maintainAspectRatio:false,
        plugins:{ legend:{ labels:{ color:"#94a3b8", font:{ family:"Inter,sans-serif", size:11 } } } },
        scales:{
          x:{ ticks:{ color:"#94a3b8" }, grid:{ color:"rgba(255,255,255,.03)" } },
          y:{ min:0, max:1, ticks:{ color:"#94a3b8", stepSize:.2 }, grid:{ color:"rgba(255,255,255,.03)" } }
        }
      }
    });
  }, []);

  /* ── Helpers ── */
  const addLog = (msg, cls="") => setLogs(p => [...p, { msg, cls }]);
  const showToast = (msg, type="success") => {
    setToast({ show:true, msg, type });
    setTimeout(() => setToast(p => ({ ...p, show:false })), 3500);
  };

  /* ── File handling ── */
  const handleFile = (f) => {
    if (!f || !["image/png","image/jpeg"].includes(f.type)) return;
    setSelectedSampleId(null);
    setFile(f);
    setMriUrl(URL.createObjectURL(f));
    setMaskUrl(null); setError("");
    addLog(`[INFO] File loaded: ${f.name}`, "log-info");
  };

  const clearFile = () => {
    setSelectedSampleId(null);
    setFile(null); setMriUrl(null); setMaskUrl(null); setError("");
  };

  /* ── Drag & Drop ── */
  const onDragOver = (e) => { e.preventDefault(); dragRef.current = true; };
  const onDragLeave = () => { dragRef.current = false; };
  const onDrop = (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); dragRef.current = false; };

  /* ── Unified Prediction Logic ── */
  const executePrediction = async (targetFile) => {
    if (!targetFile) return;
    setLoading(true); setError(""); setProgress(0);
    addLog("[INFO] Sending MRI to FastAPI backend...", "log-info");

    const iv = setInterval(() => setProgress(p => Math.min(p + Math.random()*8, 88)), 150);

    try {
      const form = new FormData(); form.append("image", targetFile);
      const res = await fetch(API_URL, { method:"POST", body:form });
      if (!res.ok) throw new Error((await res.text()) || "Prediction failed");
      const blob = await res.blob();
      setMaskUrl(URL.createObjectURL(blob));
      setProgress(100);
      addLog("[SUCCESS] Segmentation mask generated successfully.", "log-success");
      showToast("Mask generated successfully!", "success");
    } catch(err) {
      const msg = err.message.includes("fetch")
    ? "Our AI inference server is currently experiencing resource limitations. The service may be temporarily unavailable while the hosting environment recovers. Please try again shortly."
    : err.message;
      setError(msg);
      addLog(`[ERROR] ${msg}`, "log-warn");
      showToast("Prediction failed", "error");
    } finally {
      clearInterval(iv);
      setLoading(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const runPrediction = () => executePrediction(file);

  /* ── Try Sample MRI Handler ── */
  const handleSelectSample = async (sample) => {
    if (loading) return;
    setSelectedSampleId(sample.id);
    setError("");
    addLog(`[INFO] Selected sample: ${sample.label} (${sample.category} · ${sample.filename})`, "log-info");
    try {
      const response = await fetch(sample.src);
      if (!response.ok) throw new Error("Could not load sample image from server.");
      const blob = await response.blob();
      const sampleFile = new File([blob], sample.filename, { type: "image/png" });

      setFile(sampleFile);
      setMriUrl(URL.createObjectURL(sampleFile));
      setMaskUrl(null);
      addLog(`[INFO] File loaded: ${sample.filename}`, "log-info");

      await executePrediction(sampleFile);
    } catch (err) {
      const msg = err.message || "Failed to process sample image";
      setError(msg);
      addLog(`[ERROR] ${msg}`, "log-warn");
      showToast("Sample loading failed", "error");
    }
  };

  /* ── Gallery filter ── */
  const galleryItems = galleryFilter === "all" ? SAMPLES : SAMPLES.filter(s => s.category === galleryFilter);

  /* ── Render ── */
  return (
    <>
      <div className="grid-mesh" />
      <div className="glow-blob blob-top" />
      <div className="glow-blob blob-bottom" />

      {/* ── Navbar ── */}
      <nav className="navbar">
        <div className="navbar-container">
          <a href="#" className="navbar-logo">
            <div className="logo-icon"><i className="fas fa-brain" /></div>
            <span className="logo-name">NeuroMask</span>
          </a>
          <ul className="navbar-menu">
            <li><a href="#features" className="navbar-link">Capabilities</a></li>
            <li><a href="#demo"     className="navbar-link">Live Demo</a></li>
            <li><a href="#gallery"  className="navbar-link">Sample Results</a></li>
            <li><a href="#metrics"  className="navbar-link">ML Metrics</a></li>
            <li><a href="#faq"      className="navbar-link">FAQ</a></li>
          </ul>
          <a href="#demo" className="btn btn-primary btn-sm">
            <i className="fas fa-play" /> Try Live
          </a>
        </div>
      </nav>

      <div className="app-wrap">
        {/* ── Hero ── */}
        <section className="hero">
          <div className="hero-badge">
            <span className="hero-badge-dot" />
            <span>U-Net Model Active · Dice Score 0.76</span>
          </div>
          <h1 className="hero-title">Detecting Brain Tumors<br />From MRI Scans, Instantly</h1>
          <p className="hero-desc">
            A deep learning segmentation platform powered by a custom U-Net architecture trained on
            2,192 annotated MRI scans. Upload any brain MRI and receive a pixel-accurate tumor mask in real time.
          </p>
          <div className="hero-ctas">
            <a href="#demo"    className="btn btn-primary"><i className="fas fa-terminal" /> Try Live Workspace</a>
            <a href="#gallery" className="btn btn-secondary"><i className="fas fa-images" /> View Sample Results</a>
          </div>
        </section>

        {/* ── Features ── */}
        <section className="section-container" id="features">
          <div className="section-header">
            <span className="section-tag">System Features</span>
            <h2 className="section-title">Platform Capabilities</h2>
          </div>
          <div className="features-grid">
            {[
              { icon:"fa-upload",        name:"MRI Upload & Inference",       desc:"Upload any PNG or JPEG brain MRI scan and receive a 512 × 512 binary tumor segmentation mask in seconds via the FastAPI backend." },
              { icon:"fa-network-wired", name:"Custom U-Net Architecture",    desc:"5-level encoder–decoder U-Net with skip connections, BatchNorm, and ConvTranspose2d upsampling. Trained 50 epochs using combined BCE + Dice loss." },
              { icon:"fa-chart-bar",     name:"3-Category Coverage",          desc:"Trained across Glioma, Meningioma, and Pituitary tumor categories — achieving a best Dice score of 0.81 on the Pituitary class." },
            ].map(f => (
              <div className="feature-card" key={f.name}>
                <div className="feature-icon"><i className={`fas ${f.icon}`} /></div>
                <h4 className="feature-name">{f.name}</h4>
                <p className="feature-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Demo ── */}
        <section className="section-container" id="demo">
          <div className="section-header">
            <span className="section-tag">Interactive Environment</span>
            <h2 className="section-title">Live Segmentation Console</h2>
          </div>
          <div className="demo-panel">
            <div className="demo-tabs-bar">
              <button className={`demo-tab-btn ${tab==="upload"?"active":""}`} onClick={() => setTab("upload")}>
                <i className="fas fa-file-medical" /> MRI File Upload
              </button>
              <button className={`demo-tab-btn ${tab==="info"?"active":""}`} onClick={() => setTab("info")}>
                <i className="fas fa-info-circle" /> Pipeline Info
              </button>
            </div>
            <div className="demo-pane-body">

              {/* Upload pane */}
              {tab === "upload" && (
                <div className="workspace-grid">
                  {/* Left */}
                  <div className="workspace-card">
                    <div className="workspace-card-title"><i className="fas fa-file-import" /> MRI File Selection</div>

                    <label
                      className="dropzone-area"
                      onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
                    >
                      <div className="dropzone-icon"><i className="fas fa-cloud-upload-alt" /></div>
                      <h3>Drag &amp; Drop MRI File</h3>
                      <p>Supports PNG or JPEG · Max 10 MB</p>
                      <span className="btn btn-secondary btn-sm">Select File</span>
                      <input type="file" accept="image/png,image/jpeg" style={{display:"none"}}
                        onChange={e => handleFile(e.target.files?.[0])} />
                    </label>

                    {file && (
                      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between"}}>
                        <span style={{fontFamily:"var(--font-mono)",fontSize:".75rem",color:"var(--accent-teal)",wordBreak:"break-all"}}>{file.name}</span>
                        <button className="btn btn-secondary btn-sm" onClick={clearFile}><i className="fas fa-times" /></button>
                      </div>
                    )}

                    <button className="btn btn-primary" onClick={runPrediction} disabled={!file || loading} style={{width:"100%"}}>
                      {loading ? <><i className="fas fa-spinner fa-spin" /> Running Inference...</> : <><i className="fas fa-bolt" /> Generate Tumor Mask</>}
                    </button>

                    {error && <div className="error-bar"><i className="fas fa-exclamation-circle" /> {error}</div>}

                    {/* ── Try Sample MRI Scans ── */}
                    <div className="try-samples-section">
                      <div className="try-samples-header">
                        <div className="try-samples-title">
                          <i className="fas fa-images" /> Try Sample MRI Scans
                        </div>
                        <div className="try-samples-desc">
                          Don't have an MRI scan? Try one of our sample cases.
                        </div>
                      </div>
                      <div className="try-samples-grid">
                        {TRY_SAMPLES.map(sample => (
                          <div
                            key={sample.id}
                            className={`try-sample-card ${selectedSampleId === sample.id ? "active" : ""}`}
                            onClick={() => handleSelectSample(sample)}
                            title={`Click to analyze ${sample.label} (${sample.category})`}
                          >
                            <div className="try-sample-thumb">
                              <img src={sample.src} alt={`${sample.label} (${sample.category})`} />
                            </div>
                            <div className="try-sample-info">
                              <span className={`try-sample-badge ${BADGE[sample.category]}`}>
                                {sample.category}
                              </span>
                              <div className="try-sample-label">{sample.label}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <div className="console-header-bar">
                        <span className="console-tab-tag"><i className="fas fa-terminal" /> System Logs</span>
                        <span style={{fontSize:".6rem",color:"var(--text-muted)",fontFamily:"var(--font-mono)"}}>STDOUT</span>
                      </div>
                      <div className="logs-box" ref={logsRef}>
                        {logs.map((l, i) => (
                          <div key={i} className={typeof l === "object" ? l.cls : ""}>{typeof l === "object" ? l.msg : l}</div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Right */}
                  <div className="workspace-card">
                    <div className="workspace-card-title"><i className="fas fa-tachometer-alt" /> Segmentation Output</div>

                    {maskUrl && (
                      <div className="success-bar">
                        <span><i className="fas fa-check-circle" /> Segmentation generated successfully</span>
                        <a href={maskUrl} download="neuromask_mask.png" className="btn btn-secondary btn-sm">
                          <i className="fas fa-download" /> Download
                        </a>
                      </div>
                    )}

                    <div className="result-panels">
                      <div className="image-panel-dark">
                        <div className="panel-label">Original MRI</div>
                        {mriUrl
                          ? <img src={mriUrl} alt="MRI" />
                          : <div className="preview-placeholder"><i className="fas fa-brain" style={{fontSize:"1.75rem"}} /><span>MRI preview here</span></div>
                        }
                      </div>
                      <div className="image-panel-dark">
                        <div className="panel-label">Predicted Tumor Mask</div>
                        {maskUrl
                          ? <img src={maskUrl} alt="Mask" />
                          : <div className="preview-placeholder"><i className="fas fa-mask" style={{fontSize:"1.75rem"}} /><span>Mask output here</span></div>
                        }
                      </div>
                    </div>

                    {loading && (
                      <div>
                        <div style={{fontSize:".7rem",color:"var(--text-muted)",fontFamily:"var(--font-display)",marginBottom:".35rem"}}>Running U-Net inference...</div>
                        <div className="progress-meter"><div className="progress-fill" style={{width:`${progress}%`}} /></div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Pipeline pane */}
              {tab === "info" && (
                <div className="workspace-grid">
                  <div className="workspace-card">
                    <div className="workspace-card-title"><i className="fas fa-route" /> Inference Pipeline</div>
                    {[
                      ["Upload",           "PNG/JPEG brain MRI (≤ 10 MB) validated for format & size"],
                      ["Preprocess",       "Grayscale → resize 512 × 512 → min-max normalize → PyTorch tensor [1,1,512,512]"],
                      ["U-Net Forward",    "5-level encoder → bottleneck → decoder with skip connections"],
                      ["Post-process",     "Sigmoid → threshold @ 0.5 → binary mask (0/255) uint8"],
                      ["Response",         "512 × 512 PNG mask streamed back; downloadable instantly"],
                    ].map(([step,desc],i) => (
                      <div className="pipeline-step" key={i}>
                        <div className="step-num">{i+1}</div>
                        <div><strong style={{color:"#fff"}}>{step}</strong> — {desc}</div>
                      </div>
                    ))}
                  </div>
                  <div className="workspace-card">
                    <div className="workspace-card-title"><i className="fas fa-code-branch" /> API Endpoints</div>
                    {[
                      { method:"GET",  cls:"method-get",  path:"/health",      desc:"Verify backend is running and model is loaded" },
                      { method:"POST", cls:"method-post", path:"/api/predict", desc:"Upload MRI → returns segmentation mask PNG" },
                      { method:"GET",  cls:"method-get",  path:"/docs",        desc:"Interactive Swagger UI for API exploration" },
                    ].map(ep => (
                      <div className="api-card" key={ep.path} style={{marginBottom:".65rem"}}>
                        <div style={{display:"flex",alignItems:"center",gap:".65rem"}}>
                          <span className={`method-badge ${ep.cls}`}>{ep.method}</span>
                          <span style={{fontFamily:"var(--font-mono)",fontSize:".8rem",color:"#fff"}}>{ep.path}</span>
                        </div>
                        <p style={{fontSize:".75rem",color:"var(--text-secondary)",marginTop:".35rem"}}>{ep.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── Gallery ── */}
        <section className="section-container" id="gallery">
          <div className="section-header">
            <span className="section-tag">Model Outputs</span>
            <h2 className="section-title">Sample Prediction Results</h2>
            <p className="section-desc">Click any case to inspect side-by-side: Original MRI (Left) vs. Predicted Tumor Mask (Right)</p>
          </div>
          <div className="gallery-filter-row">
            <div className="filter-chips">
              {["all","Glioma","Meningioma","Pituitary"].map(f => (
                <span key={f} className={`filter-chip ${galleryFilter===f?"active":""}`}
                  onClick={() => setGalleryFilter(f)}>
                  {f === "all" ? "All Cases" : f}
                </span>
              ))}
            </div>
          </div>
          <div className="gallery-grid">
            {galleryItems.map(s => (
              <div className="gallery-card" key={s.id} onClick={() => setModal(s)}>
                <div className="gallery-card-images">
                  <div className="gallery-img-box">
                    <span className="mini-tag">MRI</span>
                    <img src={s.mri} alt={`${s.label} MRI`} />
                  </div>
                  <div className="gallery-img-box">
                    <span className="mini-tag tag-pred">PREDICTED</span>
                    <img src={s.mask} alt={`${s.label} Predicted Mask`} />
                  </div>
                </div>
                <div className="gallery-card-body">
                  <span className={`gallery-badge ${BADGE[s.category]}`}>{s.category}</span>
                  <div className="gallery-card-label">{s.label}</div>
                  <div className="gallery-card-cta">Click to expand side-by-side →</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Metrics ── */}
        <section className="section-container" id="metrics">
          <div className="section-header">
            <span className="section-tag">Model Diagnostics</span>
            <h2 className="section-title">U-Net Classifier Telemetry</h2>
            <p className="section-desc">Architecture specifications and training diagnostics for the NeuroMask segmentation engine.</p>
          </div>
          <div className="telemetry-layout">
            <div className="specs-column">
              {/* Specs */}
              <div className="spec-card">
                <div className="spec-card-header"><i className="fas fa-microchip spec-icon" /><span className="spec-title">Model Specifications</span></div>
                <div className="spec-list">
                  {[
                    ["Architecture","Custom U-Net"],["Encoder Depth","5 levels"],
                    ["Input Shape","[B, 1, 512, 512]"],["Output Shape","[B, 1, 512, 512]"],
                    ["Loss Function","BCE + Dice"],["Epochs Trained","50"],
                    ["Training Device","Colab GPU"],["Threshold","0.5 (sigmoid)"],
                  ].map(([l,v]) => (
                    <div className="spec-row" key={l}>
                      <span className="spec-label">{l}</span><span className="spec-value">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
              {/* Bars */}
              <div className="spec-card">
                <div className="spec-card-header"><i className="fas fa-chart-bar spec-icon" /><span className="spec-title">Category Dice Scores</span></div>
                <div className="bar-list">
                  {[
                    { label:"Pituitary Tumor", val:0.8118, pct:"81.18%", cls:"fill-green",  color:"var(--success)" },
                    { label:"Glioma",          val:0.7490, pct:"74.90%", cls:"fill-teal",   color:"var(--accent-teal)" },
                    { label:"Meningioma",      val:0.7403, pct:"74.03%", cls:"fill-sky",    color:"var(--accent-sky)" },
                    { label:"Overall (329)",   val:0.7642, pct:"76.42%", cls:"fill-teal",   color:"#fff" },
                  ].map(b => (
                    <div className="bar-item" key={b.label}>
                      <div className="bar-label-row">
                        <span>{b.label}</span><span style={{color:b.color}}>{b.val}</span>
                      </div>
                      <div className="bar-track"><div className={`bar-fill ${b.cls}`} style={{width:b.pct}} /></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {/* Chart */}
            <div className="chart-card">
              <div className="chart-header">
                <div>
                  <div style={{fontFamily:"var(--font-display)",fontWeight:700,fontSize:".9rem",color:"#fff"}}>Category Performance</div>
                  <div style={{fontSize:".75rem",color:"var(--text-secondary)",marginTop:".2rem"}}>Dice · IoU · Precision · Recall</div>
                </div>
                <div style={{display:"flex",alignItems:"center",gap:".5rem"}}>
                  <span className="status-dot" /><span style={{fontSize:".7rem",fontFamily:"var(--font-mono)",color:"var(--text-muted)"}}>Test Set</span>
                </div>
              </div>
              <div className="chart-canvas-wrap"><canvas ref={chartRef} /></div>
              <div className="metric-mini-cards">
                {[
                  { val:"0.76", label:"Dice",      color:"var(--accent-teal)" },
                  { val:"0.62", label:"IoU",       color:"var(--accent-sky)"  },
                  { val:"0.77", label:"Precision", color:"var(--success)"     },
                  { val:"0.76", label:"Recall",    color:"#a78bfa"            },
                ].map(m => (
                  <div className="metric-mini" key={m.label}>
                    <div className="metric-mini-val" style={{color:m.color}}>{m.val}</div>
                    <div className="metric-mini-label">{m.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ── FAQ ── */}
        <section className="section-container" id="faq">
          <div className="section-header">
            <span className="section-tag">Help Center</span>
            <h2 className="section-title">Frequently Asked Questions</h2>
          </div>
          <div className="faq-list">
            {[
              ["What tumor types can NeuroMask segment?",
               "NeuroMask is trained on three primary brain tumor categories: Glioma, Meningioma, and Pituitary Tumor. The model produces a binary mask (tumor vs. background) regardless of category."],
              ["What image format and size does the API accept?",
               "The API accepts PNG and JPEG images up to 10 MB. The image is internally resized to 512 × 512 during preprocessing, so any input resolution is supported."],
              ["How was the U-Net trained?",
               "The model was trained for 50 epochs on a Google Colab GPU using a custom PyTorch implementation. Training used a combined BCE + Dice loss on 2,192 annotated 512 × 512 MRI images split into train/validation/test sets."],
              ["How do I run the backend locally?",
               'Activate the venv, then run: python -m uvicorn app.main:app --app-dir backend --reload — The API will be at http://127.0.0.1:8000'],
              ["What does the output mask pixel value represent?",
               "Pixel value 0 = background (no tumor). Pixel value 255 = predicted tumor region. The mask is returned as a 512 × 512 grayscale PNG."],
            ].map(([q,a]) => (
              <details className="faq-item" key={q}>
                <summary className="faq-summary">{q}</summary>
                <div className="faq-body">{a}</div>
              </details>
            ))}
          </div>
        </section>

        {/* ── Footer ── */}
        <footer className="footer">
          <div className="footer-grid">
            <div className="footer-col">
              <div style={{display:"flex",alignItems:"center",gap:".65rem",marginBottom:"1rem"}}>
                <div style={{width:28,height:28,borderRadius:6,background:"linear-gradient(135deg,#0d9488,#0ea5e9)",display:"flex",alignItems:"center",justifyContent:"center",color:"#fff",fontSize:".9rem"}}><i className="fas fa-brain"/></div>
                <span style={{fontFamily:"var(--font-display)",fontWeight:700,fontSize:"1.1rem",color:"#fff"}}>NeuroMask</span>
              </div>
              <p className="footer-desc">AI-powered brain MRI tumor segmentation using a custom U-Net architecture. Upload. Predict. Download.</p>
              <div className="footer-socials">
                <a href="https://github.com/bipulsinghaniya" target="_blank" rel="noopener noreferrer" className="footer-social-btn"><i className="fab fa-github"/></a>
                <a href="https://www.linkedin.com/in/bipul-singhaniya/" target="_blank" rel="noopener noreferrer" className="footer-social-btn"><i className="fab fa-linkedin"/></a>
              </div>
            </div>
            <div className="footer-col">
              <h4>Navigation</h4>
              <ul className="footer-links">
                {[["Capabilities","#features"],["Live Demo","#demo"],["Sample Results","#gallery"],["ML Metrics","#metrics"],["FAQ","#faq"]].map(([l,h]) => (
                  <li key={l}><a href={h}>{l}</a></li>
                ))}
              </ul>
            </div>
            <div className="footer-col">
              <h4>Tech Stack</h4>
              {[["fa-fire","PyTorch"],["fa-bolt","FastAPI"],["fab fa-react","React 19"],["fa-cube","U-Net"],["fa-server","Uvicorn"],["fa-image","Pillow"]].map(([icon,name]) => (
                <span className="tech-pill" key={name}><i className={`${icon.startsWith("fab")?"fab ":"fas "}${icon}`}/>{name}</span>
              ))}
            </div>
            <div className="footer-col">
              <h4>Creator</h4>
              <div className="team-card">
                <div className="team-row">
                  <div className="team-avatar"><i className="fas fa-user"/></div>
                  <div>
                    <div className="team-name">Bipul Singhaniya</div>
                    <div className="team-role">ML Engineer · Full-Stack Developer</div>
                  </div>
                </div>
              </div>
              <p style={{fontSize:".8rem",color:"var(--text-secondary)",marginTop:".85rem",lineHeight:1.5}}>
                Dataset: <a href="https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification" style={{color:"var(--accent-teal)",textDecoration:"none"}}>Kaggle — Indrakumar K</a>
              </p>
            </div>
          </div>
          <div className="footer-bottom">
            <span>© 2026 NeuroMask · AI-Powered Brain MRI Tumor Segmentation</span>
            <span>Built by <strong style={{color:"var(--accent-teal)"}}>Bipul Singhaniya</strong></span>
          </div>
        </footer>
      </div>

      {/* ── Sample Modal ── */}
      {modal && (
        <div className="modal-overlay active" onClick={e => { if(e.target===e.currentTarget) setModal(null); }}>
          <div className="modal-body">
            <div className="modal-top-bar">
              <span className="modal-heading">{modal.label} · Segmentation Output</span>
              <button className="modal-close" onClick={() => setModal(null)}><i className="fas fa-times"/></button>
            </div>
            <div className="modal-panels">
              <div className="modal-panel-box">
                <div className="panel-label"><i className="fas fa-brain" /> Original Brain MRI Scan</div>
                <img className="modal-panel-img" src={modal.mri} alt={`${modal.label} Original MRI`} />
              </div>
              <div className="modal-panel-box">
                <div className="panel-label"><i className="fas fa-mask" style={{color:"var(--accent-teal)"}} /> Predicted Tumor Mask (U-Net)</div>
                <img className="modal-panel-img" src={modal.mask} alt={`${modal.label} Predicted Mask`} />
              </div>
            </div>
            <p className="modal-caption">Left: Original Brain MRI Scan &nbsp;·&nbsp; Right: Predicted Tumor Mask (Pixel 255 = Tumor, 0 = Background)</p>
          </div>
        </div>
      )}

      {/* ── Toast ── */}
      <div className={`toast ${toast.type} ${toast.show?"show":""}`}>
        <i className={toast.type==="success"?"fas fa-check-circle":"fas fa-exclamation-circle"} />
        <span>{toast.msg}</span>
      </div>
    </>
  );
}
