from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from PIL import Image
import base64
from io import BytesIO
import json
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tb_model.h5")
try:
    model = load_model(MODEL_PATH)
    gradcam_engine = Gradcam(model, model_modifier=ReplaceToLinear())
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def compute_statistics(image):
    return {
        "Mean": float(np.mean(image)),
        "Std Dev": float(np.std(image)),
        "Min": float(np.min(image)),
        "Max": float(np.max(image)),
        "Median": float(np.median(image))
    }

def create_3d_json(z_data, title, colorscale='Viridis'):
    z_smooth = cv2.bilateralFilter(z_data.astype(np.float32), 9, 75, 75)
    z = z_smooth
    x = np.linspace(0, 1, z.shape[1])
    y = np.linspace(0, 1, z.shape[0])
    x, y = np.meshgrid(x, y)
    
    fig = go.Figure(data=[go.Surface(
        z=z, x=x, y=y, 
        colorscale=colorscale,
        lighting=dict(roughness=0.5, specular=0.05, ambient=0.6, diffuse=0.9),
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(showgrid=False, showbackground=False, visible=False),
            yaxis=dict(showgrid=False, showbackground=False, visible=False),
            zaxis=dict(range=[0, 255], showgrid=True, showbackground=False),
            aspectratio=dict(x=1, y=1, z=0.4),
        ),
        margin=dict(r=0, l=0, b=0, t=40),
        height=500
    )
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()
    image_pil = Image.open(BytesIO(contents)).convert("RGB")
    img = np.array(image_pil)
    img_resized = cv2.resize(img, (224, 224))

    # Prediction
    processed_img = preprocess_image(img_resized)
    pred = model.predict(processed_img, verbose=0)
    class_names = ['Normal', 'TB Detected']
    predicted_class = class_names[np.argmax(pred)]
    confidence = round(100 * float(np.max(pred)), 2)
    risk_level = "HIGH" if predicted_class == "TB Detected" and confidence > 80 else "MODERATE" if predicted_class == "TB Detected" else "LOW"

    # GradCAM
    cam = gradcam_engine(CategoricalScore([np.argmax(pred)]), processed_img, penultimate_layer=-1)
    heatmap = cam[0]
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap_smooth = cv2.GaussianBlur(heatmap_norm, (0, 0), sigmaX=3, sigmaY=3)
    heatmap_resized = cv2.resize(heatmap_smooth, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    heatmap_2d = np.uint8(255 * heatmap_resized)

    # Overlay
    colormap_cv = cv2.COLORMAP_JET # Default from streamlit
    heatmap_color = cv2.applyColorMap(heatmap_2d, colormap_cv)
    overlayed_img = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

    # Convert images to base64
    _, buffer_orig = cv2.imencode('.jpg', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    base64_orig = base64.b64encode(buffer_orig).decode('utf-8')

    _, buffer_overlay = cv2.imencode('.jpg', cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))
    base64_overlay = base64.b64encode(buffer_overlay).decode('utf-8')

    # Edge detection
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    _, buffer_edges = cv2.imencode('.jpg', edges)
    base64_edges = base64.b64encode(buffer_edges).decode('utf-8')

    # 3D Data
    # Streamlit app used (500, 500) for 3D
    img_3d_res = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LANCZOS4)
    gray_3d = cv2.cvtColor(img_3d_res, cv2.COLOR_RGB2GRAY)
    
    overlay_3d = cv2.resize(overlayed_img, (500, 500), interpolation=cv2.INTER_CUBIC)
    overlay_gray_3d = cv2.cvtColor(overlay_3d, cv2.COLOR_BGR2GRAY)

    fig1_json = create_3d_json(gray_3d, "3D Original X-ray", "Greys")
    fig2_json = create_3d_json(overlay_gray_3d, "3D Heatmap Overlay", "Hot")

    # Stats
    stats_orig = compute_statistics(gray)
    stats_heat = compute_statistics(heatmap_2d)

    return {
        "results": {
            "prediction": predicted_class,
            "confidence": confidence,
            "risk_level": risk_level
        },
        "images": {
            "original": f"data:image/jpeg;base64,{base64_orig}",
            "overlay": f"data:image/jpeg;base64,{base64_overlay}",
            "edges": f"data:image/jpeg;base64,{base64_edges}"
        },
        "visualizations": {
            "plot3d_original": fig1_json,
            "plot3d_overlay": fig2_json
        },
        "statistics": {
            "original": stats_orig,
            "heatmap": stats_heat
        },
        "metadata": {
            "filename": file.filename,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
