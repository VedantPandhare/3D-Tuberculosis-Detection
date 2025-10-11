import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from scipy.ndimage import gaussian_filter

st.set_page_config(layout="wide")
st.title("🧠 3D Grad-CAM TB Visualizer")

# Load model
@st.cache_resource
def load_tb_model():
    return load_model("tb_model.h5")

model = load_tb_model()
gradcam = Gradcam(model, model_modifier=ReplaceToLinear())
score = CategoricalScore([1])  # Class 1: TB

# Preprocess image
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.
    return np.expand_dims(image, axis=0)

# Create smooth 3D surface
def create_smooth_surface(image, is_heatmap=False):
    # Convert to grayscale if not heatmap
    if not is_heatmap:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Resize for smoother 3D visualization
    resized = cv2.resize(gray, (200, 200))
    
    # Apply Gaussian filter for smoothing
    smoothed = gaussian_filter(resized, sigma=1.0)
    
    # Create meshgrid
    x = np.linspace(0, 1, smoothed.shape[1])
    y = np.linspace(0, 1, smoothed.shape[0])
    x, y = np.meshgrid(x, y)
    
    return x, y, smoothed

# Upload image
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Convert to array
    pil_img = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(pil_img)
    st.image(original_np, caption="🩻 Uploaded X-ray", use_column_width=True)
    
    # Preprocess and get prediction
    input_tensor = preprocess(original_np)
    prediction = model.predict(input_tensor)[0][0]
    tb_detected = prediction > 0.5
    
    # Display prediction result
    st.subheader("🔍 Prediction Result")
    if tb_detected:
        st.error(f"TB Detected with {prediction:.2%} confidence")
    else:
        st.success(f"No TB Detected (Normal) with {1-prediction:.2%} confidence")
    
    # Create 3D visualization of original image
    st.subheader("🫁 3D Visualization of Original X-ray")
    x_orig, y_orig, z_orig = create_smooth_surface(original_np, is_heatmap=False)
    
    fig_orig = go.Figure(data=[go.Surface(
        z=z_orig, 
        x=x_orig, 
        y=y_orig, 
        colorscale='Gray',
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5),
        lightposition=dict(x=0, y=0, z=1)
    )])
    
    fig_orig.update_layout(
        title='3D Visualization of Original X-ray',
        scene=dict(
            zaxis=dict(title='Intensity'),
            xaxis=dict(title='Width'),
            yaxis=dict(title='Height'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ),
        width=900,
        height=500,
        margin=dict(r=20, l=20, b=20, t=40),
    )
    
    st.plotly_chart(fig_orig, use_container_width=True)
    
    # If TB is detected, show heatmap visualization
    if tb_detected:
        st.subheader("🔥 3D Grad-CAM Heatmap with TB Regions Highlighted")
        
        # Generate Grad-CAM
        cam = gradcam(score, input_tensor, penultimate_layer=-1)
        heatmap = np.uint8(255 * cam[0])
        
        # Create smooth 3D heatmap surface
        x_heat, y_heat, z_heat = create_smooth_surface(heatmap, is_heatmap=True)
        
        # Create figure with TB regions highlighted in red
        fig_heat = go.Figure()
        
        # Add base surface (blue to yellow gradient)
        fig_heat.add_trace(go.Surface(
            z=z_heat, 
            x=x_heat, 
            y=y_heat, 
            colorscale=[[0, 'blue'], [0.5, 'yellow'], [1, 'red']],
            opacity=0.8,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5),
            lightposition=dict(x=0, y=0, z=1),
            showscale=True,
            colorbar=dict(title="Activation")
        ))
        
        # Add TB regions as red highlights
        tb_threshold = np.percentile(z_heat, 85)  # Top 15% activations
        tb_mask = z_heat > tb_threshold
        
        if np.any(tb_mask):
            fig_heat.add_trace(go.Surface(
                z=np.where(tb_mask, z_heat, None),
                x=x_heat,
                y=y_heat,
                colorscale=[[0, 'red'], [1, 'darkred']],
                opacity=0.9,
                showscale=False
            ))
        
        fig_heat.update_layout(
            title='3D Grad-CAM: TB Regions Highlighted in Red',
            scene=dict(
                zaxis=dict(title='Activation', range=[0, 255]),
                xaxis=dict(title='Width'),
                yaxis=dict(title='Height'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
            ),
            width=900,
            height=500,
            margin=dict(r=20, l=20, b=20, t=40),
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Add interpretation guide
        st.markdown("""
        **Interpretation Guide:**
        - **Red areas**: Regions most indicative of TB (high activation)
        - **Yellow areas**: Moderately indicative regions
        - **Blue areas**: Less relevant regions
        """)
else:
    st.info("Upload a chest X-ray to see 3D TB heatmap visualization.")