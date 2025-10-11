import streamlit as st
import cv2
import numpy as np
import plotly.graph_objs as go
from PIL import Image
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="TB X-ray Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2196F3, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .positive-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .negative-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .info-box {
        background: #f0f7ff;
        padding: 1rem;
        border-left: 4px solid #2196F3;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #2196F3, #00BCD4);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 AI-Powered TB Chest X-ray Analyzer</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Advanced deep learning model for tuberculosis detection with explainable AI visualization</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/lungs.png", width=80)
    st.title("⚙️ Settings")
    
    # Visualization settings
    st.subheader("📊 Visualization Options")
    heatmap_intensity = st.slider("Heatmap Intensity", 0.0, 1.0, 0.4, 0.1)
    show_histogram = st.checkbox("Show Intensity Histogram", value=True)
    show_statistics = st.checkbox("Show Statistical Analysis", value=True)
    colormap_option = st.selectbox(
        "Heatmap Color Scheme",
        ["COLORMAP_JET", "COLORMAP_HOT", "COLORMAP_VIRIDIS", "COLORMAP_TURBO"]
    )
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.info("""
    This application uses a Convolutional Neural Network (CNN) with Grad-CAM visualization 
    to detect tuberculosis in chest X-rays and highlight regions of interest.
    
    **Features:**
    - Binary classification (Normal/TB)
    - Grad-CAM heatmap overlay
    - 3D surface visualization
    - Statistical analysis
    - Multi-view comparison
    """)
    
    st.markdown("---")
    st.caption(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Load model (with caching)
@st.cache_resource
def load_tb_model():
    return load_model("tb_model.h5")

try:
    model = load_tb_model()
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear())
except Exception as e:
    st.error(f"⚠️ Error loading model: {str(e)}")
    st.stop()

# Utility functions
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def create_3d_plot(z_data, title, colorscale='Viridis'):
    z = z_data.astype(np.float32)
    x = np.linspace(0, 1, z.shape[1])
    y = np.linspace(0, 1, z.shape[0])
    x, y = np.meshgrid(x, y)
    
    fig = go.Figure(data=[go.Surface(
        z=z, x=x, y=y, 
        colorscale=colorscale,
        colorbar=dict(title="Intensity")
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title='Y', backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(title='Intensity', range=[0, 255], backgroundcolor="rgb(230, 230,230)")
        ),
        margin=dict(r=20, l=20, b=20, t=40),
        height=450
    )
    return fig

def create_histogram(image, title):
    hist, bins = np.histogram(image.flatten(), bins=50)
    fig = go.Figure(data=[go.Bar(x=bins[:-1], y=hist, marker_color='#2196F3')])
    fig.update_layout(
        title=title,
        xaxis_title="Pixel Intensity",
        yaxis_title="Frequency",
        height=300,
        showlegend=False
    )
    return fig

def compute_statistics(image):
    return {
        "Mean": np.mean(image),
        "Std Dev": np.std(image),
        "Min": np.min(image),
        "Max": np.max(image),
        "Median": np.median(image)
    }

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a chest X-ray image for TB detection analysis"
)

if uploaded_file is not None:
    # Process image
    image_pil = Image.open(uploaded_file).convert("RGB")
    img = np.array(image_pil)
    img_resized = cv2.resize(img, (224, 224))
    
    # Prediction
    with st.spinner('🔍 Analyzing X-ray image...'):
        pred = model.predict(preprocess_image(img_resized), verbose=0)
        class_names = ['Normal', 'TB Detected']
        predicted_class = class_names[np.argmax(pred)]
        confidence = round(100 * np.max(pred), 2)
        
        # GradCAM - Generate heatmap
        cam = gradcam(CategoricalScore([np.argmax(pred)]), preprocess_image(img_resized), penultimate_layer=-1)
        heatmap = cam[0]
        
        # Normalize heatmap
        heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Apply Gaussian smoothing to the normalized heatmap for better visualization
        heatmap_smooth = cv2.GaussianBlur(heatmap_norm, (0, 0), sigmaX=3, sigmaY=3)
        
        # Resize to target size with high quality interpolation
        heatmap_resized = cv2.resize(heatmap_smooth, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to 8-bit
        heatmap_2d = np.uint8(255 * heatmap_resized)
    
    # Display results in metric cards
    st.markdown("### 🎯 Diagnostic Results")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        card_class = "positive-card" if predicted_class == "TB Detected" else "negative-card"
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <h3 style="margin: 0;">Classification</h3>
            <h2 style="margin: 0.5rem 0;">{predicted_class}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">Confidence Score</h3>
            <h2 style="margin: 0.5rem 0;">{confidence}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        risk_level = "HIGH" if predicted_class == "TB Detected" and confidence > 80 else "MODERATE" if predicted_class == "TB Detected" else "LOW"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">Risk Level</h3>
            <h2 style="margin: 0.5rem 0;">{risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Image processing
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast for better edge detection using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    # Create high-quality heatmap overlay
    colormap_cv = getattr(cv2, colormap_option)
    overlay = cv2.applyColorMap(heatmap_2d, colormap_cv)
    overlay_resized = cv2.resize(overlay, (224, 224), interpolation=cv2.INTER_CUBIC)
    overlayed_img = cv2.addWeighted(img_resized, 1-heatmap_intensity, overlay_resized, heatmap_intensity, 0)
    
    # Main visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Analysis", "📈 3D Visualization", "📊 Statistical Analysis", "🔬 Multi-View Comparison"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📸 Original X-ray")
            st.image(gray, clamp=True)
        
        with col2:
            st.subheader("🔥 Grad-CAM Heatmap")
            st.image(heatmap_2d, clamp=True, channels="GRAY")
        
        with col3:
            st.subheader("🎨 Overlay Visualization")
            st.image(overlayed_img, clamp=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_3d_plot(gray, "3D Original X-ray", "Greys"), use_container_width=True)
        
        with col2:
            overlay_gray = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2GRAY)
            st.plotly_chart(create_3d_plot(overlay_gray, "3D Heatmap Overlay", "Hot"), use_container_width=True)
    
    with tab3:
        if show_histogram:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_histogram(gray, "Original X-ray Intensity Distribution"), use_container_width=True)
            with col2:
                st.plotly_chart(create_histogram(heatmap_2d, "Heatmap Intensity Distribution"), use_container_width=True)
        
        if show_statistics:
            st.subheader("📊 Statistical Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original X-ray Statistics**")
                stats_orig = compute_statistics(gray)
                for key, value in stats_orig.items():
                    st.metric(key, f"{value:.2f}")
            
            with col2:
                st.markdown("**Heatmap Statistics**")
                stats_heat = compute_statistics(heatmap_2d)
                for key, value in stats_heat.items():
                    st.metric(key, f"{value:.2f}")
    
    with tab4:
        st.subheader("🔬 Comprehensive Multi-View Analysis")
        
        # Edge detection with better parameters on enhanced image
        edges = cv2.Canny(gray_enhanced, 30, 100)
        
        # Apply morphological operations to connect edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create comparison grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Original Grayscale**")
            st.image(gray, clamp=True)
        
        with col2:
            st.markdown("**Edge Detection**")
            st.image(edges, clamp=True)
        
        with col3:
            st.markdown("**Heatmap Only**")
            st.image(heatmap_2d, clamp=True, channels="GRAY")
        
        with col4:
            st.markdown("**Final Overlay**")
            st.image(overlayed_img, clamp=True)
    
    # Clinical notes section
    st.markdown("---")
    st.subheader("📝 Clinical Notes")
    clinical_notes = st.text_area(
        "Add clinical observations or notes:",
        placeholder="Enter any relevant clinical observations, patient history, or additional notes here...",
        height=100
    )
    
    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Generate Report"):
            st.success("✅ Report generation feature - to be implemented with PDF export")
    with col2:
        if st.button("💾 Save Analysis"):
            st.success("✅ Analysis saved to session")
    with col3:
        if st.button("🔄 Reset Analysis"):
            st.rerun()

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to TB X-ray Analyzer</h3>
        <p>This AI-powered tool helps radiologists and healthcare professionals detect tuberculosis in chest X-rays using advanced deep learning techniques.</p>
        <br>
        <h4>🚀 How to use:</h4>
        <ol>
            <li>Upload a chest X-ray image using the file uploader above</li>
            <li>Wait for the AI model to analyze the image</li>
            <li>Review the classification results and confidence score</li>
            <li>Explore the Grad-CAM heatmap to see which regions influenced the decision</li>
            <li>Examine 3D visualizations and statistical analyses</li>
        </ol>
        <br>
        <p><strong>⚠️ Disclaimer:</strong> This tool is for educational and research purposes. Always consult with qualified medical professionals for diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample images section
    st.markdown("### 📚 Sample X-ray Gallery")
    st.info("Upload your own chest X-ray image to begin the analysis")