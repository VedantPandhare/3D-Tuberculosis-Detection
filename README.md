# 🏥 AI-Powered TB Chest X-ray Analyzer

A modern, high-precision medical imaging application designed to assist healthcare professionals in the detection and analysis of Tuberculosis (TB) from chest X-ray scans. This project leverages a hybrid architecture combining a powerful **FastAPI** backend for machine learning and a premium **Next.js 16** frontend for a "classy" and interactive user experience.

---

## ✨ Features

### 🔍 Precision Diagnostics
- **Neural Scan**: Utilizes a deep learning Convolutional Neural Network (CNN) to classify chest X-rays as "Normal" or "TB Detected".
- **Risk Profiling**: Provides confidence scores and risk levels (Low/Moderate/High) for each analysis.

### 🎨 Explainable AI (XAI)
- **Grad-CAM Saliency Maps**: Intuitive heatmaps that highlight the specific regions of the lungs that influenced the AI's decision.
- **Volumetric 3D Visualization**: Interactive 3D surface plots of X-ray intensities, allowing for detailed spatial inspection of lung density.

### 📊 Comprehensive Analytics
- **Multi-View Comparison**: Synchronized views of raw grayscale scans, edge-detected features, and saliency overlays.
- **Statistical Breakdown**: Real-time computation of pixel intensity distributions and statistical markers (Mean, Std Dev, Max/Min).

### 📝 Clinical Workflow
- **Integrated Reporting**: Add clinical observations and generate professional diagnostic reports (text format).
- **Interactive Background**: A custom "Ripple Effect" background provides a modern, state-of-the-art diagnostic environment.
- **Responsive Settings**: Slide-in settings panel to control heatmap intensity and toggle analytical tools.

---

## 🏗️ Technical Architecture

The application is split into two main components:

- **Frontend**: Built with **Next.js 16**, **TypeScript**, **Tailwind CSS 4**, and **Framer Motion**. It provides a high-performance, interactive dashboard.
- **Backend**: Built with **FastAPI**, **TensorFlow**, and **tf-keras-vis**. It handles model inference, image processing, and 3D data generation.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- NPM / Yarn

### 1. Installation

Clone the repository and install dependencies for both the backend and frontend.

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd frontend
npm install
```

### 2. Running the Application

You need to run both the backend and frontend servers simultaneously.

#### Start Backend (FastAPI)
```bash
cd backend
python main.py
```
*The API will be available at `http://localhost:8000`*

#### Start Frontend (Next.js)
```bash
cd frontend
npm run dev
```
*The application will be accessible at `http://localhost:3000`*

---

## 🔬 Model Information

- **Architecture**: Custom CNN optimized for radiography data.
- **Training Source**: TB Chest Radiography Database.
- **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping).

---

## ⚠️ Disclaimer

**This application is for educational and research purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified physician or healthcare provider for medical concerns.

---

## 👨‍💻 Developed By

Developed with ❤️ for the medical community using modern AI and Web technologies.