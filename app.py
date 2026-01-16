from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model once
model = load_model("tb_model.h5")
gradcam = Gradcam(model, model_modifier=ReplaceToLinear())

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tb', methods=['GET', 'POST'])
def tb():
    if request.method == 'POST':
        file = request.files['xray']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")
        img = np.array(img)

        pred = model.predict(preprocess_image(img))
        class_names = ['Normal', 'TB Detected']
        predicted_class = class_names[np.argmax(pred)]
        confidence = round(float(np.max(pred) * 100), 2)

        # Grad-CAM
        cam = gradcam(
            CategoricalScore([np.argmax(pred)]),
            preprocess_image(img),
            penultimate_layer=-1
        )[0]

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        cam = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            cv2.resize(img, (224, 224)), 0.6, heatmap, 0.4, 0
        )

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "overlay.jpg")
        cv2.imwrite(output_path, overlay)

        return render_template(
            'result.html',
            prediction=predicted_class,
            confidence=confidence,
            image_path=output_path
        )

    return render_template('tb.html')

if __name__ == '__main__':
    app.run(debug=True)
