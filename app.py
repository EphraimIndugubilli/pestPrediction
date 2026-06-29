"""
Flask web app for plant disease & pest detection.
Loads a TensorFlow/Keras .h5 model and serves predictions via a browser UI.
"""

import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# --- Model loading (lazy, on first request) ---
MODEL = None
MODEL_PATH = os.environ.get('MODEL_PATH', 'Plant_Disease_Detection/plant_disease_model.h5')
IMG_SIZE = int(os.environ.get('IMG_SIZE', '224'))

# PlantVillage 38-class labels (default — override with LABELS_PATH env var)
DEFAULT_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

LABELS = DEFAULT_LABELS
_labels_path = os.environ.get('LABELS_PATH', '')
if _labels_path and os.path.exists(_labels_path):
    with open(_labels_path) as f:
        LABELS = json.load(f)

ALLOWED = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        print(f"[app] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[app] Could not load model: {e}")
        MODEL = None
    return MODEL


def preprocess(image_bytes: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


TREATMENT_URGENCY = {
    'Late_blight': 'critical',
    'Early_blight': 'high',
    'Black_rot': 'high',
    'Bacterial_spot': 'high',
    'Common_rust_': 'high',
    'Northern_Leaf_Blight': 'medium',
    'Cercospora_leaf_spot': 'medium',
    'Powdery_mildew': 'medium',
    'Leaf_scorch': 'medium',
    'Leaf_Mold': 'medium',
    'Septoria_leaf_spot': 'medium',
    'Spider_mites': 'medium',
    'Target_Spot': 'medium',
    'Esca_(Black_Measles)': 'high',
    'Haunglongbing_(Citrus_greening)': 'critical',
    'Tomato_Yellow_Leaf_Curl_Virus': 'critical',
    'Tomato_mosaic_virus': 'high',
    'Cedar_apple_rust': 'medium',
    'Isariopsis_Leaf_Spot': 'medium',
}

CONFIDENCE_THRESHOLDS = {
    'high': 75.0,
    'medium': 45.0,
}


def confidence_level(score: float) -> str:
    if score >= CONFIDENCE_THRESHOLDS['high']:
        return 'high'
    if score >= CONFIDENCE_THRESHOLDS['medium']:
        return 'medium'
    return 'low'


def format_label(raw: str) -> dict:
    parts = raw.split('___')
    crop = parts[0].replace('_', ' ')
    condition = parts[1].replace('_', ' ') if len(parts) > 1 else raw
    healthy = 'healthy' in condition.lower()

    urgency = 'none' if healthy else 'low'
    if not healthy:
        for keyword, level in TREATMENT_URGENCY.items():
            if keyword.lower().replace('_', ' ') in condition.lower():
                urgency = level
                break

    return {'crop': crop, 'condition': condition, 'healthy': healthy, 'raw': raw, 'urgency': urgency}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if not file.filename or not allowed(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, WEBP, or BMP.'}), 400

    image_bytes = file.read()
    model = load_model()

    if model is None:
        # Demo mode: return mock predictions when model is not available
        import random
        random.seed(len(image_bytes) % 100)
        idxs = random.sample(range(len(LABELS)), 3)
        probs = sorted([random.uniform(0.5, 0.99), random.uniform(0.01, 0.4), random.uniform(0.001, 0.1)], reverse=True)
        predictions = [
            {**format_label(LABELS[i]), 'confidence': round(p * 100, 2),
             'confidence_level': confidence_level(round(p * 100, 2))}
            for i, p in zip(idxs, probs)
        ]
        return jsonify({'predictions': predictions, 'demo': True})

    try:
        arr = preprocess(image_bytes)
        preds = model.predict(arr, verbose=0)[0]
        top3 = np.argsort(preds)[::-1][:3]
        predictions = [
            {**format_label(LABELS[i]), 'confidence': round(float(preds[i]) * 100, 2),
             'confidence_level': confidence_level(round(float(preds[i]) * 100, 2))}
            for i in top3
        ]
        return jsonify({'predictions': predictions, 'demo': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction: process up to 10 images in one request.

    Modern REST pattern — avoids per-image round-trip latency. Send images
    as multipart form-data with field name 'images' (repeat the field for
    each file). Returns an ordered list of predictions matching the input.
    """
    files = request.files.getlist('images')
    if not files or all(not f.filename for f in files):
        return jsonify({'error': 'No images uploaded. Use field name "images" (repeatable).'}), 400

    MAX_BATCH = 10
    if len(files) > MAX_BATCH:
        return jsonify({'error': f'Batch capped at {MAX_BATCH} images per request. Got {len(files)}.'}), 400

    model = load_model()
    results = []

    for file in files:
        if not file.filename or not allowed(file.filename):
            results.append({'filename': file.filename or 'unknown', 'error': 'Invalid file type — use PNG, JPG, JPEG, WEBP, or BMP.'})
            continue
        try:
            image_bytes = file.read()
            if model is None:
                import random
                random.seed(len(image_bytes) % 100)
                idxs = random.sample(range(len(LABELS)), 3)
                probs = sorted([random.uniform(0.5, 0.99), random.uniform(0.01, 0.4), random.uniform(0.001, 0.1)], reverse=True)
                predictions = [
                    {**format_label(LABELS[i]), 'confidence': round(p * 100, 2),
                     'confidence_level': confidence_level(round(p * 100, 2))}
                    for i, p in zip(idxs, probs)
                ]
                results.append({'filename': file.filename, 'predictions': predictions, 'demo': True})
            else:
                arr = preprocess(image_bytes)
                preds = model.predict(arr, verbose=0)[0]
                top3 = np.argsort(preds)[::-1][:3]
                predictions = [
                    {**format_label(LABELS[i]), 'confidence': round(float(preds[i]) * 100, 2),
                     'confidence_level': confidence_level(round(float(preds[i]) * 100, 2))}
                    for i in top3
                ]
                results.append({'filename': file.filename, 'predictions': predictions, 'demo': False})
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})

    return jsonify({'count': len(results), 'results': results})


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'model_path': MODEL_PATH,
        'classes': len(LABELS),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1', host='0.0.0.0', port=port)
