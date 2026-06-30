"""
Flask web app for plant disease & pest detection.
Loads a TensorFlow/Keras .h5 model and serves predictions via a browser UI.
"""

import os
import io
import json
import time
from collections import deque, Counter
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB
MAX_CONTENT_LENGTH_MB = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)

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


class PredictionMonitor:
    """In-memory drift/quality monitor for served predictions.

    Keeps a rolling window of recent top-1 predictions so /stats can surface
    class distribution and confidence trends without a database — the kind
    of lightweight model-monitoring loop teams add once a model is in
    production and they need to notice drift or quality regressions early.
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.total_predictions = 0
        self.total_low_confidence = 0
        self.started_at = time.time()

    def record(self, top_prediction: dict, demo: bool):
        self.total_predictions += 1
        if top_prediction.get('confidence_level') == 'low':
            self.total_low_confidence += 1
        self.history.append({
            'raw': top_prediction.get('raw'),
            'crop': top_prediction.get('crop'),
            'condition': top_prediction.get('condition'),
            'healthy': top_prediction.get('healthy'),
            'confidence': top_prediction.get('confidence'),
            'confidence_level': top_prediction.get('confidence_level'),
            'demo': demo,
            'timestamp': time.time(),
        })

    def stats(self) -> dict:
        n = len(self.history)
        if n == 0:
            return {
                'window_size': self.window_size,
                'samples_in_window': 0,
                'total_predictions': self.total_predictions,
                'total_low_confidence': self.total_low_confidence,
                'avg_confidence': None,
                'low_confidence_rate': None,
                'healthy_rate': None,
                'class_distribution': {},
                'uptime_seconds': round(time.time() - self.started_at, 1),
            }
        confidences = [h['confidence'] for h in self.history]
        low_count = sum(1 for h in self.history if h['confidence_level'] == 'low')
        healthy_count = sum(1 for h in self.history if h['healthy'])
        class_counts = Counter(h['raw'] for h in self.history)
        return {
            'window_size': self.window_size,
            'samples_in_window': n,
            'total_predictions': self.total_predictions,
            'total_low_confidence': self.total_low_confidence,
            'avg_confidence': round(sum(confidences) / n, 2),
            'low_confidence_rate': round(low_count / n * 100, 2),
            'healthy_rate': round(healthy_count / n * 100, 2),
            'class_distribution': dict(class_counts.most_common(10)),
            'uptime_seconds': round(time.time() - self.started_at, 1),
        }


MONITOR = PredictionMonitor()


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
    if not image_bytes:
        return jsonify({'error': 'Uploaded file is empty.'}), 400
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
        MONITOR.record(predictions[0], demo=True)
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
        MONITOR.record(predictions[0], demo=False)
        return jsonify({'predictions': predictions, 'demo': False})
    except (IOError, OSError, ValueError) as e:
        return jsonify({'error': f'Could not process image: {e}'}), 400
    except Exception as e:
        app.logger.exception('Prediction failed')
        return jsonify({'error': 'Prediction failed due to an internal error.'}), 500


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
            if not image_bytes:
                results.append({'filename': file.filename, 'error': 'Uploaded file is empty.'})
                continue
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
                MONITOR.record(predictions[0], demo=True)
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
                MONITOR.record(predictions[0], demo=False)
                results.append({'filename': file.filename, 'predictions': predictions, 'demo': False})
        except (IOError, OSError, ValueError) as e:
            results.append({'filename': file.filename, 'error': f'Could not process image: {e}'})
        except Exception:
            app.logger.exception('Batch prediction failed for %s', file.filename)
            results.append({'filename': file.filename, 'error': 'Prediction failed due to an internal error.'})

    return jsonify({'count': len(results), 'results': results})


@app.errorhandler(413)
def handle_too_large(e):
    return jsonify({'error': f'Image too large. Max upload size is {MAX_CONTENT_LENGTH_MB} MB.'}), 413


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({'error': 'Not found.'}), 404


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({'error': e.description or e.name}), e.code


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.exception('Unhandled error')
    return jsonify({'error': 'Internal server error. Please try again.'}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'model_path': MODEL_PATH,
        'classes': len(LABELS),
    })


@app.route('/stats')
def stats():
    """Lightweight model-monitoring endpoint.

    Reports rolling class distribution, average confidence, and low-confidence
    rate over the most recent predictions — the kind of drift/quality signal
    an MLOps governance setup checks to catch a model silently degrading in
    production before it becomes a support ticket.
    """
    return jsonify(MONITOR.stats())


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1', host='0.0.0.0', port=port)
