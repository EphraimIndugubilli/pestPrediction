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


DISEASE_KB: dict = {
    "apple scab": {
        "pathogen": "Venturia inaequalis (fungus)",
        "symptoms": "Olive-green to brown scabby lesions on leaves and fruit; premature leaf drop.",
        "spread": "Rain splash and wind disperse ascospores and conidia during wet spring weather.",
        "treatment": [
            "Apply captan, myclobutanil, or mancozeb fungicide at bud-break and every 7–14 days during wet periods.",
            "Remove and destroy fallen infected leaves to reduce spore load.",
            "Plant scab-resistant varieties (e.g. Liberty, Enterprise) where possible.",
        ],
        "prevention": "Ensure good air circulation; avoid overhead irrigation.",
        "urgency": "high",
        "organic_option": "Sulfur or copper-based sprays applied preventively.",
    },
    "black rot": {
        "pathogen": "Botryosphaeria obtusa (fungus)",
        "symptoms": "Circular brown lesions with purple borders on leaves; mummified fruit; cankers on branches.",
        "spread": "Spores released from mummified fruit and dead wood during warm wet weather.",
        "treatment": [
            "Remove mummified fruit and cankers; prune infected wood 15 cm below visible lesions.",
            "Apply captan or thiophanate-methyl during pink bud to petal-fall.",
        ],
        "prevention": "Sanitation is key — eliminate all overwintering inoculum.",
        "urgency": "high",
        "organic_option": "Copper hydroxide sprays; aggressive pruning and sanitation.",
    },
    "cedar apple rust": {
        "pathogen": "Gymnosporangium juniperi-virginianae (fungus)",
        "symptoms": "Bright orange spots on upper leaf surface; tube-like spore structures beneath.",
        "spread": "Two-host cycle requiring both apple/crabapple and eastern red cedar (juniper).",
        "treatment": [
            "Myclobutanil or propiconazole fungicides from pink stage through third cover spray.",
            "Remove nearby juniper hosts if feasible.",
        ],
        "prevention": "Plant rust-resistant apple varieties; create distance from junipers.",
        "urgency": "medium",
        "organic_option": "Sulfur sprays (preventive only, not after infection).",
    },
    "late blight": {
        "pathogen": "Phytophthora infestans (oomycete)",
        "symptoms": "Water-soaked grey-green lesions on leaves rapidly turning brown; white mold under leaves; tuber/fruit rot.",
        "spread": "Airborne sporangia; extremely rapid spread in cool (10–25°C) wet conditions.",
        "treatment": [
            "Apply mancozeb, chlorothalonil, or metalaxyl-M preventively before symptoms appear.",
            "Remove and destroy infected plant material immediately — do not compost.",
            "Avoid overhead irrigation; improve air flow.",
        ],
        "prevention": "Use certified disease-free seed; resistant varieties; monitor forecasts.",
        "urgency": "critical",
        "organic_option": "Copper-based fungicides (bordeaux mixture) applied preventively.",
    },
    "early blight": {
        "pathogen": "Alternaria solani (fungus)",
        "symptoms": "Dark brown concentric-ring lesions ('target spots') on older leaves; defoliation from bottom up.",
        "spread": "Wind and rain splash from soil and infected debris.",
        "treatment": [
            "Chlorothalonil, mancozeb, or azoxystrobin applied every 7–10 days after first symptoms.",
            "Remove lower infected leaves; mulch to reduce soil splash.",
        ],
        "prevention": "Crop rotation (3-year); avoid wetting foliage; adequate plant spacing.",
        "urgency": "high",
        "organic_option": "Copper octanoate or neem oil; remove infected tissue promptly.",
    },
    "powdery mildew": {
        "pathogen": "Podosphaera xanthii / Erysiphe spp. (fungi)",
        "symptoms": "White powdery coating on leaves, stems and buds; distorted growth; premature drop.",
        "spread": "Wind-dispersed conidia; thrives in warm dry days with cool nights and high humidity.",
        "treatment": [
            "Sulfur, potassium bicarbonate, or myclobutanil at 7–14 day intervals.",
            "Neem oil as a contact killer of existing colonies.",
        ],
        "prevention": "Good air circulation; avoid excess nitrogen fertilization; resistant varieties.",
        "urgency": "medium",
        "organic_option": "Baking soda spray (1 tbsp/L water); potassium bicarbonate; neem oil.",
    },
    "bacterial spot": {
        "pathogen": "Xanthomonas spp. (bacterium)",
        "symptoms": "Small water-soaked lesions becoming angular, dark, and scab-like on leaves and fruit.",
        "spread": "Rain splash; infected transplants and seeds; thrives above 24°C in wet conditions.",
        "treatment": [
            "Copper bactericide sprays (copper hydroxide or copper sulfate) every 7 days during wet periods.",
            "Remove heavily infected plants; avoid working in wet crops.",
        ],
        "prevention": "Use disease-free seed; resistant pepper/tomato varieties; drip irrigation.",
        "urgency": "high",
        "organic_option": "Copper-based sprays are the primary organic option.",
    },
    "common rust": {
        "pathogen": "Puccinia sorghi (fungus)",
        "symptoms": "Oval to elongated, brick-red pustules on both leaf surfaces; pustules turn dark-brown as season progresses.",
        "spread": "Wind-dispersed urediniospores; rapid spread in cool (16–23°C) humid weather.",
        "treatment": [
            "Azoxystrobin, propiconazole, or trifloxystrobin foliar application at first sign.",
            "Early-season infections require prompt treatment to prevent yield loss.",
        ],
        "prevention": "Plant resistant hybrids; monitor from tassel emergence.",
        "urgency": "high",
        "organic_option": "No highly effective organic option; resistant varieties are the best defence.",
    },
    "northern leaf blight": {
        "pathogen": "Exserohilum turcicum (fungus)",
        "symptoms": "Cigar-shaped, greyish-green to tan lesions (2.5–15 cm long) starting on lower leaves.",
        "spread": "Wind and rain splash; favoured by moderate temperatures and high humidity.",
        "treatment": [
            "Strobilurin or triazole fungicides at VT/early silk if disease is present on 3rd leaf below ear.",
            "Economic threshold: treat when >50% of plants show infection before silking.",
        ],
        "prevention": "Resistant hybrids (single-copy Ht genes); crop rotation; tillage to bury residue.",
        "urgency": "medium",
        "organic_option": "Limited — copper fungicides have low efficacy; rely on resistant varieties.",
    },
    "leaf mold": {
        "pathogen": "Passalora fulva / Cladosporium fulvum (fungus)",
        "symptoms": "Pale-green to yellow spots on upper leaf; olive to grey-green mold growth beneath.",
        "spread": "Airborne conidia; greenhouse tomatoes most affected; thrives >85% humidity.",
        "treatment": [
            "Chlorothalonil or mancozeb spray every 5–7 days.",
            "Reduce greenhouse humidity below 85%; increase ventilation.",
        ],
        "prevention": "Resistant varieties (Cf genes); remove and destroy infected leaves.",
        "urgency": "medium",
        "organic_option": "Copper-based sprays; aggressive humidity management.",
    },
    "tomato yellow leaf curl virus": {
        "pathogen": "Tomato yellow leaf curl virus — TYLCV (begomovirus, whitefly-vectored)",
        "symptoms": "Upward leaf curling; yellowing of leaf margins; stunted growth; flower drop; no effective cure post-infection.",
        "spread": "Transmitted exclusively by silverleaf whitefly (Bemisia tabaci); not mechanically transmitted.",
        "treatment": [
            "No cure — remove and destroy infected plants immediately to limit spread.",
            "Control whitefly vector with imidacloprid, pymetrozine, or insecticidal soap.",
            "Yellow sticky traps to monitor whitefly populations.",
        ],
        "prevention": "Resistant/tolerant varieties; reflective mulch to deter whiteflies; insect-proof netting in seedling stage.",
        "urgency": "critical",
        "organic_option": "Neem oil, insecticidal soap, or pyrethrin against whitefly; reflective mulch.",
    },
    "septoria leaf spot": {
        "pathogen": "Septoria lycopersici (fungus)",
        "symptoms": "Small circular spots with dark-brown border and white-grey center; tiny black pycnidia visible in lesion center.",
        "spread": "Rain splash from soil or infected debris; moves up plant rapidly in wet weather.",
        "treatment": [
            "Chlorothalonil or mancozeb sprays every 7–10 days after first symptoms.",
            "Remove infected lower leaves to slow upward progression.",
        ],
        "prevention": "Mulch to prevent soil splash; avoid overhead watering; crop rotation 3+ years.",
        "urgency": "medium",
        "organic_option": "Copper octanoate; remove infected tissue; mulching.",
    },
    "haunglongbing": {
        "pathogen": "Candidatus Liberibacter asiaticus (bacterium, psyllid-vectored)",
        "symptoms": "Blotchy mottled yellowing ('yellow dragon'); lopsided, bitter, undersized fruit; eventually tree decline and death.",
        "spread": "Asian citrus psyllid (Diaphorina citri); no cure exists for infected trees.",
        "treatment": [
            "No cure — infected trees should be removed and destroyed to prevent spread.",
            "Control psyllid vector with systemic insecticides (imidacloprid, thiamethoxam).",
            "Nutritional programmes can prolong productive life of mildly affected trees.",
        ],
        "prevention": "Certified disease-free nursery stock; psyllid monitoring and control; quarantine.",
        "urgency": "critical",
        "organic_option": "Kaolin clay to reduce psyllid feeding; no effective organic cure.",
    },
}

def _normalize_disease_key(name: str) -> str:
    return name.lower().replace('_', ' ').replace('-', ' ').strip()


@app.route('/disease-info/<path:name>')
def disease_info(name: str):
    """Return structured treatment and pathology data for a named disease.

    Supports both raw PlantVillage label format (e.g. Tomato___Late_blight)
    and human-readable names (e.g. late blight). Returns a 404 with the
    closest partial matches if the disease is not in the knowledge base —
    following the 2026 MLOps pattern of agentic context enrichment where
    the inference API also serves the knowledge needed to act on its output.
    """
    key = _normalize_disease_key(name)
    if '___' in key:
        parts = key.split('___')
        key = parts[1] if len(parts) > 1 else key

    if key in DISEASE_KB:
        info = DISEASE_KB[key]
        return jsonify({
            'disease': key.title(),
            'pathogen': info['pathogen'],
            'symptoms': info['symptoms'],
            'spread': info['spread'],
            'treatment': info['treatment'],
            'prevention': info['prevention'],
            'urgency': info['urgency'],
            'organic_option': info['organic_option'],
        })

    matches = [d for d in DISEASE_KB if key in d or any(w in d for w in key.split())]
    return jsonify({
        'error': f'No data for "{name}".',
        'did_you_mean': matches[:3],
        'available': sorted(DISEASE_KB.keys()),
    }), 404


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
