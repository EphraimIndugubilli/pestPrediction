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


def preprocess(image_bytes: bytes) -> tuple:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_w, orig_h = img.size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    info = {
        'original_size': f'{orig_w}×{orig_h}',
        'model_input_size': f'{IMG_SIZE}×{IMG_SIZE}',
        'file_size_kb': round(len(image_bytes) / 1024, 1),
        'normalized': True,
    }
    return np.expand_dims(arr, 0), info


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

# 2026 MLOps risk-aware inference: asymmetric confidence thresholds per disease.
# Critical diseases use LOWER thresholds (higher sensitivity, fewer missed cases)
# because the cost of a false negative far exceeds the cost of a false positive.
# Healthy classifications use a HIGHER threshold (higher specificity) so we avoid
# giving a false "all clear" when the model isn't truly confident.
DISEASE_RISK_THRESHOLDS: dict[str, dict[str, float]] = {
    # Critical — irreversible spread or no cure; bias toward early detection
    'Late_blight':                    {'high': 50.0, 'medium': 28.0},
    'Haunglongbing_(Citrus_greening)': {'high': 50.0, 'medium': 28.0},
    'Tomato_Yellow_Leaf_Curl_Virus':   {'high': 50.0, 'medium': 28.0},
    # High urgency — rapid spread; earlier treatment window matters
    'Early_blight':                    {'high': 62.0, 'medium': 36.0},
    'Black_rot':                       {'high': 62.0, 'medium': 36.0},
    'Bacterial_spot':                  {'high': 62.0, 'medium': 36.0},
    'Common_rust_':                    {'high': 62.0, 'medium': 36.0},
    'Esca_(Black_Measles)':            {'high': 62.0, 'medium': 36.0},
    'Tomato_mosaic_virus':             {'high': 62.0, 'medium': 36.0},
}

HEALTHY_HIGH_THRESHOLD = 85.0  # require strong confidence before signalling healthy


def confidence_level(score: float, raw_label: str = '') -> str:
    """Return confidence tier, calibrated to disease risk.

    Critical/high-urgency diseases use lower thresholds so the system errs on
    the side of early detection (high sensitivity). Healthy classifications use
    a raised threshold so we don't emit a false "all clear" (high specificity).
    """
    if 'healthy' in raw_label.lower():
        if score >= HEALTHY_HIGH_THRESHOLD:
            return 'high'
        if score >= CONFIDENCE_THRESHOLDS['medium']:
            return 'medium'
        return 'low'
    thresholds = DISEASE_RISK_THRESHOLDS.get(raw_label, CONFIDENCE_THRESHOLDS)
    if score >= thresholds['high']:
        return 'high'
    if score >= thresholds['medium']:
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


def assess_image_quality(image_bytes: bytes) -> dict:
    """Score an uploaded image for prediction-readiness before inference.

    2026 MLOps best practice: validate input quality at the boundary before
    spending compute on a prediction that is likely to produce low confidence.
    Returns a quality_score (0-100) and a list of quality_flags describing
    any detected issues (blur, darkness, low resolution, clipping).
    """
    try:
        from PIL import Image, ImageStat
        import math
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        w, h = img.size
        flags: list[str] = []
        score = 100

        # Resolution check
        if w < 64 or h < 64:
            flags.append('very_low_resolution')
            score -= 40
        elif w < 128 or h < 128:
            flags.append('low_resolution')
            score -= 20

        # Brightness check — mean pixel value across channels
        stat = ImageStat.Stat(img)
        mean_brightness = sum(stat.mean) / 3.0
        if mean_brightness < 30:
            flags.append('too_dark')
            score -= 25
        elif mean_brightness > 225:
            flags.append('overexposed')
            score -= 20

        # Blur check — variance of per-channel standard deviations.
        # Low std-dev across all channels means a flat/uniform or blurry image.
        avg_stddev = sum(stat.stddev) / 3.0
        if avg_stddev < 10:
            flags.append('likely_blurry_or_uniform')
            score -= 30
        elif avg_stddev < 20:
            flags.append('low_contrast')
            score -= 10

        score = max(0, min(100, score))
        quality_level = 'good' if score >= 70 else 'fair' if score >= 40 else 'poor'

        return {
            'quality_score': score,
            'quality_level': quality_level,
            'quality_flags': flags,
            'image_width': w,
            'image_height': h,
            'mean_brightness': round(mean_brightness, 1),
            'contrast_stddev': round(avg_stddev, 1),
        }
    except Exception:
        return {
            'quality_score': None,
            'quality_level': 'unknown',
            'quality_flags': ['assessment_failed'],
            'image_width': None,
            'image_height': None,
            'mean_brightness': None,
            'contrast_stddev': None,
        }


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


def _seasonal_context_for_prediction(top_prediction: dict) -> dict | None:
    """Auto-inject seasonal disease risk context for the detected crop.

    2026 agri-AI trend: enrich inference with proactive seasonal alerts so
    farmers know what to watch for given the time of year, not just what
    was detected. Uses the current month and detected crop to surface the
    most relevant risk from SEASONAL_RISK calendar.
    """
    import datetime
    crop = (top_prediction.get('crop') or '').strip().lower()
    if not crop or top_prediction.get('healthy'):
        return None
    month = datetime.datetime.now().month
    matched_key = next((k for k in SEASONAL_RISK if crop in k or k in crop), None)
    if not matched_key:
        return None
    risks = SEASONAL_RISK[matched_key]
    active = [r for r in risks if month in r['peak_months']]
    if not active:
        return None
    active.sort(key=lambda r: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}.get(r['risk'], 4))
    return {
        'crop': matched_key,
        'month': month,
        'active_diseases': active,
        'highest_risk': active[0]['risk'],
        'note': f'Active disease risks for {matched_key} in month {month} based on seasonal calendar.',
    }


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

    quality = assess_image_quality(image_bytes)
    model = load_model()

    if model is None:
        # Demo mode: return mock predictions when model is not available
        import random
        random.seed(len(image_bytes) % 100)
        idxs = random.sample(range(len(LABELS)), 3)
        probs = sorted([random.uniform(0.5, 0.99), random.uniform(0.01, 0.4), random.uniform(0.001, 0.1)], reverse=True)
        predictions = []
        for i, p in zip(idxs, probs):
            lbl = format_label(LABELS[i])
            conf = round(p * 100, 2)
            predictions.append({**lbl, 'confidence': conf, 'confidence_level': confidence_level(conf, lbl['raw'])})
        predictions[0]['treatment_advice'] = _lookup_treatment(LABELS[idxs[0]])
        MONITOR.record(predictions[0], demo=True)
        seasonal = _seasonal_context_for_prediction(predictions[0])
        return jsonify({'predictions': predictions, 'demo': True, 'image_quality': quality, 'seasonal_context': seasonal})

    try:
        arr, preprocess_info = preprocess(image_bytes)
        preds = model.predict(arr, verbose=0)[0]
        top3 = np.argsort(preds)[::-1][:3]
        predictions = []
        for i in top3:
            lbl = format_label(LABELS[i])
            conf = round(float(preds[i]) * 100, 2)
            predictions.append({**lbl, 'confidence': conf, 'confidence_level': confidence_level(conf, lbl['raw'])})
        predictions[0]['treatment_advice'] = _lookup_treatment(LABELS[top3[0]])
        MONITOR.record(predictions[0], demo=False)
        seasonal = _seasonal_context_for_prediction(predictions[0])
        return jsonify({'predictions': predictions, 'demo': False, 'image_quality': quality, 'seasonal_context': seasonal, 'preprocess_info': preprocess_info})
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
                predictions = []
                for i, p in zip(idxs, probs):
                    lbl = format_label(LABELS[i])
                    conf = round(p * 100, 2)
                    predictions.append({**lbl, 'confidence': conf, 'confidence_level': confidence_level(conf, lbl['raw'])})
                MONITOR.record(predictions[0], demo=True)
                results.append({'filename': file.filename, 'predictions': predictions, 'demo': True})
            else:
                arr, _ = preprocess(image_bytes)
                preds = model.predict(arr, verbose=0)[0]
                top3 = np.argsort(preds)[::-1][:3]
                predictions = []
                for i in top3:
                    lbl = format_label(LABELS[i])
                    conf = round(float(preds[i]) * 100, 2)
                    predictions.append({**lbl, 'confidence': conf, 'confidence_level': confidence_level(conf, lbl['raw'])})
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


def _lookup_treatment(raw_label: str) -> dict | None:
    """Return a concise treatment summary from DISEASE_KB for a PlantVillage label.

    Matches on the condition part of the label (after '___') using the same
    normalisation as the /disease-info endpoint so results are consistent.
    Returns None for healthy plants or unknown conditions.
    """
    if '___' in raw_label:
        condition_part = raw_label.split('___')[1]
    else:
        condition_part = raw_label
    if 'healthy' in condition_part.lower():
        return None
    key = _normalize_disease_key(condition_part)
    if key in DISEASE_KB:
        info = DISEASE_KB[key]
        return {
            'pathogen': info['pathogen'],
            'first_steps': info['treatment'][:2],
            'prevention': info['prevention'],
            'organic_option': info['organic_option'],
        }
    # Partial match fallback
    for db_key, info in DISEASE_KB.items():
        if any(word in db_key for word in key.split() if len(word) > 4):
            return {
                'pathogen': info['pathogen'],
                'first_steps': info['treatment'][:2],
                'prevention': info['prevention'],
                'organic_option': info['organic_option'],
            }
    return None
@app.route('/disease-info/<path:name>')
def disease_info(name: str):
    """Return structured treatment and pathology data for a named disease.

    Supports both raw PlantVillage label format (e.g. Tomato___Late_blight)
    and human-readable names (e.g. late blight). Returns a 404 with the
    closest partial matches if the disease is not in the knowledge base —
    following the 2026 MLOps pattern of agentic context enrichment where
    the inference API also serves the knowledge needed to act on its output.
    """
    # Extract the condition part from a raw PlantVillage label (e.g. "Tomato___Late_blight"
    # → "Late_blight") before normalising. The check must run on the original string
    # because _normalize_disease_key replaces underscores with spaces, which destroys "___".
    raw_name = name.strip()
    if '___' in raw_name:
        parts = raw_name.split('___')
        raw_name = parts[1] if len(parts) > 1 else raw_name
    key = _normalize_disease_key(raw_name)

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


@app.route('/confidence-history')
def confidence_history():
    """Time-series confidence trend endpoint for MLOps monitoring dashboards.

    2026 MLOps best practice: expose a rolling confidence time series so
    teams can plot prediction quality over time in Grafana, Retool, or any
    dashboard tool — without a database. Returns the most recent `limit`
    records (default 50, max 500) in ascending timestamp order so the caller
    can feed them directly into a line chart.

    Query params:
      limit (int, 1-500): number of records to return (default 50)
      include_demo (bool): whether to include demo-mode predictions (default false)

    Response schema:
      {
        "count": 12,
        "history": [
          {
            "timestamp": 1751234567.89,
            "confidence": 87.4,
            "confidence_level": "high",
            "label": "Tomato___Late_blight",
            "crop": "Tomato",
            "healthy": false,
            "demo": false
          },
          ...
        ],
        "trend": "improving" | "degrading" | "stable" | "insufficient_data",
        "slope_per_10_samples": 1.23
      }
    """
    try:
        limit = min(500, max(1, int(request.args.get('limit', 50))))
    except (TypeError, ValueError):
        limit = 50
    include_demo = request.args.get('include_demo', 'false').lower() in ('1', 'true', 'yes')

    records = list(MONITOR.history)
    if not include_demo:
        records = [r for r in records if not r.get('demo')]
    records = records[-limit:]

    history = [
        {
            'timestamp': r['timestamp'],
            'confidence': r['confidence'],
            'confidence_level': r['confidence_level'],
            'label': r['raw'],
            'crop': r.get('crop'),
            'healthy': r.get('healthy'),
            'demo': r.get('demo', False),
        }
        for r in records
    ]

    # Compute linear trend slope over the window to surface "improving" vs "degrading".
    # A simple least-squares slope over confidence values is enough for a dashboard
    # alert — no scipy needed.
    trend = 'insufficient_data'
    slope = None
    if len(history) >= 10:
        n = len(history)
        xs = list(range(n))
        ys = [h['confidence'] for h in history]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        numerator = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))
        raw_slope = numerator / denominator if denominator else 0.0
        slope = round(raw_slope * 10, 4)  # normalise to per-10-samples
        if slope > 1.0:
            trend = 'improving'
        elif slope < -1.0:
            trend = 'degrading'
        else:
            trend = 'stable'

    return jsonify({
        'count': len(history),
        'history': history,
        'trend': trend,
        'slope_per_10_samples': slope,
    })



class FeedbackStore:
    """Human-in-the-loop correction tracker for 2026 MLOps best practice.

    Records whether users agree with top-1 predictions and what the correct
    label was when they don't. Tracks per-class accuracy rates in a rolling
    window, giving a lightweight signal for catching model drift before it
    becomes a support problem.
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.entries = deque(maxlen=window_size)
        self.total_submitted = 0
        self.total_corrections = 0

    def record(self, predicted: str, correct: bool, correction: str | None, source: str):
        self.total_submitted += 1
        if not correct:
            self.total_corrections += 1
        self.entries.append({
            'predicted': predicted,
            'correct': correct,
            'correction': correction,
            'source': source,
            'timestamp': time.time(),
        })

    def stats(self) -> dict:
        n = len(self.entries)
        if n == 0:
            return {
                'window_size': self.window_size,
                'samples_in_window': 0,
                'total_submitted': self.total_submitted,
                'total_corrections': self.total_corrections,
                'accuracy_rate': None,
                'most_corrected_classes': [],
            }
        correct_count = sum(1 for e in self.entries if e['correct'])
        corrections = [e['correction'] for e in self.entries if not e['correct'] and e['correction']]
        correction_counts = Counter(corrections).most_common(5)
        return {
            'window_size': self.window_size,
            'samples_in_window': n,
            'total_submitted': self.total_submitted,
            'total_corrections': self.total_corrections,
            'accuracy_rate': round(correct_count / n * 100, 2),
            'most_corrected_classes': [{'label': k, 'count': v} for k, v in correction_counts],
        }


FEEDBACK = FeedbackStore()


@app.route('/feedback', methods=['POST'])
def feedback():
    """Human-in-the-loop correction endpoint — 2026 MLOps best practice.

    Accepts a JSON body:
      { "predicted": "Tomato___Late_blight",
        "correct": false,
        "correction": "Tomato___Early_blight",   // null if just marking wrong
        "source": "user" }

    Records the feedback in a rolling window. Aggregate accuracy and the
    most-corrected classes are available at /feedback/stats. This enables
    lightweight model drift monitoring without a database — a pattern widely
    adopted in 2026 production ML deployments to catch silent regressions
    before they reach support.
    """
    body = request.get_json(silent=True) or {}
    predicted = str(body.get('predicted', '')).strip()
    correct = bool(body.get('correct', True))
    correction = body.get('correction')
    source = str(body.get('source', 'user')).strip() or 'user'

    if not predicted:
        return jsonify({'error': '"predicted" field is required.'}), 400
    if correction is not None:
        correction = str(correction).strip() or None

    FEEDBACK.record(predicted=predicted, correct=correct, correction=correction, source=source)
    return jsonify({
        'recorded': True,
        'total_submitted': FEEDBACK.total_submitted,
        'total_corrections': FEEDBACK.total_corrections,
    })


@app.route('/feedback/stats')
def feedback_stats():
    """Aggregate human-feedback accuracy stats for drift monitoring."""
    return jsonify(FEEDBACK.stats())


# Seasonal disease risk calendar — 2026 MLOps trend: enriching inference APIs
# with agronomic context so practitioners know WHEN to watch for each disease,
# not just WHAT to look for. Each entry maps month ranges to disease risk levels
# driven by the environmental conditions (temperature, humidity, rainfall) that
# favour spore germination and spread.
SEASONAL_RISK: dict[str, list[dict]] = {
    "tomato": [
        {"disease": "Late blight",       "peak_months": [6, 7, 8, 9],   "risk": "critical", "trigger": "Cool (10–25°C), wet"},
        {"disease": "Early blight",      "peak_months": [7, 8, 9, 10],  "risk": "high",     "trigger": "Warm, humid, older leaves"},
        {"disease": "Septoria leaf spot","peak_months": [6, 7, 8],      "risk": "medium",   "trigger": "Wet, splashing rain"},
        {"disease": "Bacterial spot",    "peak_months": [5, 6, 7, 8],   "risk": "high",     "trigger": "Warm (>24°C), wet"},
        {"disease": "TYLCV (virus)",     "peak_months": [4, 5, 6, 7, 8],"risk": "critical", "trigger": "Whitefly peak activity"},
        {"disease": "Leaf Mold",         "peak_months": [3, 4, 5, 10, 11],"risk": "medium", "trigger": "High humidity >85%, greenhouse"},
    ],
    "potato": [
        {"disease": "Late blight",       "peak_months": [6, 7, 8, 9],   "risk": "critical", "trigger": "Cool, wet, foggy"},
        {"disease": "Early blight",      "peak_months": [7, 8, 9],      "risk": "high",     "trigger": "Warm, dry spells after rain"},
    ],
    "apple": [
        {"disease": "Apple scab",        "peak_months": [3, 4, 5, 6],   "risk": "high",     "trigger": "Wet spring, ascospore release"},
        {"disease": "Cedar apple rust",  "peak_months": [4, 5, 6],      "risk": "medium",   "trigger": "Wet spring, nearby junipers"},
        {"disease": "Black rot",         "peak_months": [5, 6, 7, 8],   "risk": "high",     "trigger": "Warm, wet, from mummified fruit"},
    ],
    "corn": [
        {"disease": "Northern leaf blight","peak_months": [6, 7, 8],    "risk": "medium",   "trigger": "Moderate temp, high humidity"},
        {"disease": "Common rust",        "peak_months": [6, 7, 8, 9],  "risk": "high",     "trigger": "Cool (16–23°C), humid"},
        {"disease": "Grey leaf spot",     "peak_months": [7, 8, 9],     "risk": "medium",   "trigger": "Warm, humid, minimal tillage"},
    ],
    "grape": [
        {"disease": "Black rot",          "peak_months": [5, 6, 7],     "risk": "high",     "trigger": "Warm, wet spring"},
        {"disease": "Powdery mildew",     "peak_months": [5, 6, 7, 8],  "risk": "medium",   "trigger": "Warm days, cool nights"},
        {"disease": "Downy mildew",       "peak_months": [4, 5, 6, 7],  "risk": "high",     "trigger": "Cool, wet (>10mm rain)"},
    ],
    "strawberry": [
        {"disease": "Leaf scorch",        "peak_months": [5, 6, 7, 8],  "risk": "medium",   "trigger": "Warm, wet"},
        {"disease": "Botrytis (grey mold)","peak_months": [3, 4, 5, 10, 11],"risk": "high", "trigger": "Cool, wet, high humidity"},
    ],
    "pepper": [
        {"disease": "Bacterial spot",     "peak_months": [5, 6, 7, 8],  "risk": "high",     "trigger": "Warm (>24°C), wet"},
        {"disease": "Phytophthora blight","peak_months": [6, 7, 8, 9],  "risk": "critical", "trigger": "Saturated soil, warm"},
    ],
    "peach": [
        {"disease": "Bacterial spot",     "peak_months": [4, 5, 6, 7],  "risk": "high",     "trigger": "Wet, wind-driven rain"},
        {"disease": "Brown rot",          "peak_months": [7, 8, 9],     "risk": "high",     "trigger": "Warm, humid near harvest"},
    ],
    "orange": [
        {"disease": "Citrus greening (HLB)","peak_months": list(range(1,13)),"risk": "critical","trigger": "Year-round — psyllid vector"},
        {"disease": "Citrus canker",      "peak_months": [5, 6, 7, 8, 9],"risk": "high",    "trigger": "Warm, wet, wind"},
    ],
}

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@app.route('/seasonal-risk', methods=['GET'])
def seasonal_risk():
    """Return disease risk calendar for a given crop and optional month.

    2026 agronomic AI trend: proactive risk alerts based on time of year so
    farmers can apply preventive treatments before symptoms appear, not after.
    The endpoint enriches the prediction API with seasonal context — the same
    pattern emerging across agri-AI platforms in 2026.

    Query params:
      crop (str): crop name — tomato, potato, apple, corn, grape, strawberry,
                  pepper, peach, orange (case-insensitive, partial match OK)
      month (int, 1-12): target month; if omitted, returns the full year calendar

    Response:
      {
        "crop": "tomato",
        "month": 7,
        "month_name": "Jul",
        "risks": [
          { "disease": "Late blight", "risk": "critical", "trigger": "Cool (10-25°C), wet",
            "active_this_month": true, "peak_months": [6,7,8,9] },
          ...
        ],
        "active_count": 3,
        "highest_risk": "critical"
      }
    """
    raw_crop = request.args.get('crop', '').strip().lower()
    if not raw_crop:
        available = sorted(SEASONAL_RISK.keys())
        return jsonify({
            'error': 'crop parameter required',
            'available_crops': available,
        }), 400

    # Partial-match lookup
    matched_key = None
    for key in SEASONAL_RISK:
        if raw_crop in key or key in raw_crop:
            matched_key = key
            break

    if not matched_key:
        return jsonify({
            'error': f'No seasonal data for "{raw_crop}".',
            'available_crops': sorted(SEASONAL_RISK.keys()),
        }), 404

    try:
        month = int(request.args.get('month', 0))
        if not (0 <= month <= 12):
            month = 0
    except (TypeError, ValueError):
        month = 0

    calendar = SEASONAL_RISK[matched_key]
    risks = []
    for entry in calendar:
        active = month in entry['peak_months'] if month else True
        risks.append({
            'disease':          entry['disease'],
            'risk':             entry['risk'],
            'trigger':          entry['trigger'],
            'peak_months':      entry['peak_months'],
            'peak_month_names': [MONTH_NAMES[m] for m in entry['peak_months']],
            'active_this_month': active,
        })

    if month:
        risks.sort(key=lambda r: (not r['active_this_month'],
                                  {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}.get(r['risk'], 4)))
    active_risks = [r for r in risks if r['active_this_month']]
    risk_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    highest = min(active_risks, key=lambda r: risk_order.get(r['risk'], 4))['risk'] if active_risks else 'none'

    return jsonify({
        'crop':          matched_key,
        'month':         month or None,
        'month_name':    MONTH_NAMES[month] if month else None,
        'risks':         risks,
        'active_count':  len(active_risks),
        'highest_risk':  highest,
        'note':          'Risk levels are agronomic guidelines based on environmental conditions typical for each month. Actual risk depends on local weather, variety resistance, and field history.',
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1', host='0.0.0.0', port=port)
