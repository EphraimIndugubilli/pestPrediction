from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
# tensorflow is imported lazily so the server still starts if it's not installed
try:
    from tensorflow.keras.models import load_model
    MODEL_PATH = "pest_cnn_model.h5"
    model = load_model(MODEL_PATH, compile=False)
    MODEL_LOADED = True
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    MODEL_LOADED = False
    print(f"⚠️  Could not load model: {e}")


# ─── CLASS NAMES ───────────────────────────────────────────────────────────────
# Replace these with your actual folder names from pest_dataset(main)/train
# They must be in the SAME sorted order as when you trained the model.
DATASET_TRAIN_PATH = "pest_dataset(main)/train"

if os.path.exists(DATASET_TRAIN_PATH):
    CLASS_NAMES = sorted(os.listdir(DATASET_TRAIN_PATH))
    print(f"📂 Classes loaded from dataset: {CLASS_NAMES}")
else:
    # Sorted alphabetically — must match the order used during training
    CLASS_NAMES = sorted([
        "Adristyrannus",
        "Aleurocanthus spiniferus",
        "Ampelophaga",
        "Aphis citricola Vander Goot",
        "Apolygus lucorum",
        "alfalfa plant bug",
        "alfalfa seed chalcid",
        "alfalfa weevil",
        "aphids",
    ])
    print(f"📋 Using class names: {CLASS_NAMES}")


# ─── PESTICIDE DATABASE ────────────────────────────────────────────────────────
# Keys must exactly match the class folder names above (case-sensitive).
# Add a severity_note to any entry for extra advisory text in the UI.
PESTICIDE_DB = {
    "Adristyrannus": {
        "pesticide": "Chlorpyrifos",
        "dose": "2 ml/L",
        "severity_note": "Apply in the early morning; avoid spraying near water sources.",
    },
    "Aleurocanthus spiniferus": {
        "pesticide": "Neem Oil",
        "dose": "3 ml/L",
        "severity_note": "Spray undersides of leaves where nymphs cluster.",
    },
    "Ampelophaga": {
        "pesticide": "Spinosad",
        "dose": "0.3 ml/L",
        "severity_note": "Target young larvae; older larvae are harder to control.",
    },
    "Aphis citricola Vander Goot": {
        "pesticide": "Imidacloprid",
        "dose": "0.5 ml/L",
        "severity_note": "Monitor for ant activity — ants protect aphid colonies.",
    },
    "Apolygus lucorum": {
        "pesticide": "Thiamethoxam",
        "dose": "0.25 g/L",
        "severity_note": "Most damaging during flowering; prioritize treatment then.",
    },
    "alfalfa plant bug": {
        "pesticide": "Malathion",
        "dose": "2 ml/L",
        "severity_note": "Scout fields weekly during spring flush.",
    },
    "alfalfa seed chalcid": {
        "pesticide": "Lambda-cyhalothrin",
        "dose": "0.5 ml/L",
        "severity_note": "Time applications to coincide with adult emergence.",
    },
    "alfalfa weevil": {
        "pesticide": "Carbaryl",
        "dose": "1 g/L",
        "severity_note": "Check for larval feeding on terminals before spraying.",
    },
    "aphids": {
        "pesticide": "Imidacloprid",
        "dose": "0.5 ml/L",
        "severity_note": "Check weekly; aphids spread fast in warm weather.",
    },
}


# ─── SETTINGS ──────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)          # Must match what your model was trained on
CONF_THRESHOLD = 40.0          # Minimum % confidence to report a detection
STABLE_THRESHOLD = 1           # For single image, 1 is fine (no temporal filter)


# ─── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Pest Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html", "r") as f:
        return f.read()


@app.get("/status")
async def status():
    return {
        "model_loaded": MODEL_LOADED,
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "conf_threshold": CONF_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Check that pest_cnn_model.h5 exists."},
        )

    # ── Read & validate image ──
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    # ── Green-ratio check (mirrors original leaf detection) ──
    img_resized = img.resize(IMG_SIZE)
    img_np = np.array(img_resized)
    green_ratio = float(np.mean(img_np[:, :, 1]))  # mean of green channel

    if green_ratio < 80:
        return {
            "status": "uncertain",
            "reason": "Image doesn't appear to contain a leaf (low green content).",
            "green_ratio": round(green_ratio, 1),
            "pest_name": None,
            "confidence": 0,
            "pesticide": None,
            "dose": None,
            "severity_note": None,
        }

    # ── Preprocess & predict ──
    img_array = img_np / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    confidence = float(np.max(preds)) * 100
    class_idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[class_idx]

    # ── Confidence gate ──
    if confidence < CONF_THRESHOLD or predicted_class not in PESTICIDE_DB:
        return {
            "status": "uncertain",
            "reason": f"Confidence too low ({confidence:.1f}%) or class not in database.",
            "green_ratio": round(green_ratio, 1),
            "pest_name": predicted_class,
            "confidence": round(confidence, 1),
            "pesticide": None,
            "dose": None,
            "severity_note": None,
        }

    info = PESTICIDE_DB[predicted_class]

    # Build top-3 for display
    top3_idx = np.argsort(preds[0])[::-1][:3]
    top3 = [
        {"class": CLASS_NAMES[i], "confidence": round(float(preds[0][i]) * 100, 1)}
        for i in top3_idx
    ]

    return {
        "status": "detected",
        "pest_name": predicted_class,
        "confidence": round(confidence, 1),
        "green_ratio": round(green_ratio, 1),
        "pesticide": info["pesticide"],
        "dose": info["dose"],
        "severity_note": info.get("severity_note", ""),
        "top3": top3,
    }
