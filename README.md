# 🌿 PlantAI — Pest & Plant Disease Detection

AI-powered plant disease and pest detection from leaf images. Supports 38 disease classes across 14 crop types (PlantVillage dataset).

**Features:**
- Drag-and-drop web UI with live confidence bars
- Top-3 predictions with treatment advice
- Demo mode when model is not loaded (safe to explore the UI)
- REST API endpoint (`/predict`) for programmatic use
- Health check endpoint (`/health`)

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/EphraimIndugubilli/pestPrediction.git
cd pestPrediction
pip install -r requirements.txt

# 2. (Optional) Place your trained model
#    Default expected path:
#    Plant_Disease_Detection/plant_disease_model.h5

# 3. Run the app
python app.py

# 4. Open http://localhost:5000 in your browser
```

## API Usage

```bash
# Predict from an image file
curl -X POST http://localhost:5000/predict \
  -F "image=@leaf.jpg"

# Response
{
  "predictions": [
    { "crop": "Tomato", "condition": "Late blight", "healthy": false, "confidence": 87.4 },
    { "crop": "Tomato", "condition": "Early blight", "healthy": false, "confidence": 8.2 },
    { "crop": "Tomato", "condition": "healthy",      "healthy": true,  "confidence": 3.1 }
  ],
  "demo": false
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Plant_Disease_Detection/plant_disease_model.h5` | Path to Keras `.h5` model |
| `IMG_SIZE` | `224` | Input image size (pixels) |
| `LABELS_PATH` | *(built-in)* | Path to JSON array of class labels |
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `0` | Set to `1` for debug mode |

## Production Deployment

```bash
# Using gunicorn (Linux/Mac)
gunicorn -w 2 -b 0.0.0.0:8000 app:app

# Using Docker
docker build -t plantai .
docker run -p 5000:5000 -v $(pwd)/Plant_Disease_Detection:/app/Plant_Disease_Detection plantai
```

## Supported Crops & Classes

38 classes across: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

## Project Structure

```
pestPrediction/
├── app.py                              # Flask web app
├── requirements.txt
├── templates/
│   └── index.html                      # Drag-and-drop web UI
├── Plant_Disease_Detection/
│   └── plant_disease_model.h5          # Your trained Keras model (place here)
└── Pesticides_With_Agri_Guideline_Dosage.xlsx
```

## Tech Stack

Python · Flask · TensorFlow/Keras · PIL · NumPy

---

Built by [EphraimIndugubilli](https://github.com/EphraimIndugubilli)
