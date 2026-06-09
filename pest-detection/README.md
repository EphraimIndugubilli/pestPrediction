# Pest Detection System — Setup Guide

## Project Structure

```
pest_detection/
├── main.py                    ← FastAPI backend
├── requirements.txt           ← Python dependencies
├── pest_cnn_model.h5          ← YOUR model (copy it here)
├── pest_dataset(main)/        ← YOUR dataset (optional, for class names)
│   └── train/
│       ├── aphids/
│       ├── whitefly/
│       └── ...
├── templates/
│   └── index.html             ← Frontend UI
└── static/                    ← (empty, for future static assets)
```

---

## Step 1 — Copy your model and dataset

Copy your files into this folder:

```
pest_cnn_model.h5           → pest_detection/pest_cnn_model.h5
pest_dataset(main)/         → pest_detection/pest_dataset(main)/
```

If you don't have the dataset folder here, open `main.py` and manually edit
the `CLASS_NAMES` list so it matches your training class folders exactly, in
**sorted (alphabetical) order**.

---

## Step 2 — Install dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

> If you use a virtual environment (recommended):
> ```bash
> python -m venv venv
> venv\Scripts\activate        # Windows
> source venv/bin/activate     # Mac/Linux
> pip install -r requirements.txt
> ```

---

## Step 3 — Edit PESTICIDE_DB (important)

Open `main.py` and find the `PESTICIDE_DB` dictionary.
Replace the placeholder entries with your actual pests. The **keys must exactly
match** your class folder names (same spelling, same case).

Example:
```python
PESTICIDE_DB = {
    "aphids": {
        "pesticide": "Imidacloprid",
        "dose": "0.5 ml/L water, spray on undersides of leaves",
        "severity_note": "Check weekly; aphids spread fast in warm weather",
    },
    # ... add all your classes
}
```

---

## Step 4 — Run the server

```bash
uvicorn main:app --reload
```

You should see:
```
✅ Model loaded from pest_cnn_model.h5
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## Step 5 — Open the app

Open your browser and go to:

```
http://127.0.0.1:8000
```

Drag and drop a leaf image — the CNN model will run inference and show the
pest name, confidence, pesticide recommendation, and dosage.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model not loaded` badge | Make sure `pest_cnn_model.h5` is in the same folder as `main.py` |
| `Could not reach the backend` | Run `uvicorn main:app --reload` first |
| Wrong class names in result | Edit `CLASS_NAMES` in `main.py` to match your training folders exactly |
| Low confidence / always uncertain | Lower `CONF_THRESHOLD` in `main.py` (default 85) |
| `tensorflow` install fails | Try `pip install tensorflow-cpu` instead |

---

## Adjustable settings in main.py

```python
IMG_SIZE = (224, 224)     # Change if your model uses a different input size
CONF_THRESHOLD = 85.0     # Lower this if you get too many "uncertain" results
```
