from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import torch
import torch.nn.functional as F
from PIL import Image
import io
import time
import json
import torchvision.transforms as transforms
import torchvision.models as tv_models

from models import BaselineCNN, ImprovedCNN, get_resnet


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LOAD CLASS LABELS (LOCKED ORDER)
# -----------------------------
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)


# -----------------------------
# APP
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return FileResponse("templates/index.html")


# -----------------------------
# TRANSFORMS (MUST MATCH TRAINING EXACTLY)
# -----------------------------
TRANSFORMS = {
    "baseline": transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ]),

    "improved": transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ]),

    "resnet": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


# -----------------------------
# MODEL LOADING
# -----------------------------
def load_model(model_class, path, *args):
    model = model_class(*args).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


models_dict = {
    "baseline": load_model(BaselineCNN, "models/best_baseline_model.pth", NUM_CLASSES),
    "improved": load_model(ImprovedCNN, "models/best_improved_model.pth", NUM_CLASSES),
}


# -----------------------------
# RESNET (MATCH TRAINING EXACTLY)
# -----------------------------
resnet = tv_models.resnet18(weights=None)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet.load_state_dict(torch.load("models/resnet18_best.pth", map_location=device))
resnet.eval()

models_dict["resnet"] = resnet


# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...), model_name: str = Form(...)):

    start = time.time()

    # -------------------------
    # LOAD IMAGE
    # -------------------------
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    model_name = model_name.lower().strip()

    if model_name not in models_dict:
        model_name = "baseline"

    model = models_dict[model_name]
    transform = TRANSFORMS[model_name]

    # -------------------------
    # PREPROCESS
    # -------------------------
    img_tensor = transform(img).unsqueeze(0).to(device)

    # -------------------------
    # INFERENCE
    # -------------------------
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)

        top_prob, top_idx = torch.max(probs, dim=1)

        top_idx = top_idx.item()
        confidence = top_prob.item() * 100

        # ---------------- DEBUG (KEEP DURING TESTING)
        print("\n================ DEBUG ================")
        print("MODEL:", model_name)
        print("OUTPUT SHAPE:", outputs.shape)
        print("TOP INDEX:", top_idx)
        print("TOP CLASS:", class_names[top_idx])
        print("CONFIDENCE:", confidence)
        print("======================================\n")

    # -------------------------
    # FULL PREDICTIONS
    # -------------------------
    predictions = [
        {
            "label": class_names[i],
            "confidence": round(probs[0][i].item() * 100, 2)
        }
        for i in range(NUM_CLASSES)
    ]

    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # -------------------------
    # RESPONSE
    # -------------------------
    return {
        "model": model_name,
        "top_prediction": {
            "label": class_names[top_idx],
            "confidence": round(confidence, 2)
        },
        "predictions": predictions,
        "duration_ms": round((time.time() - start) * 1000, 2)
    }