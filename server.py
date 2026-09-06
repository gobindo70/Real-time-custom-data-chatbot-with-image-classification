import io
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="AI Image Inference Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "catdog_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load weights: {str(e)}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((160, 160))
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

# এটি আপনার HTML ফাইলকে সরাসরি লোকালহোস্টে সার্ভ করবে
@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Uploaded file must be an image."})

    try:
        content = await file.read()
        tensor_input = preprocess_image(content)

        raw_prediction = model.predict(tensor_input)
        prediction_score = float(raw_prediction[0][0])

        if prediction_score < 0.5:
            detected_class = "Cat"
            confidence = (1 - prediction_score) * 100
        else:
            detected_class = "Dog"
            confidence = prediction_score * 100

        return {
            "success": True,
            "class": detected_class,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
