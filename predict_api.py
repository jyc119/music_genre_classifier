from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from deep_model import DeepLearningGenreClassifier
import shutil
import os

app = FastAPI()
classifier = DeepLearningGenreClassifier(model_path="model.pth")
GENRES = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

class PredictionResult(BaseModel):
    genre: str
    confidence: float


@app.post("/predict", response_model=PredictionResult)
async def predict_genre(file: UploadFile = File(...)):
    """
    Predict genre from uploaded audio file.
    """
    temp_path = f"temp_audio/{file.filename}"
    os.makedirs("temp_audio", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    index, confidence = classifier.predict(temp_path)
    genre = GENRES[index]
    os.remove(temp_path)  # Clean up temp file
    return {"genre": genre, "confidence": confidence}

@app.get("/")
def read_root():
    return {"Hello": "World"}