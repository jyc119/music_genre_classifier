from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from deep_model import DeepLearningGenreClassifier
import shutil
import os

app = FastAPI()
classifier = DeepLearningGenreClassifier(model_path="model.pth")


class PredictionResult(BaseModel):
    genre_id: int


@app.post("/predict", response_model=PredictionResult)
async def predict_genre(file: UploadFile = File(...)):
    """
    Predict genre from uploaded audio file.
    """
    temp_path = f"temp_audio/{file.filename}"
    os.makedirs("temp_audio", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    genre_id = classifier.predict(temp_path)
    os.remove(temp_path)  # Clean up temp file
    return {"genre_id": genre_id}
