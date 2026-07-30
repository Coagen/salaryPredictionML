from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model\\lr_model.pkl')

app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictionRequest(BaseModel):
    YearsExperience:float

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([[request.YearsExperience]])
    return {"prediction":prediction[0]}

