from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np
import pandas as pd

app = FastAPI()


model_uri = "s3://jedhajules/4/eabec8a80187403690855a82e0c788e1/artifacts/CatBoostRegressor_best_model"
model = mlflow.sklearn.load_model(model_uri)


class PredictionRequest(BaseModel):
    input: list

@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    print("Received request:", request.input)
    
    input_data = pd.DataFrame(request.input, columns=[
        "model_key", "mileage", "engine_power", "fuel", "paint_color", 
        "car_type", "private_parking_available", "has_gps", 
        "has_air_conditioning", "automatic_car", "has_getaround_connect", 
        "has_speed_regulator", "winter_tires"
    ])
    print("Input data DataFrame:", input_data)
    
    prediction = model.predict(input_data).tolist()
    print("Prediction result:", prediction)
    return {"prediction": prediction}
