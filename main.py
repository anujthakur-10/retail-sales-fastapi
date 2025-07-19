from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import uvicorn

app = FastAPI()

# Load model
model = load_model("model/lstm_model.h5")

# Define input schema
from typing import List

class SalesInput(BaseModel):
    sales_sequence: List[float]

@app.post("/predict")
def predict(input_data: SalesInput):
    # Preprocess input
    sequence = np.array(input_data.sales_sequence).reshape(1, -1, 1)

    # Predict
    prediction = model.predict(sequence)
    predicted_sales = float(prediction[0][0])

    return {"predicted_sales": round(predicted_sales, 2)}
