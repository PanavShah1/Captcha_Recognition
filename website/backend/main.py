from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import subprocess
import os
import tempfile
from pathlib import Path
import ast
import json
from src.predict import predict_captcha
from src.clean_output import clean_output

app = FastAPI()

@app.post("/predict")
async def predict_captcha_route(file: UploadFile = File(...)):
    try:
        # Define your desired save path
        save_path = "website/backend/input_file.jpg"

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the uploaded file to the desired path
        with open(save_path, "wb") as image_file:
            image_file.write(await file.read())

        print(f"Image successfully saved to: {save_path}")

        model_path = os.path.abspath("models/captcha_model_temp_3_final.pth")
        encoder_path = os.path.abspath("assets/encoder_temp_3_final.pkl")

        print(f"Using model file from: {model_path}")
        print(f"Using encoder file from: {encoder_path}")

        prediction = predict_captcha(save_path)
        print(prediction)
        prediction = prediction["predictions"][0]
        clean_prediction = clean_output(prediction)
        print("clean_prediction", clean_prediction)

        return JSONResponse(content={"prediction": prediction, "clean_prediction": clean_prediction}, status_code=200)



    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return JSONResponse(content={"error": f"An exception occurred: {str(e)}"}, status_code=500)
