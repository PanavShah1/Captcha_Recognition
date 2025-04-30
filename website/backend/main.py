from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import subprocess
import os
import tempfile
from pathlib import Path
import ast
import json

app = FastAPI()

@app.post("/predict")
async def predict_captcha_route(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        model_path = os.path.abspath("models/captcha_model_temp_3_final.pth")
        encoder_path = os.path.abspath("assets/encoder_temp_3_final.pkl")

        print(f"Temporary file saved at: {temp_file_path}")
        print(f"Using model file from: {model_path}")
        print(f"Using encoder file from: {encoder_path}")

        working_directory = Path(__file__).parent.parent.parent  

        result = subprocess.run(
            ['python3', 'src/predict.py', temp_file_path, encoder_path, model_path],
            capture_output=True, text=True, env=os.environ, cwd=working_directory
        )

        print(f"Subprocess stdout: {result.stdout}")
        print(f"Subprocess stderr: {result.stderr}")

        if result.returncode != 0:
            return JSONResponse(content={"error": "Error occurred while running prediction", "details": result.stderr}, status_code=500)

        output = result.stdout
        print(f"Prediction Output    : {output}")
        print(type(output))
        output1 = ast.literal_eval(output) 
        print("output1", output1)

        print("output1['predictions']", output1['predictions'][0])
        return JSONResponse(content={
            "predictions": output1['predictions'][0],
        })

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return JSONResponse(content={"error": f"An exception occurred: {str(e)}"}, status_code=500)
