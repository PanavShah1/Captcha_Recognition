import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations
import pickle as pkl
import os
from pathlib import Path
from sklearn import preprocessing

from src.model import CaptchaModel, DeepCaptchaModel
from src.dataset import ClassificationDataset
import src.config as config
from src.train import decode_predictions
from pprint import pprint

MODEL_PATH = Path("models/captcha_model_final.pth")
IMAGE_PATHS = Path("test_images")
ENCODER_PATH = Path("assets/encoder_final.pkl")

def multiple_predict_captcha(IMAGE_PATHS, ENCODER_PATH, MODEL_PATH):
    # Load the model
    images = list(IMAGE_PATHS.glob("*.jpg"))  
    # print(images)

    with open(ENCODER_PATH, "rb") as f:
        data = pkl.load(f)
        lbl_enc = data["lbl_enc"]

    targets_enc = [[2 for _ in range(5)] for _ in range(len(images))]  # Improve

    lbl_enc_ = preprocessing.LabelEncoder()
    lbl_enc_.classes_ = lbl_enc.classes_
    # print(lbl_enc_.classes_)

    dataset = ClassificationDataset(images, targets=targets_enc, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    # print(dataset[0]['images'].shape)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    # print(f"Dataset size: {len(dataset)}")

    model = DeepCaptchaModel(num_chars=len(lbl_enc.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    for data in loader:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        with torch.no_grad():
            batch_preds, _ = model(**data)
        batch_preds = batch_preds.cpu().numpy()
        batch_preds = torch.tensor(batch_preds)
        decoded_preds = decode_predictions(batch_preds, lbl_enc_)
        # for image, decoded_pred in zip(images, decoded_preds):
        #     print(image)
        #     print(decoded_pred)

        return {
            "predictions": decoded_preds,
            "images": [str(image) for image in images],
        }
    

def predict_captcha(IMAGE_PATH, ENCODER_PATH=ENCODER_PATH, MODEL_PATH=MODEL_PATH):
    # Load the model
    images = [IMAGE_PATH] 
    # print(images)

    with open(ENCODER_PATH, "rb") as f:
        data = pkl.load(f)
        lbl_enc = data["lbl_enc"]

    targets_enc = [[2 for _ in range(5)] for _ in range(len(images))]  # Improve

    lbl_enc_ = preprocessing.LabelEncoder()
    lbl_enc_.classes_ = lbl_enc.classes_
    # print(lbl_enc_.classes_)

    dataset = ClassificationDataset(images, targets=targets_enc, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    # print(dataset[0]['images'].shape)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    # print(f"Dataset size: {len(dataset)}")

    model = DeepCaptchaModel(num_chars=len(lbl_enc.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    for data in loader:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        with torch.no_grad():
            batch_preds, _ = model(**data)
        batch_preds = batch_preds.cpu().numpy()
        batch_preds = torch.tensor(batch_preds)
        decoded_preds = decode_predictions(batch_preds, lbl_enc_)
        # for image, decoded_pred in zip(images, decoded_preds):
        #     print(image)
        #     print(decoded_pred)

        return {
            "predictions": decoded_preds,
            "images": [str(image) for image in images],
        }


if __name__ == "__main__":
    predicted = multiple_predict_captcha(IMAGE_PATHS, ENCODER_PATH, MODEL_PATH)
    for image, pred in zip(predicted["images"], predicted["predictions"]):
        print(f"Image: {image}")
        print(f"Prediction: {pred}")
    print(predict_captcha("test_images/03qk.jpg", ENCODER_PATH, MODEL_PATH))