import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations
import pickle as pkl
import os
from pathlib import Path
from sklearn import preprocessing

from model import CaptchaModel, DeepCaptchaModel, DeepCaptchaModelSmallerTimeSteps
from dataset import ClassificationDataset
import config
from train import decode_predictions
from pprint import pprint

MODEL_PATH = Path("models/captcha_model_temp_3.pth")
IMAGE_PATHS = Path("test_images")
ENCODER_PATH = Path("assets/encoder_temp_3_copy.pkl")

images = list(IMAGE_PATHS.glob("*.jpg"))  
print(images)

with open(ENCODER_PATH, "rb") as f:
    data = pkl.load(f)
    lbl_enc = data["lbl_enc"]

targets_enc = [[2 for _ in range(5)] for _ in range(len(images))]  # Improve

lbl_enc_ = preprocessing.LabelEncoder()
lbl_enc_.classes_ = lbl_enc.classes_
print(lbl_enc_.classes_)

dataset = ClassificationDataset(images, targets=targets_enc, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
print(dataset[0]['images'].shape)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
)
print(f"Dataset size: {len(dataset)}")

model = DeepCaptchaModelSmallerTimeSteps(num_chars=len(lbl_enc.classes_))
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
    for image, decoded_pred in zip(images, decoded_preds):
        print(image)
        print(decoded_pred)
