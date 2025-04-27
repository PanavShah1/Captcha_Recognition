import os
import glob
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
from model import CaptchaModel, DeepCaptchaModel
import engine
from pprint import pprint
import pickle as pkl
import datetime 

def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("°")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_preds.append(tp)
    return cap_preds


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
    print(len(image_files))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    if not os.path.exists(f"assets/{config.ENCODER}"):
        create_encoder = True
    else:
        with open(f"assets/{config.ENCODER}", "rb") as f:
            data = pkl.load(f)
            lbl_enc = data["lbl_enc"]
            targets_enc = data["targets_enc"]
        
        if len(targets_enc) == 0:
            create_encoder = True
        else:
            create_encoder = False

    if create_encoder:
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(targets_flat)
        targets_enc = [lbl_enc.transform(x) for x in targets]
        targets_enc = np.array(targets_enc) 
        targets_enc = targets_enc + 1

        with open(f"assets/{config.ENCODER}", "wb") as f:
            pkl.dump({
                "lbl_enc": lbl_enc,
                "targets_enc": targets_enc,
            }, f)



    print(targets_enc)
    print(len(lbl_enc.classes_))

    train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    # # Shorten the dataset
    # train_dataset = torch.utils.data.Subset(train_dataset, range(0, len(train_dataset)//3))
    # test_dataset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset)//3))
    # print(len(train_dataset), len(test_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, verbose=True
    )

    os.makedirs("models", exist_ok=True)
    if os.path.exists(f"models/{config.MODEL_NAME}.pth"):
        model.load_state_dict(torch.load(f"models/{config.MODEL_NAME}.pth"))
    
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(f"assets/loss/", exist_ok=True)
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)
        valid_cap_preds = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_cap_preds.extend(current_preds)
        
        random_5_nums = np.random.randint(0, len(test_orig_targets), 5)
        pairs = list(zip(test_orig_targets, valid_cap_preds))
        pprint([pairs[i] for i in random_5_nums])
        pprint(f"Epoch {epoch}, train_loss={train_loss}, valid_loss={valid_loss}")
        
        torch.save(model.state_dict(), f"models/{config.MODEL_NAME}.pth")

        with open(f"assets/loss/{date}.csv", "a") as f:
            f.write(f"{epoch}, {train_loss}, {valid_loss}, {random_5_nums}\n")
            
        
        scheduler.step(valid_loss)



if __name__ == "__main__":
    run_training()
    # 75 values for each image: 0 to 20 (0 is unknown), 
    # representing unknown by °