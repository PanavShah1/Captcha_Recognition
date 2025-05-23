from tqdm import tqdm #
import torch
import src.config as config

def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    # tk = tqdm(data_loader, total=len(data_loader))
    tk = data_loader
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)

def eval_fn(model, data_loader):
    model.eval()
    with torch.no_grad():
        fin_loss = 0
        fin_preds = []
        # tk = tqdm(data_loader, total=len(data_loader))
        tk = data_loader
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)
            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)

