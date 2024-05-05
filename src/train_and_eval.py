import torch
import copy
import json
from collections import defaultdict
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score


def train_model(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    total_preds = []
    total_targets = []

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        total_preds.append(preds.cpu().detach())
        total_targets.append(targets.cpu().detach())
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)

    f1 = f1_score(total_targets, total_preds, average='weighted')
    return f1, np.mean(losses)


def eval_model(model, data_loader, criterian, device, n_examples):
    model.eval()
    eval_loss = []
    total_preds = []
    total_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            # forward prop
            predictions = model(input_ids, attention_masks)
            loss = criterian(predictions, targets)
            _, pred_classes = torch.max(predictions, dim=1)

            eval_loss.append(loss.item())

            total_preds.append(pred_classes.cpu().detach())
            total_targets.append(targets.cpu().detach())

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)

    f1 = f1_score(total_targets, total_preds, average='weighted')
    return f1, np.mean(eval_loss)


def train(model, train_loader, val_loader, criterian, opt, device, epochs, path_to_save_model):
    history = defaultdict(list)
    best_f1 = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")

        train_f1, train_loss = train_model(
            model, train_loader, criterian, opt, device, len(train_loader))

        val_f1, val_loss = eval_model(
            model, val_loader, criterian, device, len(val_loader))

        history['train_f1'].append(train_f1)
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)

        print(
            f"Train loss: {train_loss} | Train F1: {train_f1}")
        print(
            f"Val loss: {val_loss} | Val F1: {val_f1}")

        if val_f1 > best_f1:
            best_model_name = f'{
                path_to_save_model}/best_model_state_{val_f1}.bin'
            torch.save(model.state_dict(), best_model_name)
            best_model_wts = copy.deepcopy(model.state_dict())
            best_f1 = val_f1
    return history, best_f1, best_model_wts


def get_predictions(model, data_loader, device):
    model = model.eval()

    corpus_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    results = []

    with torch.no_grad():
        for d in tqdm(data_loader):

            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            corpus_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

            for text, pred, real in zip(texts, preds, targets):
                results.append(
                    {"text": text, "prediction": pred.item(), "real_value": real.item()})

    predictions = torch.stack(predictions).to(device)
    prediction_probs = torch.stack(prediction_probs).to(device)
    real_values = torch.stack(real_values).to(device)

    with open('/home/user/Documents/Dataset/dataset/predictions.json', 'w') as f:
        json.dump(results, f)

    return corpus_texts, predictions, prediction_probs, real_values
