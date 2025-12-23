import numpy as np
import torch
import pandas as pd
import os

from torch import nn
from fastai.callback.all import ShowGraphCallback
from tsai.all import get_ts_dls, Learner


def sliding_windows(df, parameters, window_size, step_size):
    
    df_clean = df.dropna(subset=parameters).reset_index(drop=True)
    n = len(df_clean)
    
    windows = []
    timestamps = []
    indices = []
    
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = df_clean[parameters].iloc[start:end].values.astype(np.float32)
        windows.append(window.T)  # transpose to [channels, seq_len]
        
        timestamps.append(df_clean["DateTime"].iloc[start:end])
        indices.append((start, end))
    
    #X = torch.tensor(np.stack(windows), dtype=torch.float32)  # [num_windows, channels, seq_len]
    
    return windows, timestamps, indices

def get_window_labels(df_artifact, timestamps, threshold):

    labels = []
    artifact_times = df_artifact["DateTime"].sort_values()

    for timestamp in timestamps:
        start_time = timestamp.iloc[0]
        end_time = timestamp.iloc[-1]

        # check if any artifact timestamp falls within window
        artifact_count  = ((artifact_times >= start_time) & (artifact_times <= end_time)).sum()
        window_label = int(artifact_count / len(timestamp) >= threshold)
        labels.append(window_label)
    
    return labels

def recall_label1(inp, targ):
    # Ensure shape [batch, 1]
    if inp.dim() == 3: 
        inp = inp.squeeze(-1)
    if targ.dim() == 3: 
        targ = targ.squeeze(-1)
    
    pred = (torch.sigmoid(inp) > 0.5).float()
    targ = targ.float()
    
    TP = ((pred == 1) & (targ == 1)).sum()
    FN = ((pred == 0) & (targ == 1)).sum()
    
    recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)
    return recall


def get_features(file_directory_res, file_directory_art, patient_ID, parameters,
                 window_size, step_size, window_threshold):
    
    # Getting the non clean residuals file
    file = patient_ID + "_non_clean.csv"
    file_path = os.path.join(file_directory_res, file)
    df_residuals= pd.read_csv(file_path)
    df_residuals = df_residuals.dropna(subset=parameters).reset_index(drop=True)
    
    # Getting the artifacts file with timestamps
    file = patient_ID + "_Artifact.csv"
    file_path = os.path.join(file_directory_art, file)
    df_artifact = pd.read_csv(file_path)
    
    artifact_times = set(df_artifact['DateTime'])
    df_residuals['true_artifacts'] = df_residuals['DateTime'].isin(artifact_times).astype(int)

    windows, timestamps, indices = sliding_windows(df_residuals, parameters, window_size, step_size)
    labels = get_window_labels(df_artifact, timestamps, window_threshold)
    X = torch.tensor(np.stack(windows), dtype=torch.float32)  # [num_windows, channels, seq_len]
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    result = {}
    result['X'] = X
    result['y'] = y
    result['Residuals'] = df_residuals
    result['Window_timestamps'] = timestamps
    result['Window_indices'] = indices
    
    return result

def training_model(X, y, train_idx, valid_idx, 
                   model, epoch, batch_size, learning_rate):    
    # Create splits tuple
    splits = (train_idx, valid_idx)
    
    y_ = y.float().unsqueeze(-1)
    dls = get_ts_dls(X, y_, splits=splits, bs=batch_size)
    
    # -----------------------------
    # 2. Define PatchTST model
    # -----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 3. Loss and optimizer
    loss_func = nn.BCEWithLogitsLoss()  # for binary classification
    
    # 4. Create Learner
    learn = Learner(
        dls,
        model,
        loss_func=loss_func,
        metrics=[recall_label1],
        cbs=[ShowGraphCallback()]
    )
    
    # 5. Train the model
    learn.fit_one_cycle(epoch, learning_rate)

    result = {}
    result['Model'] = model
    result['DLS'] = dls
    result['Learner'] = learn

    return result

def get_prediction_labels(model, dls, pred_threshold):
    model.eval()
    window_preds = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Iterating over the DataLoader
    for x_batch, _ in dls.valid:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            logits = model(x_batch).squeeze(-1)        # shape [batch, 1]
            probs = torch.sigmoid(logits).cpu().numpy()  # convert to probability
            window_preds.extend(probs.flatten())      # flatten to 1D
    
    y_pred = (np.array(window_preds) >= pred_threshold).astype(int)

    return y_pred