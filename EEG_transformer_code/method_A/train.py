import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import params_full_c as params
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from model import EEGT

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_LOCATION = "../data"
DATASET_TRAIN = "train_c.npz"
DATASET_TEST = "test_c.npz"
RESULTS_LOCATION = "./results"


def define_model(i):
    return EEGT(
        params.ENC_INPUT_LEN,
        params.CLASSES,
        params.ENC_INPUT_DIM,
        params.ENC_DEPTH[i],
        params.ENC_HEADS,
        params.ENC_MLP_DIM[i],
        params.ENC_POOL,
        params.ENC_DIM_HEAD[i],
        params.ENC_DROPOUT,
        params.ENC_EMB_DROPOUT,
    )


def train(i):
    # Generate the model
    model = define_model(i).to(device)

    # Generate the optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=params.LR[i], weight_decay=params.WEIGHT_DECAY[i]
    )

    # Load dataset
    train_npz = np.load(f"{DATASET_LOCATION}/{DATASET_TRAIN}")
    test_npz = np.load(f"{DATASET_LOCATION}/{DATASET_TEST}.npz")
    train_data = TensorDataset(
        torch.Tensor(np.swapaxes(train_npz["X"][i], 0, 1)),
        torch.LongTensor(train_npz["Y"][i] + 1),
    )
    test_data = TensorDataset(
        torch.Tensor(np.swapaxes(test_npz["X"][i], 0, 1)),
        torch.LongTensor(test_npz["Y"][i] + 1),
    )
    train_loader = DataLoader(
        dataset=train_data, batch_size=params.BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=params.BATCH_SIZE, shuffle=True
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0

    for epoch in range(params.EPOCHS):
        # Train model
        model.train()
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output, _ = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test model
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = deepcopy(model.state_dict())
            best_optimizer = deepcopy(optimizer.state_dict())

    return best_accuracy, best_model, best_optimizer


if __name__ == "__main__":

    for i in range(15):

        # Initialize seed
        np.random.seed(params.SEED)
        torch.manual_seed(params.SEED)
        torch.cuda.manual_seed(params.SEED)
        torch.cuda.manual_seed_all(params.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        accuracy, model, optimizer = train(i)
        print(f"student: {i} | accuracy: {accuracy}")

        # Save best results
        torch.save(
            {"model": model, "optimizer": optimizer}, f"{RESULTS_LOCATION}/best{i}.pth"
        )
