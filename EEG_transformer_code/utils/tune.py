import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import EEGT
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_LOCATION = "./data"
DATASET_TRAIN = "train_a.npz"
DATASET_TEST = "test_a.npz"
RESULTS_LOCATION = "./results"

CLASSES = 3
ENC_DROPOUT = 0.0
ENC_EMB_DROPOUT = 0.0
ENC_POOL = "cls"
ENC_INPUT_DIM = 5
ENC_INPUT_LEN = 62
ENC_HEADS = 2
BATCH_SIZE = 32
EPOCHS = 10
SEED = 3793269


def define_model(i, depth, mlp_dim, dim_head):
    return EEGT(
        ENC_INPUT_LEN,
        CLASSES,
        ENC_INPUT_DIM,
        depth,
        ENC_HEADS,
        mlp_dim,
        ENC_POOL,
        dim_head,
        ENC_DROPOUT,
        ENC_EMB_DROPOUT,
    )


def train(i, alpha, weight_decay, depth, mlp_dim, dim_head, lr):
    # Generate the model
    model = define_model(i, depth, mlp_dim, dim_head).to(device)

    # Generate the optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Load dataset
    train_npz = np.load(f"{DATASET_LOCATION}/{DATASET_TRAIN}")
    test_npz = np.load(f"{DATASET_LOCATION}/{DATASET_TEST}")
    train_data = TensorDataset(
        torch.Tensor(np.swapaxes(train_npz["X"][i], 0, 1)),
        torch.LongTensor(train_npz["Y"][i] + 1),
    )
    test_data = TensorDataset(
        torch.Tensor(np.swapaxes(test_npz["X"][i], 0, 1)),
        torch.LongTensor(test_npz["Y"][i] + 1),
    )
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    # Loss function
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    best_accuracy = 0

    for epoch in range(EPOCHS):
        # Train model
        model.train()
        for data, label in train_loader:
            # Prepare dropout mask
            keep = np.random.choice([5, 10, 20, 30, 40, 50, 62])
            mask = np.zeros((62, 5), dtype=int)
            mask[:keep, :] = 1
            np.random.shuffle(mask)
            mask = mask.astype(bool)
            mask = np.broadcast_to(mask, data.shape)
            data = (data * mask).to(device)
            label1 = label.to(device)
            label2 = label.unsqueeze(1).repeat(1, keep).to(device)

            output1, output2 = model(data)

            pred1 = output1.argmax(dim=1, keepdim=True)
            loss1 = criterion1(output1, label1)
            loss2 = criterion2(output2.permute(0, 2, 1), label2)
            loss = loss1 + loss2 * alpha

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
                
                output, _ = model(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

    return best_accuracy


if __name__ == "__main__":

    for i in range(15):
        best_accuracy = 0

        for lr in [3e-4, 1e-3]:
            for dim_head in [8, 16, 32]:
                for mlp_dim in [8, 16, 32]:
                    for depth in [4, 6, 8]:
                        for weight_decay in [1e-6, 1e-4, 1e-2]:
                            for alpha in [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]:

                                # Initialize seed
                                np.random.seed(SEED)
                                torch.manual_seed(SEED)
                                torch.cuda.manual_seed(SEED)
                                torch.cuda.manual_seed_all(SEED)
                                torch.backends.cudnn.deterministic = True
                                torch.backends.cudnn.benchmark = False

                                accuracy = train(i, alpha, weight_decay, depth, mlp_dim, dim_head, lr)

                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_alpha = alpha
                                    best_lr = lr
                                    best_weight_decay = weight_decay
                                    best_dim_head = dim_head
                                    best_mlp_dim = mlp_dim
                                    best_depth = depth

        print(
            f"RESULTS: best_accuracy: {best_accuracy} | alpha: {best_alpha} | weight_decay: {best_weight_decay} | depth: {best_depth} | mlp_dim: {best_mlp_dim} | dim_head: {best_dim_head} | lr: {best_lr}"
        )
