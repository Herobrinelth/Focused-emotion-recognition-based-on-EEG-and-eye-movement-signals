import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import params_full_c as params
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import EEGT

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_LOCATION = "../data"
DATASET_FILENAME = "test_c.npz"
RESULTS_LOCATION = "./results/c"


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


def test(i, keep):
    # Generate the model
    model = define_model(i).to(device)
    model.load_state_dict(
        torch.load(f"{RESULTS_LOCATION}/best{i}.pth", map_location=device)["model"]
    )

    # Load dataset
    test_npz = np.load(f"{DATASET_LOCATION}/{DATASET_FILENAME}")
    test_data_x = np.swapaxes(test_npz["X"][i], 0, 1)
    test_data_y = test_npz["Y"][i] + 1

    accuracies = []

    for j in range(50):
        # Prepare dropout mask
        mask = np.zeros((62, 5), dtype=int)
        mask[:keep, :] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)
        mask = np.broadcast_to(mask, test_data_x.shape)

        # Apply mask
        masked_test_data_x = test_data_x * mask

        # Prepare dataloader
        test_data = TensorDataset(
            torch.Tensor(masked_test_data_x), torch.LongTensor(test_data_y)
        )
        test_loader = DataLoader(
            dataset=test_data, batch_size=params.BATCH_SIZE, shuffle=True
        )

        # Test model
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)
                test_output = model(data)
                pred = test_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)
        accuracies.append(accuracy)

    return sum(accuracies) / len(accuracies)


if __name__ == "__main__":
    for keep in [5, 10, 20, 30, 40, 50, 62]:
        accuracies = []

        for i in range(15):
            # Initialize seed
            np.random.seed(params.SEED)
            torch.manual_seed(params.SEED)
            torch.cuda.manual_seed(params.SEED)
            torch.cuda.manual_seed_all(params.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            accuracy = test(i, keep)
            accuracies.append(accuracy)
            print(f"student: {i} | avg_accuracy: {accuracy}")

        print(f"keep: {keep} | final accuracy: {sum(accuracies) / len(accuracies)}")