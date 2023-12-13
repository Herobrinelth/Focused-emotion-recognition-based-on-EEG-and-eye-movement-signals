import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import params_dropout_viz as params
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import EEGT

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_LOCATION = "../data"
DATASET_TEST_B = "test_b.npz"
DATASET_TRAIN_B = "train_b.npz"
DATASET_TEST_C = "test_c.npz"
DATASET_TRAIN_C = "train_c.npz"
RESULTS_LOCATION = "./viz_results"


def define_model(i):
    return EEGT(
        params.ENC_INPUT_LEN,
        params.CLASSES,
        params.ENC_INPUT_DIM,
        params.ENC_DEPTH[i],
        params.ENC_HEADS[i],
        params.ENC_MLP_DIM[i],
        params.ENC_POOL,
        params.ENC_DIM_HEAD[i],
        params.ENC_DROPOUT,
        params.ENC_EMB_DROPOUT,
    )


def test(i):
    # Generate the model
    model = define_model(i).to(device)
    model.load_state_dict(
        torch.load(f"{RESULTS_LOCATION}/best{i}.pth", map_location=device)["model"]
    )

    # Load dataset
    if i <= 14:
        test_npz = np.load(f"{DATASET_LOCATION}/{DATASET_TEST_B}")
        test_data_x = np.swapaxes(test_npz["X"][i], 0, 1)
        test_data_y = test_npz["Y"][i] + 1

        test_data_x = test_data_x[test_data_y == 2]
        test_data_y = test_data_y[test_data_y == 2]

        train_npz = np.load(f"{DATASET_LOCATION}/{DATASET_TRAIN_B}")
        train_data_x = np.swapaxes(train_npz["X"][i], 0, 1)
        train_data_y = train_npz["Y"][i] + 1

        train_data_x = train_data_x[train_data_y == 2]
        train_data_y = train_data_y[train_data_y == 2]
    
    if i > 14:
        test_npz = np.load(f"{DATASET_LOCATION}/{DATASET_TEST_C}")
        test_data_x = np.swapaxes(test_npz["X"][i-15], 0, 1)
        test_data_y = test_npz["Y"][i-15] + 1

        test_data_x = test_data_x[test_data_y == 2]
        test_data_y = test_data_y[test_data_y == 2]

        train_npz = np.load(f"{DATASET_LOCATION}/{DATASET_TRAIN_C}")
        train_data_x = np.swapaxes(train_npz["X"][i-15], 0, 1)
        train_data_y = train_npz["Y"][i-15] + 1

        train_data_x = train_data_x[train_data_y == 2]
        train_data_y = train_data_y[train_data_y == 2]

    viz_data_x = np.concatenate((test_data_x, train_data_x))
    viz_data_y = np.concatenate((test_data_y, train_data_y))

    # Prepare dataloader
    test_data = TensorDataset(
        torch.Tensor(viz_data_x), torch.LongTensor(viz_data_y)
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=params.BATCH_SIZE, shuffle=True
    )

    # Test model
    model.eval()
    attn = [torch.zeros(63, 63)] * 4
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            _, attn_mat = model(data)

            for j in range(4):
                attn[j] = attn[j].add(attn_mat[j])

    for k in range(4):
        attn[k] = torch.div(attn[k], len(test_loader))

    return attn


if __name__ == "__main__":

    final_attn = [torch.zeros(63, 63)] * 4
    for i in range(30):
        print(i)
        # Initialize seed
        np.random.seed(params.SEED)
        torch.manual_seed(params.SEED)
        torch.cuda.manual_seed(params.SEED)
        torch.cuda.manual_seed_all(params.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        attn = test(i)

        for j in range(4):
            final_attn[j] = final_attn[j].add(attn[j])

    for k in range(4):
        final_attn[k] = torch.div(final_attn[k], 30)
        np.savetxt(f'attn_mat_{k}.txt', final_attn[k].numpy(), fmt='%1.4f')