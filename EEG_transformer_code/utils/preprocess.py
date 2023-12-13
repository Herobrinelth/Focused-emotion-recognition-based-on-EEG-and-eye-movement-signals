import scipy.io as sio
import numpy as np

students = []
for i in range(0, 15):
    students.append(sio.loadmat(f"./data/{i+1}a.mat"))
label = sio.loadmat("./data/label.mat")

trainX = []
trainY = []
testX = []
testY = []

for student in students:
    # 1. For each student (9 train, 6 test)
    tempX = []
    tempY = []

    for clip_index in range(0, 15):
        # 2. Calculate length of current video clip
        # 3. Get DE feature of current video clip
        # 4. Get label of current video clip
        length = student[f"de_LDS{clip_index+1}"].shape[1]
        tempX.append(student[f"de_LDS{clip_index+1}"])
        tempY.append([label["label"][0][clip_index]] * length)

    trainX.append(np.concatenate(tempX[:9], axis=1))
    trainY.append(np.concatenate(tempY[:9], axis=0))
    testX.append(np.concatenate(tempX[-6:], axis=1))
    testY.append(np.concatenate(tempY[-6:], axis=0))

np.savez_compressed(
    "train",
    X=np.asarray(trainX, dtype=np.float32),
    Y=np.asarray(trainY, dtype=np.float32),
)
np.savez_compressed(
    "test", X=np.asarray(testX, dtype=np.float32), Y=np.asarray(testY, dtype=np.float32)
)
