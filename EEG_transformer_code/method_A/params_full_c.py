# Transformer parameters
# Classes (3 emotions: -1 negative, 0 neutral, 1 positive)
CLASSES = 2
# Optimizer type
OPTIMIZER = "AdamW"
# Dropout rate
ENC_DROPOUT = 0.0
# Position embedding dropout rate
ENC_EMB_DROPOUT = 0.0
# Encoder pool type
ENC_POOL = "cls"
# Number of dimensions per token
ENC_INPUT_DIM = 5
# Number of tokens as input
ENC_INPUT_LEN = 62
# Number of encoder modules
ENC_DEPTH = [8, 8, 4, 8, 4, 4, 6, 4, 8, 8, 4, 8, 4, 4, 8]
# Number of multi-attention heads
ENC_HEADS = 2
# Number of dimensions per head
ENC_DIM_HEAD = [16, 16, 32, 8, 32, 8, 16, 16, 16, 32, 16, 8, 8, 32, 8]
# Number of MLP dimensions
ENC_MLP_DIM = [16, 8, 16, 16, 32, 8, 32, 8, 8, 32, 8, 16, 8, 16, 8]

# Training hyperparameters
# Training batch size
BATCH_SIZE = 32
# Training epochs
EPOCHS = 200
# Learning rate
LR = [1e-3, 1e-3, 1e-3, 3e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 3e-4, 1e-3, 3e-4]
# Weight decay
WEIGHT_DECAY = [1e-2, 1e-4, 1e-4, 1e-2, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-4, 1e-6, 1e-4, 1e-2, 1e-4, 1e-6]
# Random seed initialization
SEED = 3793269