import torch
import torch.nn.functional as F
from models import CharModel
from text_preprocessing import Encoder
from data_helpers import NGramDataSet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

# Step 0 - Define hyperparameters.
# These parameters can be modified / tuned to improve model performance
block_size = 8
batch_size = 32
embedding_dim = 100
print('Block size: {}'.format(block_size))
print('Batch size: {}'.format(batch_size))
print('Embedding dim: {}'.format(embedding_dim))

# Step 1 - Import the data
with open('data/wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
char_set = sorted(set(text))
vocab_size = len(char_set)

# Step 2 - Create an encoder to decompose strings into integer arrays
encoder = Encoder(char_set)
data = encoder.encode(text)
print('Unencoded data sample: {}'.format(text[:block_size]))
print('Encoded data sample: {}'.format(data[:block_size]))

# Step 3 - Generate matrices to store X and y data
n_samples = len(data) - block_size
X = torch.tensor([data[i:block_size + i] for i in range(n_samples)])
y = torch.tensor([data[block_size + i] for i in range(n_samples)])

# Step 3 - Split data into training and validation partitions
# training: Used to fit the model
# validation: Used to determine the optimal point to terminate training
train_frac = 0.8
n_train = int(n_samples * train_frac)

# The data are partitioned chronologically to preserve the char sequence
train = torch.tensor([i < n_train for i in range(n_samples)])
val = [i >= n_train for i in range(n_samples)]

data_train = NGramDataSet(X[train], y[train])
data_val = NGramDataSet(X[val], y[val])

# Step 4 - Create the model
model = CharModel(vocab_size, embedding_dim)

# Step 5 - Train the model