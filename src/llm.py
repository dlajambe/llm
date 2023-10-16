import torch
import torch.nn.functional as F
from models import BiGramModel, CharModel
from text_preprocessing import CharTokenizer
from torch.utils.data import DataLoader
import time

start_time = time.perf_counter()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

# Step 0 - Define hyperparameters.
# These parameters can be modified / tuned to improve model performance
block_size = 8
batch_size = 32
embedding_dim = 100
seed = 42

torch.manual_seed(seed)
print('Block size: {}'.format(block_size))
print('Batch size: {}'.format(batch_size))
print('Embedding dim: {}'.format(embedding_dim))

# Step 1 - Import the data
with open('data/wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
char_set = sorted(set(text))
vocab_size = len(char_set)

# Step 2 - Create an tokenizer to decompose strings into integer arrays
tokenizer = CharTokenizer(char_set)

# Step 3 - Generate matrices to store X and y data
data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
n_samples = len(data) - block_size
print('Text length: {} characters'.format(n_samples))

# Step 3 - Split data into training and validation partitions
# training: Used to fit the model
# validation: Used to determine when to terminate training
train_frac = 0.8
n_train = int(n_samples * train_frac)

data_train = data[:n_train]
data_val = data[n_train:]

# # Step 4 - Create the model
model = BiGramModel(vocab_size)
model = model.to(device)

# Step 5 - Train the model
print('Training model...')
model.train(data_train, batch_size, block_size)
context = torch.zeros((1, 1), dtype=torch.long, device=device)


# # Step 6 - Have some fun with text generation
print('Generating output...')
generated = model.generate(context, 100)
print('Generated text: {}\n'.format(tokenizer.decode(generated.tolist()[0])))

end_time = time.perf_counter()

print('Script runtime: {} seconds'.format(end_time - start_time))