import torch
from modules.llm import LLM, train_model
from modules.text_preprocessing import CharTokenizer
import time

start_time = time.perf_counter()

# Step 0 - Define hyperparameters.
# These parameters can be modified / tuned to improve model performance
device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 256
n_embed = 32
n_heads = 4
ff_proj_factor = 4
if n_embed % n_heads != 0:
    raise ValueError(
        'Embedding dimension must be divisible by number of heads')
head_size = int(n_embed / n_heads)
n_trans_blocks = 4

batch_size = 64
lr = 1e-3
max_training_iters = 10000
seed = 1337
eval_interval = 250
eval_batches = 200
dropout_frac = 0.2

torch.manual_seed(seed)

print('Device: {}'.format(device))
print('Model Hyperparameters:')
print('\tBlock size: {}'.format(block_size))
print('\tNum embeddings: {}'.format(n_embed))
print('\tNum heads size: {}'.format(n_heads))
print('\tHead size: {}'.format(head_size))
print('\tFeed forward projection factor: {}\n'.format(ff_proj_factor))

print('Training Hyperparameters:')
print('\tBatch size: {}'.format(batch_size))
print('\tLearning rate: {}'.format(lr))
print('\tMax training iterations: {}'.format(max_training_iters))
print('\tEval interval: {}'.format(eval_interval))
print('\tEval batches: {}\n'.format(eval_batches))

# Step 1 - Import the data
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
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
train_frac = 0.9
n_train = int(n_samples * train_frac)

data_train = data[:n_train]
data_val = data[n_train:]

# # Step 4 - Create the model
model = LLM(block_size, n_embed, vocab_size, 
            head_size, n_heads, ff_proj_factor, n_trans_blocks, dropout_frac)
model = model.to(device)

# Step 5 - Train the model
print('Training model...')
train_model(
    model, data_train, data_val, batch_size, block_size, 
    lr, max_training_iters, eval_interval, eval_batches)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Step 6 - Save the model parameters so they can be loaded in the future
# without retraining the model
torch.save(model.state_dict(), 'output/llm.pt')

# Step 6 - Have some fun with text generation
print('Generating output...')
generated = model.generate(context, 500)
print('Generated text: {}\n'.format(tokenizer.decode(generated.tolist()[0])))

end_time = time.perf_counter()

print('Script runtime: {} seconds'.format(end_time - start_time))