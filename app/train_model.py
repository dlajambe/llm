import torch
from modules.llm import LLM, train_model
from modules.text_preprocessing import CharTokenizer
from config.hyperparameters import Hyperparams
import time
import os

start_time = time.perf_counter()

# Step 0 - Define hyperparameters.
# These parameters can be modified / tuned to improve model performance
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(Hyperparams.seed)

print('Device: {}'.format(device))
print('Model Hyperparameters:')
print('\tBlock size: {}'.format(Hyperparams.block_size))
print('\tNum embeddings: {}'.format(Hyperparams.n_embed))
print('\tNum heads: {}'.format(Hyperparams.n_heads))
print('\tHead size: {}'.format(Hyperparams.head_size))
print('\tFeed forward projection factor: {}\n'.
      format(Hyperparams.ff_proj_factor))

print('Training Hyperparameters:')
print('\tBatch size: {}'.format(Hyperparams.batch_size))
print('\tLearning rate: {}'.format(Hyperparams.lr))
print('\tMax training iterations: {}'.format(Hyperparams.max_training_iters))
print('\tEval interval: {}'.format(Hyperparams.eval_interval))
print('\tEval batches: {}\n'.format(Hyperparams.eval_batches))

# Step 1 - Import the data
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
char_set = sorted(set(text))
vocab_size = len(char_set)

# Step 2 - Create a tokenizer to decompose strings into character-level
# integer arrays
tokenizer = CharTokenizer(char_set)

# Step 3 - Generate matrices to store x and y data
data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
n_samples = len(data) - Hyperparams.block_size
print('Text length: {} characters'.format(n_samples))

# Step 3 - Split data into training and validation partitions
# training: Used to fit the model
# validation: Used to determine when to terminate training
train_frac = 0.9
n_train = int(n_samples * train_frac)

data_train = data[:n_train]
data_val = data[n_train:]

# # Step 4 - Create the model
model = LLM(Hyperparams.block_size, 
            Hyperparams.n_embed, 
            vocab_size, 
            Hyperparams.head_size, 
            Hyperparams.n_heads, 
            Hyperparams.ff_proj_factor, 
            Hyperparams.n_trans_blocks, 
            Hyperparams.dropout_frac)
model = model.to(device)

# Step 5 - Train the model
print('Training model...')
train_model(
    model, data_train, data_val, 
    Hyperparams.batch_size, 
    Hyperparams.block_size, 
    Hyperparams.lr, 
    Hyperparams.max_training_iters, 
    Hyperparams.eval_interval, 
    Hyperparams.eval_batches)

# Step 6 - Save the model parameters so they can be loaded in the future
# without retraining the model
output_dir = 'output/'
model_filename = 'llm.pt'
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)
torch.save(model.state_dict(), output_dir + model_filename)

# Step 6 - Have some fun with text generation
print('Generating output...')
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, 500)
print('Generated text: {}\n'.format(tokenizer.decode(generated.tolist()[0])))

end_time = time.perf_counter()

print('Script runtime: {} seconds'.format(end_time - start_time))