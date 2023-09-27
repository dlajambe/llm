import torch

# Step 1 - Import the data
with open('data/wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)

# Step 2 - Create an encoder to decompose strings into integer arrays
class Encoder():
    def __init__(self, all_chars):
        self.str_to_int = {char: i for i, char in enumerate(all_chars)}
        self.int_to_str = {
            value: key for value, key in enumerate(self.str_to_int)}

    def encode(self, string):
        return [self.str_to_int[char] for char in string]

    def decode(self, int_arr):
        return ''.join([self.int_to_str[i] for i in int_arr])
    
encoder = Encoder(chars)
data = encoder.encode(text)
print(data[:50])

# Step 3 - Split data into training and testing partitions
# training: Used to fit the model
# testing: Used to determine when terminate fitting
train_frac = 0.8
n_train = int(len(data) * train_frac)

train = [i < n_train for i in range(len(data))]
test = [i >= n_train for i in range(len(data))]

# Step 4 - Implement the bigram model