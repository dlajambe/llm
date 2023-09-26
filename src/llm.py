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

# Step 3 - Split data into training and testing partitions

# Step 4 - Implement the bigram model