import torch
import torch.nn as nn
class CharModel(nn.Module):
    def __init__(self, n_unique_chars, embedding_dim):
        super(CharModel, self).__init__()
        self.embeddings = nn.Embedding(n_unique_chars, embedding_dim)

    def forward(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass