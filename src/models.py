import torch
import torch.nn as nn
class CharModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CharModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, X: torch.Tensor) -> None:
        pass

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

    def predict(self) -> torch.Tensor:
        pass