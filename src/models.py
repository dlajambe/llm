import torch
import torch.nn as nn
import torch.nn.functional as F
from data_helpers import get_batch

class BiGramModel(nn.Module):
    def __init__(self, vocab_size):
        super(BiGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        if type(x) != torch.Tensor:
            raise ValueError('Expected Tensor, received {}'.format(type(x)))
        
        logits = self.embeddings(x)

        # nn.Embedding outputs a B x T x C tensor, where:
        # B = batch size, i.e. number of samples (batch_size)
        # T = time, i.e. the length of each sample (block_size)
        # C = channels, i.e. the number of embeddings
        B, T, C = logits.shape

        # Pytorch's cross entropy loss function requires a N x C tensor
        logits = logits.view(B*T, C)
        loss = None
        if y != None:
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, context: torch.Tensor, output_length: int) -> list:
        output = context.clone()
        for i in range(output_length):
            logits, _ = self.forward(output, None)
            probs = F.softmax(logits, dim=-1)

            # The char with the highest probability is retained
            # Could also use torch.multinomial() to pick next char
            next_idx = torch.multinomial(probs[[-1]], num_samples=1)
            output = torch.cat((output, next_idx), dim=1)
        return output
    
    def train(self, data_train: torch.Tensor, 
              batch_size: int, block_size: int) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        for _ in range(5):
            xb, yb = get_batch(data_train, batch_size, block_size)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = self.forward(xb, yb)
            loss.backward()
            optimizer.step()

    def predict(self) -> torch.Tensor:
        pass

class CharModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CharModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_1 = nn.Linear(embedding_dim, 500)
        self.hidden_2 = nn.Linear(500, 500)
        self.output = nn.Linear(500, vocab_size)

    def forward(self, X: torch.Tensor) -> None:
        pass
    
    def generate(self, idx, output_length):
        pass

    def calc_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        pass

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

    def predict(self) -> torch.Tensor:
        pass