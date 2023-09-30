import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGramModel(nn.Module):
    def __init__(self, vocab_size):
        super(BiGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, vocab_size)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if type(x) != torch.Tensor:
            raise ValueError('Expected Tensor, received {}'.format(type(x)))

        logits = self.embeddings(x)
        n_samples, sample_length, vocab_size = logits.shape

        # Reshaping in this way gives the embeddings for all characters
        # received in X
        logits = logits.view(n_samples*sample_length, vocab_size)
        return logits
    
    def generate(self, context: torch.Tensor, output_length: int) -> list:
        output = [int(context[-1, -1])]
        for i in range(output_length):
            x = torch.tensor([output[-1]])
            x = x.view(1, 1)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            # The char with the highest probability is retained
            # Could also use torch.multinomial() to pick next char
            output.append(int(torch.multinomial(probs, num_samples=1)))
        return output
    
    def train(self, X: torch.Tensor, targets: torch.Tensor) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        for i in range(5):
            optimizer.zero_grad(set_to_none=True)
            logits = self.forward(X)
            loss = self.loss_function(logits, targets)
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