import torch
import torch.nn as nn
import torch.nn.functional as F
from data_helpers import get_batch, create_matrices

def train_model(model: nn.Module, 
                data_train: torch.Tensor, 
                data_val: torch.Tensor,
                batch_size: int, block_size: int) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x_val, y_val = create_matrices(data_val, block_size)
    x_train, y_train = create_matrices(data_train, block_size)
    eval_interval = 100

    def evaluate_loss(n_batches: int):
        # TODO: Check what the model's state is before modifying so
        # that the state is only modified if necessary

        model.eval()
        # TODO: Make model evaluation more efficient by calculating
        # mean training (and possibly validation) loss during 
        # backpropagation step 
        with torch.no_grad():
            _ , loss_train = model.forward(x_train, y_train)
            _ , loss_val = model.forward(x_val, y_val)
            print('\t{0} batches\tLoss (train):{1}\tLoss (val): {2}'.
                    format(n_batches, loss_train.item(), loss_val.item()))
        model.train()

    evaluate_loss(0)
    for i in range(500):
        xb, yb = get_batch(data_train, batch_size, block_size)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model.forward(xb, yb)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            evaluate_loss(i + 1)
    
    # TODO: Check model state at beginning of this function so that the 
    # state is only changed if necessary
    model.eval()
            

class BiGramModel(nn.Module):
    def __init__(self, block_size: int, embedding_dim: int, vocab_size: int):
        super(BiGramModel, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, 
                device='cpu') -> torch.Tensor:
        if type(x) != torch.Tensor:
            raise ValueError('Expected Tensor, received {}'.format(type(x)))
        
        # x enters as a B x T tensor, where:
        # B = batch size, i.e. number of samples (batch_size)
        # T = time, i.e. the length of each sample (block_size)
        B, T = x.shape

        x = self.token_embeddings(x)

        # The embedding layers add a dimension (B x T x E), where:
        # B = batch size, i.e. number of samples (batch_size)
        # T = time, i.e. the length of each sample (block_size)
        # E = embedding dimension
        
        logits = self.fc(x)
        # logits is a B x T x C tensor, where:
        # B = batch size, i.e. number of samples (batch_size)
        # T = time, i.e. the length of each sample (block_size)
        # C = channels, i.e. the vocab size

        _, _, C = logits.shape

        # Pytorch's cross entropy loss function requires a N x C tensor
        logits = logits.view(B*T, C)
        loss = None
        if y != None:
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        
        return logits, loss
    
    def generate(
            self, context: torch.Tensor, output_length: int) -> torch.long:
        output = context.clone()
        for i in range(output_length):
            logits, _ = self.forward(output)
            probs = F.softmax(logits, dim=-1)

            # The char with the highest probability is retained
            # Could also use torch.multinomial() to pick next char
            next_idx = torch.multinomial(probs[[-1]], num_samples=1)
            output = torch.cat((output, next_idx), dim=1)
        return output
    
    def predict(self) -> torch.Tensor:
        pass