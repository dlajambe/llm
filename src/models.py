import torch
import torch.nn as nn
import torch.nn.functional as F
from data_helpers import get_batch, create_matrices

def train_model(model: nn.Module, 
                data_train: torch.Tensor, 
                data_val: torch.Tensor,
                batch_size: int, 
                block_size: int,
                lr: float) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    eval_interval = 250
    eval_batches = 200

    def estimate_loss(eval_batches: int, n_batches: int):
        model.eval()

        # Gradients are not required during evaluation, so they are 
        # turned off to improve calculation speed
        with torch.no_grad():
            losses_train = torch.zeros(eval_batches)
            losses_val = torch.zeros(eval_batches)
            for i in range(eval_batches):
                x_train, y_train = get_batch(data_train, batch_size, block_size)
                x_val, y_val = get_batch(data_val, batch_size, block_size)
                _ , loss_train = model.forward(x_train, y_train)
                _ , loss_val = model.forward(x_val, y_val)
                losses_train[i] = loss_train.item()
                losses_val[i] = loss_val.item()
        print('\t{0} batches\tLoss (train):{1}\tLoss (val): {2}'.
              format(n_batches, losses_train.mean(), losses_val.mean()))
        model.train()

    estimate_loss(eval_batches, 0)
    for i in range(5000):
        xb, yb = get_batch(data_train, batch_size, block_size)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model.forward(xb, yb)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            estimate_loss(eval_batches, i + 1)
    
    # TODO: Check model state at beginning of this function so that the 
    # state is only changed if necessary
    model.eval()

class Head(nn.Module):
    def __init__(self, 
                 block_size: int, 
                 embedding_dim: int, 
                 head_size: int):
        super(Head, self).__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        self.register_buffer(
            'mask', torch.tril(torch.ones(block_size, block_size)))

        self.embedding_dim = embedding_dim
        self.head_size = head_size

    def forward(self, x: torch.Tensor):
        # x dimensions: (batch_size, block_size, emedding_dim)
        B, T, E = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # (B, T, head_size) x (B, head_size, T) = (B, T, T)
        weights = q @ k.transpose(-2, -1)

        # Normalizing the weights prevents values from ballooning as the
        # head size increases, which would cause the probabilities to
        # "sharpen" during the softmax step
        weights = weights * (1 / k.shape[-1] ** (-0.5))

        # Ensures that information from the future can not communicate
        # to the past
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Conversion to probabilities with softmax ensures that all 
        # weight vectors sum to 1.0, standardizing the scaling
        # regardless of the input
        weights = F.softmax(weights, -1)

        # (B, T, T) x (B, T, head_size) x  = (B, T, head_size)
        output = weights @ v
        return output



class LLM(nn.Module):
    def __init__(self, 
                 block_size: int, 
                 embedding_dim: int, 
                 vocab_size: int,
                 head_size: int):
        super(LLM, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(block_size, embedding_dim)
        self.head = Head(block_size, embedding_dim, head_size)
        self.fc = nn.Linear(head_size, vocab_size)
        self.register_buffer('range', torch.arange(block_size))
        self.block_size = block_size

    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        if type(x) != torch.Tensor:
            raise ValueError('Expected Tensor, received {}'.format(type(x)))
        
        # x enters as a B x T tensor, where:
        # B = batch size, i.e. number of samples (batch_size)
        # T = time, i.e. the length of each sample (block_size)
        B, T = x.shape

        # Character information is captured through the token embedding,
        # whereas positional information is captured through the
        # poisition embedding
        
        tok_vect = self.token_embeddings(x) # (B, T, embedding_dim)

        # # (T, embedding_dim) 
        pos_vect = self.positional_embeddings(self.range[:T])

        # After adding the token and position vectors together, x 
        # contains both types of information, making it more useful
        # for predicting the next token
        x = tok_vect + pos_vect # (B, T, embedding_dim)

        att = self.head(x) # (B, T, head_size)
        
        logits = self.fc(att) # (B, T, vocab_size)

        _, _, C = logits.shape

        # Pytorch's cross entropy loss function requires a N x C tensor
        logits = logits.view(B*T, C)
        loss = None
        if y != None:
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        
        return logits, loss
    
    def generate(self, 
                 context: torch.Tensor, 
                 output_length: int) -> torch.Tensor:
        if len(context.shape) != 2:
            raise ValueError('Invalid context provided, expected a 2D Tensor')
        
        output = context.clone() # (B, T)
        for i in range(output_length):

            # The positional embeddings are only defined up to
            # block_size, so only the last block_size characters are
            # retained when calling self.forward()
            logits, _ = self.forward(output[:, -self.block_size:]) # (B*T, C)
            probs = F.softmax(logits[[-1], :], dim=-1) # (1, C)

            # The char with the highest probability is retained
            # Could also use torch.multinomial() to pick next char
            # (1, 1)
            next_idx = torch.multinomial(probs[[-1]], num_samples=1)
            output = torch.cat((output, next_idx), dim=1)
        return output
    
    def predict(self) -> torch.Tensor:
        pass