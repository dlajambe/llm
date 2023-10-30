import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.data_helpers import get_batch

def train_model(model: nn.Module, 
                data_train: torch.Tensor, 
                data_val: torch.Tensor,
                batch_size: int, 
                block_size: int,
                lr: float,
                max_iters: int,
                eval_interval: int,
                eval_batches: int) -> None:
    """
    Trains a large language model with the provided hyperparameters.

    Parameters
    ----------
    data_train : Tensor
        A 1D tensor containing the data to be used to train the model.

    data_val : Tensor
        A 1D tensor containing the data to be used to evaluate the
        model's performance during training.

    batch_size : int
        The number of token sequences to be used in each batch of
        training data.

    block_size : int
        The number of tokens in each token sequence.

    lr : float
        The learning rate to be used during training.

    max_iters : int
        The maximum number training iterations to be executed before
        termination.

    eval_interval : int
        How often the model's performance should be evaluated, in number
        of batches.

    eval_batches : int
        The number of batches to use when evaluating the model's
        performance at each evaluation interval.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def estimate_loss(eval_batches: int, n_batches: int):
        model.eval()

        # Gradients are not required during evaluation, so they are 
        # turned off to improve calculation speed
        with torch.no_grad():
            losses_train = torch.zeros(eval_batches)
            losses_val = torch.zeros(eval_batches)
            for i in range(eval_batches):
                x_train, y_train = get_batch(
                    data_train, batch_size, block_size)
                x_val, y_val = get_batch(data_val, batch_size, block_size)
                _ , loss_train = model.forward(x_train, y_train)
                _ , loss_val = model.forward(x_val, y_val)
                losses_train[i] = loss_train.item()
                losses_val[i] = loss_val.item()
        print('\t{0} batches\tLoss (train):{1}\tLoss (val): {2}'.
              format(n_batches, losses_train.mean(), losses_val.mean()))
        model.train()

    estimate_loss(eval_batches, 0)
    for i in range(max_iters):
        xb, yb = get_batch(data_train, batch_size, block_size)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model.forward(xb, yb)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            estimate_loss(eval_batches, i + 1)

    model.eval()
# TODO: See if there is a way to remove all of these parameters 
# from the constructor functions of all classes

class Head(nn.Module):
    """
    Implements a single attention head, as described in the Attention is
    All You Need paper.

    Attributes
    ----------
    key : Linear
        Used to project the input into the key space, which indicates 
        what each input offers.

    query : Linear
        Used to project the input into the query space, which indicates
        what each input is looking for.

    value : Linear
        Used to project the input into the value space, which contains
        the actual values offered by an input.

    Methods
    -------
    forward(x)
        Returns the result of passing the input x through the attention
        head.
    """
    def __init__(self, 
                 block_size: int, 
                 n_embed: int, 
                 head_size: int,
                 dropout_frac: float) -> None:
        super(Head, self).__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_frac)

        self.register_buffer(
            'mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # to the past within each individual block of data
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Conversion to probabilities with softmax ensures that all 
        # weight vectors sum to 1.0, so that the magnitude of the
        # weights are standardized
        weights = F.softmax(weights, -1)

        # (B, T, T) x (B, T, head_size) x  = (B, T, head_size)
        output = weights @ v

        # Some of the interactions are turned off to prevent overfitting
        # during training
        output = self.dropout(output)

        return output

class MultiHead(nn.Module):
    """
    Implements a multi-headed attention block.

    Attributes
    ----------
    heads : ModuleList
        Contains the individual Head objects comprising the multi-headed
        attention block.

    Methods
    -------
    forward(x)
        Returns the result of passing the input x through the
        multi-headed attention block.
    """
    def __init__(self, 
                 block_size: int, 
                 n_embed: int, 
                 head_size: int,
                 n_heads: int,
                 dropout_frac: float) -> None:
        super(MultiHead, self).__init__()
        self.heads = nn.ModuleList(
            [Head(block_size, n_embed, head_size, dropout_frac) 
             for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_heads * head_size)
        self.dropout = nn.Dropout(dropout_frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class Feedforward(nn.Module):
    """
    Implements a simple feedforward block of a transformer model.

    Attributes
    ----------
    net : Sequential
        Contains the linear layers and ReLU activation function of the
        feedforward block.

    Methods
    -------
    forward(x)
        Returns the result of passing the input x through the
        feedforward block.
    """
    def __init__(
            self, 
            n_features: int, 
            proj_factor: int, 
            dropout_frac: float) -> None:
        super(Feedforward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, proj_factor * n_features),
            nn.ReLU(),
            nn.Linear(proj_factor * n_features, n_features),
            nn.Dropout(dropout_frac))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LayerNorm(nn.Module):
    """
    Standardizes the input Tensor to 0 mean and unit variance across
    the channel dimension. Includes optional gain and bias parameters
    that can be optimized during training.

    Attributes
    ----------
    eps : float
        Added to the square root term to ensure numerical stability

    gain : Parameter
        Contains the gain scaling factors to apply after standardization
        across the channels has occurred.

    bias : Parameter
        Contains the bias corrections to be added after standardization
        across the channels has occurred.
    
    Methods
    -------
    forward(x)
        Returns the result of passing the input x through the
        block.

    parameters()
        Returns the gain and bias attributes of the LayerNorm object.
    """
    def __init__(
            self, 
            dim: int, 
            eps: float=1e-5, 
            requires_grad: bool=True) -> None:
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.register_parameter(
            'gain', torch.nn.Parameter(torch.ones(dim), requires_grad))
        self.register_parameter(
            'bias', torch.nn.Parameter(torch.zeros(dim), requires_grad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        return (x - mean)/torch.sqrt(var + self.eps) * self.gain + self.bias
    
    def parameters(self):
        return [self.gain, self.bias]

class TransBlock(nn.Module):
    """
    Implements a single self-contained transformer block.

    Attributes
    ----------
    multi_head : MultiHead
        The multi-headed attention head block used to recognize patterns
        within the input sequence.

    ff : Feedforward
        A feedforward layer used to gather information from the output
        of the multi-head attention block.
    
    Methods
    -------
    forward(x)
        Returns the result of passing the input x through the
        block.
    """
    def __init__(self, 
                 block_size: int, 
                 n_embed: int, 
                 head_size: int,
                 n_heads: int,
                 ff_proj_factor: int,
                 dropout_frac: float):
        super(TransBlock, self).__init__()
        self.multi_head = MultiHead(
            block_size, n_embed, head_size, n_heads, dropout_frac)
        self.ff = Feedforward(n_heads*head_size, ff_proj_factor, dropout_frac)
        self.ln1 = LayerNorm(n_heads*head_size)
        self.ln2 = LayerNorm(n_heads*head_size)

    def forward(self, x: torch.Tensor):
        # Skip connections are added to improve model training
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class LLM(nn.Module):
    def __init__(self, 
                 block_size: int, 
                 n_embed: int, 
                 vocab_size: int,
                 head_size: int,
                 n_heads: int,
                 ff_proj_factor: int,
                 n_trans_blocks: int,
                 dropout_frac: float):
        super(LLM, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.positional_embeddings = nn.Embedding(block_size, n_embed)
        self.trans_blocks = nn.Sequential(
            *[TransBlock(
                block_size,
                n_embed,
                head_size,
                n_heads,
                ff_proj_factor,
                dropout_frac) for _ in range(n_trans_blocks)])
        self.ln = LayerNorm(head_size * n_heads)
        self.output = nn.Linear(head_size * n_heads, vocab_size)
        self.register_buffer('positions', torch.arange(block_size))
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
        
        tok_vect = self.token_embeddings(x) # (B, T, n_embed)

        # # (T, n_embed) 
        pos_vect = self.positional_embeddings(self.positions[:T])

        # After adding the token and position vectors together, x 
        # contains both types of information, making it more useful
        # for predicting the next token
        x = tok_vect + pos_vect # (B, T, n_embed)

        x = self.trans_blocks(x) # (B, T, head_size * n_heads)

        x = self.ln(x) # (B, T, head_size * n_heads)

        logits = self.output(x) # (B, T, vocab_size)

        _, _, C = logits.shape

        # Pytorch's cross entropy loss function requires an N x C tensor
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