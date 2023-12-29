import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from modules.data_helpers import sample_batch
class Head(nn.Module):
    """Implements a single attention head, as described in the Attention
    is All You Need paper.

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

    mask : Linear
        A lower triangular matrix used to vary the number of tokens used
        to generate attention scores within a sequence from 1 to
        block_size.

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
        """Passes the input tensor through the self-attention head,
        aggregating the interactions (attention scores) between the
        input tokens.

        The input tensor (x) must be of shape B x T x E, where:
        - B : Batch size, i.e. number of sequences
        - T : Block size, i.e. number of tokens per sequence
        - E : Embedding dimension, i.e. number of embedding channels

        To ensure the model can use input sequences shorter than 
        block_size at inference time, a lower triangular 'mask' is
        applied to the attention weight tensor (before softmaxing) to
        filter out a varying number of 'future' tokens within each
        block.

        An output tensor of shape B x T x head_size is produced.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        output : Tensor
            The output tensor of aggregated self-attention information
            across head_size dimensions.
        """
        # x dimensions: (batch_size, block_size, embedding_dim)
        B, T, E = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # (B, T, head_size) x (B, head_size, T) = (B, T, T)
        weights = q @ k.transpose(-2, -1)

        # The variance of the weight tensor is proportional to 
        # head_size, which causes the magnitude (both positive and 
        # negative) of the weights to increase as head_size increases. 
        # This causes the softmax output to "sharpen" towards a one-hot
        # encoding for large head_size values, which increases the
        # importance of strong connections relative to weak connections.
        # To ensure that head_size does not influence model performance,
        # the weight tensor must be normalized by dividing the weights
        # by the square root of head_size.
        weights = weights * (1 / (k.shape[-1] ** (0.5)))

        # Ensures that information from the future can not communicate
        # to the past within each individual block of data
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Conversion to probabilities with softmax ensures that all 
        # weight vectors sum to 1.0, so that the magnitude of the
        # weights are standardized
        weights = F.softmax(weights, -1)

        # Some of the interactions are turned off to prevent overfitting
        # during training
        weights = self.dropout(weights)

        # (B, T, T) x (B, T, head_size) x  = (B, T, head_size)
        output = weights @ v

        return output

class MultiHead(nn.Module):
    """Implements a multi-headed attention block.

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
    """Implements a simple feedforward block of a transformer model.

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
    """Standardizes the input Tensor to 0 mean and unit variance across
    the channel dimension. Includes gain and bias parameters that can be
    optimized during training.

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
        self.ln1 = nn.LayerNorm(n_heads*head_size)
        self.ln2 = nn.LayerNorm(n_heads*head_size)

    def forward(self, x: torch.Tensor):
        # Skip connections are added to improve model training
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class LLM(nn.Module):
    """A generative large language model.


    """
    def _inititialize_params(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        self.ln = nn.LayerNorm(head_size * n_heads)
        self.output = nn.Linear(head_size * n_heads, vocab_size)
        self.register_buffer('positions', torch.arange(block_size))
        self.block_size = block_size
        self.apply(self._inititialize_params)

    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        """Calculates the logits for each token sequence and
        subsequence in the input tensor (x). If a target (y) tensor is
        provided, the cross-entropy loss is also calculated.

        The input tensor (x) must be a B x T tensor:
        - B: Batch size, i.e. number of token sequences
        - T: Time, i.e. number of tokens per sequence (block size)

        If a target tensor (y) is provided, it must also be of shape
        B x T. Each element in y contains the next token for the
        corresponding subsequence in x.

        Parameters
        ----------
        x : Tensor
            The input tensor containing the token sequences for which 
            logits are to be calculated.

        y : Tensor, default=None
            The target tensor containing the next token for each 
            subsequence in x.

        Returns
        -------
        logits : Tensor
            The logits of all possible next tokens (vocab size) for each
            subsequence in x.

        loss : Tensor
            The cross-entropy loss of the predicted next tokens for each
            subsequence in x. Only calculated if a target tensor is
            provided.

        """
        if type(x) != torch.Tensor:
            raise ValueError('x must be a tensor, received {}'.format(type(x)))
        
        if type(y) != torch.Tensor and y != None:
            raise ValueError('y must be a tensor or None, received {}'.
                             format(type(x)))
        
        # x enters as a B x T tensor, where:
        # B = batch size, i.e. number of samples (batch_size)
        # T = time, i.e. the length of each sample (block_size)
        B, T = x.shape

        # Character information is captured through the token embedding,
        # whereas positional information is captured through the
        # position embedding
        tok_vect = self.token_embeddings(x) # (B, T, n_embed)

        # (T, n_embed) 
        pos_vect = self.positional_embeddings(self.positions[:T])

        # After adding the token and position vectors together, x 
        # contains both types of information, making it more useful
        # for predicting the next token
        x = tok_vect + pos_vect # (B, T, n_embed)

        x = self.trans_blocks(x) # (B, T, head_size * n_heads)

        x = self.ln(x) # (B, T, head_size * n_heads)

        logits = self.output(x) # (B, T, vocab_size)

        _, _, C = logits.shape

        # Pytorch's cross-entropy loss function requires an N x C tensor
        logits = logits.view(B*T, C)
        loss = None
        if y != None:
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        
        return logits, loss
    
    def generate(self, 
                 context: torch.Tensor, 
                 output_length: int) -> torch.Tensor:
        """Generates a token sequence using the provided context as a
        starting point.

        Parameters
        ----------
        context : Tensor
            A tensor containing the initial context from which the new
            token sequence should be generated.

        output_length : int
            The number of new tokens to be generated.

        Returns
        -------
        output : Tensor
            A tensor containing the generated token sequence.
        """
        if len(context.shape) != 2:
            raise ValueError('Invalid context provided, expected a 2D Tensor')
        self.eval()
        output = context.clone() # (B, T)
        for i in range(output_length):

            # The positional embeddings are only defined up to
            # block_size, so only the last block_size tokens are
            # retained when calling self.forward()
            logits, _ = self.forward(output[:, -self.block_size:]) # (B*T, C)
            probs = F.softmax(logits[[-1], :], dim=-1) # (1, C)

            # The next token is sampled from the probability
            # distribution generated by the model so that it does not
            # always give the same output.
            next_idx = torch.multinomial(probs[[-1]], num_samples=1)
            output = torch.cat((output, next_idx), dim=1)
        return output

def train_llm(model: LLM,
              data_train: torch.Tensor,
              data_val: torch.Tensor,
              batch_size: int,
              block_size: int,
              lr: float,
              max_iters: int,
              eval_interval: int,
              eval_batches: int) -> None:
    """Initializes and trains a large language model with the provided
    hyperparameters.

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

    def estimate_loss(n_eval_batches: int, 
                      current_batch_no: int) -> float:
        """Estimates the training and validation loss of the model using
        a random subset of the total number of training and validation
        batches, respectively. The loss is estimated as the mean of the
        loss of all sampled batches.

        The estimated training and validation loss are both printed to
        the console, but only the validation loss is returned.

        Parameters
        ----------
        n_eval_batches : int
            The number of randomly selected batches used to estimate the
            training and validation loss.
        
        current_batch_no : int
            The number of training batches that had been processed (used
            to update the model parameters) when this method was called.
            This number is strictly used to print console output.

        Returns
        -------
        loss_val : float
            The mean validation loss across all validation batches.
        """
        model.eval()

        # Gradients are not required during evaluation, so they are 
        # turned off to improve calculation speed
        with torch.no_grad():
            losses_train = torch.zeros(n_eval_batches)
            losses_val = torch.zeros(n_eval_batches)
            for i in range(n_eval_batches):
                x_train, y_train = sample_batch(
                    data_train, batch_size, block_size)
                x_val, y_val = sample_batch(data_val, batch_size, block_size)
                _ , loss_train = model.forward(x_train, y_train)
                _ , loss_val = model.forward(x_val, y_val)
                losses_train[i] = loss_train.item()
                losses_val[i] = loss_val.item()

        loss_train = float(losses_train.mean())
        loss_val = float(losses_val.mean())
        print('\t{0} batches\tLoss (train):{1}\tLoss (val): {2}'.
              format(current_batch_no, loss_train, loss_val))
        model.train()
        return loss_val
    
    # A variable is created to Saving the best 
    best_model_state = deepcopy(model.state_dict())
    losses_val = []
    losses_val.append(estimate_loss(eval_batches, 0))
    for i in range(max_iters):
        xb, yb = sample_batch(data_train, batch_size, block_size)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model.forward(xb, yb)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            losses_val.append(estimate_loss(eval_batches, i + 1))
            
            # An increase in the validation loss suggests that the model
            # is fitting to noise and training should be terminated
            if losses_val[-1] > losses_val[-2]:
                print('\tValidation loss increased - training terminated')
                break
            # If a performance improvement was realized, the best
            # model's state dict is saved for future use
            elif losses_val[-1] < losses_val[-2]:
                best_model_state = deepcopy(model.state_dict())

    # Training is now complete, so the best model parameters are loaded
    # and the model is set to eval mode to deactivate training-specific
    # layers (eg. dropout, batch normalization)
    model.load_state_dict(best_model_state)
    model.eval()
