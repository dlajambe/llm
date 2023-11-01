import torch
from torch.utils.data import Dataset

class NGramDataSet(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        if type(X) != torch.Tensor:
            raise ValueError('Expected X to be of type torch.Tensor')
        
        if type(y) != torch.Tensor:
            raise ValueError('Expected y to be of type torch.Tensor')
        
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of rows')
        
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> dict:
        return self.X[idx, :], self.y[idx]
    
def get_batch(
        data: torch.Tensor, 
        batch_size: int, 
        block_size: int) -> torch.long:
    """Retrieves a single batch of x and y data from the provided 
    sequence data.

    The starting points of the sequences comprising the batch are drawn
    randomly from a uniform distribution.

    Parameters
    ----------
    data : Tensor
        A 1D tensor containing the sequence data from the batch is to be
        created.

    batch_size : int
        The number of samples (rows) to be included in the batch.
    
    block_size : int
        The sequence length of each sample in the batch.
    Returns
    -------
    tuple[Tensor, Tensor]
        The x and y tensors of size (batch_size, block_size) comprising
        the batch of data.
    """
    idx = torch.randint(
        low=0, 
        high=len(data) - 1 - block_size, 
        size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

def create_matrices(
        data: torch.long,
        block_size: int
) -> torch.long:
    x = torch.stack(
        [data[i:i+block_size] for i in range(0, len(data) - block_size - 1)])
    y = torch.stack(
        [data[i:i+block_size] for i in range(1, len(data) - block_size)])
    return x, y