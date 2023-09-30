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
    
def get_batch(data, batch_size, block_size):
    idx = torch.randint(low=0, high=len(data) - 1, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y