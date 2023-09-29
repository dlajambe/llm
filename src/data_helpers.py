import torch
from torch.utils.data import Dataset, DataLoader

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