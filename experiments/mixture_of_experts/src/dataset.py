import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from pathlib import Path
import sys

# Add project root to path for imports
def setup_path():
    markers = {".git", "requirements.txt"}
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    return None

setup_path()

class TransformedSubset(Dataset):
    """
    A Subset that applies a transform to the data.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_router_datasets(data_dir: str, train_transform=None, val_transform=None, split_ratio=0.8, seed=42):
    """
    Splits the validation folder into router_train and router_val.
    Applies separate transforms for training and validation.
    """
    val_path = Path(data_dir) / "val"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found at {val_path}")
        
    # Load dataset without transform first (to be transformed later)
    full_val_ds = datasets.ImageFolder(root=str(val_path))
    
    # Deterministic split of indices
    num_samples = len(full_val_ds)
    train_size = int(split_ratio * num_samples)
    
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = Subset(full_val_ds, train_indices)
    val_subset = Subset(full_val_ds, val_indices)
    
    # Apply transforms using wrapper
    router_train_ds = TransformedSubset(train_subset, train_transform)
    router_val_ds = TransformedSubset(val_subset, val_transform)
    
    return router_train_ds, router_val_ds, full_val_ds.classes

