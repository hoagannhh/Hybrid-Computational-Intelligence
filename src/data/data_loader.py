import torch
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    """PyTorch Dataset cho dữ liệu chuỗi thời gian chứng khoán."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Tạo DataLoader cho 3 tập Train, Val, Test."""
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
