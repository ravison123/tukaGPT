import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        # Each sample is a sequence of length context_length
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        x = torch.tensor(
            self.tokens[idx : idx + self.context_length],
            dtype=torch.long
        )
        y = torch.tensor(
            self.tokens[idx + 1 : idx + self.context_length + 1],
            dtype=torch.long
        )
        return x, y


def get_dataloader(tokens, context_length, batch_size, shuffle=True):
    dataset = TextDataset(tokens, context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


