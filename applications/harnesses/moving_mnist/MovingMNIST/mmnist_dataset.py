import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view
import os

class mmnist_dataSet(Dataset):
    """Moving MNIST dataset."""

    def __init__(self, data_path='./dataset/mnist_test_seq.npy', fraction=1, fraction_side="left") -> None:
        # Load the dataset directly from the .npy file
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file {data_path} not found.")

        # Load data and transpose to match the expected format
        # Original shape: (20, 10000, 64, 64) -> (10000, 20, 64, 64)
        self.data = np.load(data_path).transpose(1, 0, 2, 3)

        total_len = self.data.shape[0]  # Number of sequences (10000)
        seq_len = self.data.shape[1]    # Number of frames per sequence (20)
        pairs_per_seq = seq_len - 1

        frac_range = range(0, int(fraction * total_len))
        if fraction_side == "right":
            frac_range = range(int(fraction * total_len), total_len)

        self.seq_id = [
            x for xs in [np.repeat(i, pairs_per_seq) for i in frac_range] for x in xs
        ]

        pairs_local = [
            sliding_window_view(np.arange(0, seq_len), window_shape=2)
            for _ in frac_range
        ]
        self.pairs_local = np.concatenate(pairs_local)

    def __len__(self) -> int:
        return len(self.seq_id)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        seq_idx = self.seq_id[index]
        frame_indices = self.pairs_local[index]

        # Get the pair of frames
        start_img = self.data[seq_idx, frame_indices[0]]
        end_img = self.data[seq_idx, frame_indices[1]]

        # Convert to torch tensors and normalize
        start_img = torch.tensor(np.expand_dims(start_img, 0)).to(torch.float32) / 255
        end_img = torch.tensor(np.expand_dims(end_img, 0)).to(torch.float32) / 255

        Dt = torch.tensor(0.25, dtype=torch.float32)  # arbitrary value
        return start_img, end_img, Dt
