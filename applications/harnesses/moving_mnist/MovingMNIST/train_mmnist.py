"""Training harness on the Moving MNIST dataset."""

import sys
import os
# Get the absolute path to the Yoke directory
yoke_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
sys.path.insert(0, yoke_dir)


import argparse
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from torchvision.datasets import MovingMNIST
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from yoke.models.vit.swin.bomberman import LodeRunner
import yoke.utils.dataload as dl
import yoke.helpers.logger as yl
from yoke.helpers import cli
from yoke.utils.training.epoch import loderunner

from mmnist_dataset import mmnist_dataSet


if __name__ == "__main__":

    # Set up the Yoke logger
    yl.configure_logger("yoke_logger", level=logging.INFO)

    # Include description and parser
    descr = "Use SWIN-UNET on Moving MNIST dataset"
    parser = argparse.ArgumentParser(
        prog="Moving MNIST Training",
        description=descr,
        fromfile_prefix_chars="@",
    )

    # standard flags (gives you --studyIDX, --csv, --rundir, --cpFile)
    parser = cli.add_default_args(parser)

    # GPU/worker flags (e.g. --multigpu, --Ngpus, --num_workers)
    parser = cli.add_computing_args(parser)
    parser = cli.add_training_args(parser)

    # Model‐specific hyperparameters
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs to train"
    )
    parser.add_argument("--data_dir", type=str, default="./dataset/mmnist_data.npy", help="path to Moving_MNIST data"
    )
    parser.add_argument("--no_cuda", action="store_true", help="disable CUDA")
    parser.add_argument("--no_mps", action="store_true", help="disable macOS MPS")
    parser.add_argument("--dry_run", action="store_true", help="run only one batch")
    parser.add_argument("--log_interval", type=int, default=10, help="batches between logs"
    )
    parser.add_argument("--save_model",action="store_true",
        help="save final model to f'mnist_studyXXX_epochYYY.pt'",
    )

    args = parser.parse_args()

    # Device selection
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    torch.manual_seed(0)

    # Data loaders
    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": args.batch_size}

    if use_cuda:
        cuda_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Create the train and test sets
    train_dataset = mmnist_dataSet(args.data_dir, 0.75, "left")
    val_dataset = mmnist_dataSet(args.data_dir, 0.25, "right")

    # Define the train and val DataLoader
    train_dataloader = dl.make_dataloader(
    dataset=train_dataset,
    batch_size=train_kwargs["batch_size"],
    num_batches=250,
    num_workers=train_kwargs.get("num_workers", 1),
    prefetch_factor=2,
    )

    val_dataloader = dl.make_dataloader(
    dataset=val_dataset,
    batch_size=test_kwargs["batch_size"],
    num_batches=25,
    num_workers=test_kwargs.get("num_workers", 1),
    prefetch_factor=2,
    )

    # Set up model and optimizer
    model = LodeRunner(
        default_vars=["var1"],
        image_size=(64, 64),
        patch_size=(8, 8),
        embed_dim=4,
        emb_factor=2,
        num_heads=2,
        block_structure=(1, 1, 3, 1),
        window_sizes=[
            (4, 4),
            (4, 4),
            (2, 2),
            (1, 1),
        ],
        patch_merge_scales=[
            (2, 2),
            (2, 2),
            (2, 2),
        ],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    # Define the loss
    loss_fn = nn.MSELoss(reduction="none")

    # Put the model onto GPU
    model.to(device)

    # Optionally resume
    start_epoch = 1
    if args.continuation and args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print("Resuming from", args.checkpoint)
        # (You could parse the epoch number out of the filename here.)

    # Uncomment to debug batches in the training loop
    # batch = next(iter(train_dataloader))
    # print(f"Batch type: {type(batch)}")
    # print(f"Batch length: {len(batch) if isinstance(batch, tuple) else 'Not a tuple'}")
    # print(f"Batch contents: {batch}")

    '''
    #OPTION 1

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):

            # Unpack the batch elements from the list
            start_img, end_img, Dt = batch

            # Move data to device
            start_img = start_img.to(device)
            end_img = end_img.to(device)
            Dt = Dt.to(device)

            # Forward pass
            optimizer.zero_grad()

            # Create tensors for the input and output variable indices
            in_var_indices = torch.tensor([0], device=device)  # Just the first channel (grayscale)
            out_var_indices = torch.tensor([0], device=device)  # Same for output

            output = model(x=start_img, in_vars=in_var_indices, out_vars=out_var_indices, lead_times=Dt)
            element_loss = loss_fn(output, end_img)
            loss = element_loss.mean() # Need to take the mean here for BP

            # Backward pass
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if batch_idx % args.log_interval == 0:
                print(
                    f"Train Epoch {epoch} [{batch_idx * batch[0].size(0)}/"
                    f"{len(train_dataloader.dataset)}]\tLoss: {loss.item():.4f}"
                )
                if args.dry_run:
                    break

        # Print the epoch/loss
        #avg_loss = epoch_loss / len(train_dataloader)
        #print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        '''

    # OPTION 2
    num_epochs = 10
    for epochIDX in tqdm(range(num_epochs)):
        loderunner.train_simple_loderunner_epoch(
            [0],
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            train_per_val=10,
            train_rcrd_filename="train.csv",
            val_rcrd_filename="val.csv",
            device=device,
            verbose=False,
        )
        torch.cuda.empty_cache()


        # (You can add a test() call here if you like, mirroring the original.)

        # Removed scheduler earlier, but you could use it if you want...?
        #scheduler.step()

    # Save
    if args.save_model:
        out_name = f"mnist_study{args.studyIDX:03d}_epoch{args.epochs:03d}.pt"
        torch.save(model.state_dict(), out_name)
        print("Saved model to", out_name)
