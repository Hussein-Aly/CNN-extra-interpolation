# -*- coding: utf-8 -*-
"""example_project/main.py

Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 15.04.2022

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Main file of example project.
"""

import os

import dill
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import plot
from cnn import SimpleCNN
from data_sets import ImageDataset, collate_fn, TestDataset, collate_fn_test


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval()
    # We will accumulate the mean loss in variable `loss`
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, knowns, targets, labels = data
            inputs = inputs.to(device)
            knowns = knowns.to(device)
            targets = targets.to(device)

            # Get outputs of the specified model
            outputs = model(inputs)

            # outputs shape and knowns shape-> 4, 3, 100, 100 (predicted input array after cnn of one mini-batch-sample)
            # loop over all samples in mini-batch to compute loss
            for i, pred_kn in enumerate(zip(outputs, knowns)):
                pred, kn = pred_kn
                pred_target = torch.clone(pred)[kn == 0]
                # Add the current loss, which is the mean loss over all minibatch samples
                # (unless explicitly otherwise specified when creating the loss function!)
                pred_target = pred_target.to(device)
                loss += loss_fn(pred_target, targets[0]).item()

    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    loss /= len(dataloader)
    model.train()
    return loss


def main(results_path="results", network_config={
    "n_hidden_layers": 3,
    "n_in_channels": 3,
    "n_kernels": 32,
    "kernel_size": 7
}, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = 50_000, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare a path to plot to
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)

    our_dataset = ImageDataset(0, 300)
    # only take 100 files for debugging
    n_samples = len(our_dataset)

    # Shuffle integers from 0 to n_samples to get shuffled sample indices
    shuffled_indices = np.random.permutation(len(our_dataset))

    trainingset_inds = shuffled_indices[:int(n_samples / 10) * 7]  # 1st 70 % of the whole set
    validationset_inds = shuffled_indices[int(n_samples / 10) * 7:int(n_samples / 10) * 9]  # 2nd 20 % of the whole set
    testset_inds = shuffled_indices[int(n_samples / 10) * 9:int(n_samples)]  # 3rd 10 % of the whole set

    # Split dataset into training, validation and test set
    # Create PyTorch subsets from our subset-indices
    testset = torch.utils.data.Subset(our_dataset, indices=testset_inds)
    validationset = torch.utils.data.Subset(our_dataset, indices=validationset_inds)
    trainingset = torch.utils.data.Subset(our_dataset, indices=trainingset_inds)

    # Create dataloaders from each subset
    testloader = torch.utils.data.DataLoader(testset,  # we want to load our dataset
                                             shuffle=False,  # shuffle for training
                                             batch_size=1,  # 1 sample at a time
                                             num_workers=0,  # no background workers
                                             collate_fn=collate_fn
                                             )
    valloader = torch.utils.data.DataLoader(validationset,  # we want to load our dataset
                                            shuffle=False,  # shuffle for training
                                            batch_size=1,  # stack 4 samples to a minibatch
                                            num_workers=0,
                                            collate_fn=collate_fn
                                            )
    trainloader = torch.utils.data.DataLoader(trainingset,  # we want to load our dataset
                                              shuffle=True,  # shuffle for training
                                              batch_size=1,  # stack 4 samples to a minibatch
                                              num_workers=0,
                                              collate_fn=collate_fn
                                              )

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    # Create Network
    net = SimpleCNN(**network_config)
    net.to(device)

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)

    print_stats_at = 100  # print status to tensorboard every x updates
    plot_at = 10_000  # plot every x updates
    validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    # Train until n_updates updates have been reached
    while update < n_updates:
        for data in trainloader:
            # Get next samples
            inputs, knowns, targets, labels = data
            knowns = knowns.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs of our network
            outputs = net(inputs)

            # outputs shape and knowns shape-> 4, 3, 100, 100 (predicted input array after cnn of one mini-batch-sample)
            # loop over all samples in mini-batch to compute loss

            loss = mse(torch.zeros(1), torch.zeros(1))
            loss = loss.to(device)
            for i, pred_kn in enumerate(zip(outputs, knowns)):
                pred, kn = pred_kn
                pred_target = torch.clone(pred)[kn == 0]
                # Add the current loss, which is the mean loss over all minibatch samples
                # (unless explicitly otherwise specified when creating the loss function!)
                pred_target = pred_target.to(device)
                loss += mse(pred_target, targets[0])

            # Calculate loss, do backward pass and update weights
            loss.backward()
            optimizer.step()

            # Print current status and score
            if (update + 1) % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss, global_step=update)

            # Plot output
            # if (update + 1) % plot_at == 0:
            #     plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
            #          plotpath, update)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param, global_step=update)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad,
                                         global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            # Increment update counter, exit if maximum number of updates is reached
            # Here, we could apply some early stopping heuristic and also exit if its
            # stopping criterion is met
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, dataloader=trainloader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
    test_loss = evaluate_model(net, dataloader=testloader, loss_fn=mse, device=device)

    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")

    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


def make_predictions(device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    # load trained model
    cnn = torch.load(os.path.join('results', 'best_model.pt'))

    with open(r"D:\JKU\S2022\Python 2\CNN-predicting-incomplete-images\test\inputs.pkl", 'rb') as test:
        testset = dill.load(test)

    # Create a torch testset from the pickle file
    processed_testset = TestDataset(testset)

    testloader = DataLoader(processed_testset,  # we want to load our dataset
                            shuffle=False,  # shuffle for training
                            batch_size=1,  # 1 sample at a time
                            num_workers=0,  # no background workers
                            # pin_memory=True,
                            collate_fn=collate_fn_test
                            )

    preds_list = []
    for data in testloader:
        inputs, knowns, labels = data
        inputs = inputs.to(device)
        knowns = knowns.to(device)
        labels = labels.to(device)

        # Get outputs for network
        outputs = cnn(inputs)

        for i, pred_kn in enumerate(zip(outputs, knowns)):
            pred, kn = pred_kn
            pred_target = torch.clone(pred)[kn == 0]
            x = pred_target.type(torch.uint8)
            preds_list.append(x.cpu().detach().numpy())

    with open('test/outputs.pkl', 'wb') as p:
        dill.dump(preds_list, p)

    print('DONE!!')


if __name__ == "__main__":
    # import argparse
    # import json
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="Path to JSON config file")
    # args = parser.parse_args()
    #
    # with open(args.config_file) as cf:
    #     config = json.load(cf)
    # main(**config)
    main()
    make_predictions()
