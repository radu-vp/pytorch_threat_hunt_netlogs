#!/usr/bin/env python3
"""Threat Detection Using Deep Learning

Check README.md for more detailed information.
"""

import sys
import textwrap

import torch

import pandas as pd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy


# Variables used for some hyperparameters
LEARNING_RATE = 0.01
EPOCHS_NUM = 10         # Nr. of Epochs - using 10 due to small dataset size
# Optimizer
OPTIMIZER = "SGD"       # Choices are either "Adam" or "SGD")
# Data files locations
fp_train = "datasets/labelled_train.csv"
fp_test = "datasets/labelled_test.csv"
fp_validation = "datasets/labelled_validation.csv"


def main():

    # Step 1: Loading the preprocessed data
    df_train = pd.read_csv(fp_train)
    df_test = pd.read_csv(fp_test)
    df_val = pd.read_csv(fp_validation)

    # Step 2: Separating features and labels for each set (training, testing, validation)
    train_x_features = df_train.drop("sus_label", axis = 1).values
    train_y_labels = df_train["sus_label"].values
    test_x_features = df_test.drop("sus_label", axis = 1).values
    test_y_labels = df_test["sus_label"].values
    val_x_features = df_val.drop("sus_label", axis = 1).values
    val_y_labels = df_val["sus_label"].values

    # Step 3: Scaling features
    # scaler variable holds the initialized StandardScaler
    scaler = StandardScaler()
    # Fit the scaler on the training data and transform the training data
    train_x_features = scaler.fit_transform(train_x_features)
    # Transform the test and validation data using the fitted scaler
    test_x_features = scaler.transform(test_x_features)
    val_x_features = scaler.transform(val_x_features)

    # Step 5: Converting the numpy arrays to PyTorch Tensors
    train_x_features_tensor = torch.tensor(train_x_features, dtype = torch.float32)
    train_y_labels_tensor = torch.tensor(train_y_labels, dtype = torch.float32).view(-1, 1)
    test_x_features_tensor = torch.tensor(test_x_features, dtype = torch.float32)
    test_y_labels_tensor = torch.tensor(test_y_labels, dtype = torch.float32).view(-1, 1)
    val_x_features_tensor = torch.tensor(val_x_features, dtype = torch.float32)
    val_y_labels_tensor = torch.tensor(val_y_labels, dtype = torch.float32).view(-1, 1)

    # Using nn.Sequential to define the model
    model = nn.Sequential(nn.Linear(train_x_features.shape[1], 128),    # First fully connected layer
                          nn.ReLU(),                                    # ReLU activation
                          nn.Linear(128, 64),                           # Second fully connected layer
                          nn.ReLU(),                                    # ReLU activation
                          nn.Linear(64, 1),                             # Third fully connected layer
                          nn.Sigmoid())                                 # Sigmoid activation for binary classification

    # Initializing the loss function and optimizer depending on desired OPTIMIZER value
    match OPTIMIZER:
        case "SGD":
            learning_rate = 1e-3
            weight_decay = 1e-4
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(),
                                  lr = learning_rate,
                                  weight_decay = weight_decay)
        case "Adam":
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(),
                                   lr = LEARNING_RATE)

    # Performing the training for the desired nr. of Epochs
    for epoch in range(EPOCHS_NUM):
        model.train()                                       # Set the model to training mode
        optimizer.zero_grad()                               # Clear the gradients
        outputs = model(train_x_features_tensor)            # Forward pass: compute the model output
        loss = criterion(outputs, train_y_labels_tensor)    # Compute the loss
        loss.backward()                                     # Backward pass: compute the gradients
        optimizer.step()                                    # Update the model parameters

    # Evaluating the model
    model.eval()            # Setting the model to evaluation mode
    with torch.no_grad():   # Disabling gradient calculation for efficiency using no_grad() method
        # Predicting on training, test and validation data and rounding the outputs
        y_predict_train = model(train_x_features_tensor).round()
        y_predict_test = model(test_x_features_tensor).round()
        y_predict_val = model(val_x_features_tensor).round()

    # Using torchmetrics to calculate the accuracy for training, testing and validation
    accuracy = Accuracy(task = "binary")
    accuracy_train = accuracy(y_predict_train, train_y_labels_tensor)
    accuracy_test = accuracy(y_predict_test, test_y_labels_tensor)
    accuracy_val = accuracy(y_predict_val, val_y_labels_tensor)
    accuracy_train = accuracy_train.item()
    accuracy_test = accuracy_test.item()
    accuracy_val = accuracy_val.item()

    # Output accuracy results
    statistics = textwrap.dedent(f"""\
                    [#] Accuracy training: {accuracy_train}
                    [#] Accuracy validation: {accuracy_val}
                    [#] Accuracy testing: {accuracy_test}""")
    print(statistics)


if __name__ == "__main__":
    # Catch CTRL+C and exit without error message
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(1)
