import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ray import train
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, f1_score


def evaluate_model(X, y, model, n_splits=5):
    """
    Evaluates a classifier using cross-validation and returns ROC AUC, PR AUC, and F1-score.
    
    Parameters:
        inputs_df (pd.DataFrame): The input features.
        targets_df (pd.DataFrame): The target labels.
        model_type (str): The model type to use. Either "RandomForest" or "LogisticRegression".
        n_splits (int): The number of splits for cross-validation (default is 5).
    
    Returns:
        dict: A dictionary containing average ROC AUC, PR AUC, and F1 scores.
    """

    # Define scoring metrics
    roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovo')
    pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
    f1_scorer = make_scorer(f1_score)

    # Create KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Calculate cross-validated ROC AUC scores
    roc_auc_scores = cross_val_score(model, X, y, cv=kf, scoring=roc_auc_scorer)

    # Calculate cross-validated PR AUC scores
    pr_auc_scores = cross_val_score(model, X, y, cv=kf, scoring=pr_auc_scorer)

    # Calculate cross-validated F1 scores
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring=f1_scorer)

    # Print and return the scores
    print(f'ROC AUC scores for each fold: {roc_auc_scores}')
    print(f'Average ROC AUC: {roc_auc_scores.mean()}')

    print(f'PR AUC scores for each fold: {pr_auc_scores}')
    print(f'Average PR AUC: {pr_auc_scores.mean()}')

    print(f'F1 scores for each fold: {f1_scores}')
    print(f'Average F1 score: {f1_scores.mean()}')


def evaluate_model_ray(X, y, model, n_splits=3):
    """
    Evaluates a classifier using cross-validation and returns ROC AUC, PR AUC, and F1-score.
    
    Parameters:
        inputs_df (pd.DataFrame): The input features.
        targets_df (pd.DataFrame): The target labels.
        model_type (str): The model type to use. Either "RandomForest" or "LogisticRegression".
        n_splits (int): The number of splits for cross-validation (default is 5).
    
    Returns:
        dict: A dictionary containing average ROC AUC, PR AUC, and F1 scores.
    """

    # Define scoring metrics
    roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovo')
    pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
    f1_scorer = make_scorer(f1_score)

    # Create KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Calculate cross-validated ROC AUC scores
    roc_auc_scores = cross_val_score(model, X, y, cv=kf, scoring=roc_auc_scorer)
    pr_auc_scores = cross_val_score(model, X, y, cv=kf, scoring=pr_auc_scorer)
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring=f1_scorer)

    # Print and return the scores
    print(f'Average ROC AUC: {roc_auc_scores.mean()}')
    print(f'Average PR AUC: {pr_auc_scores.mean()}')
    print(f'Average F1 score: {f1_scores.mean()}')
    train.report({
        "roc_auc": roc_auc_scores.mean(),
        "pr_auc": pr_auc_scores.mean(),
        "f1_score": f1_scores.mean()
    })


def evaluate_network(X, y, n_fold=3, lr=0.001, layer_sizes=[128, 64]):
    # Define the neural network
    class SimpleNN(nn.Module):

        def __init__(self, input_size, output_size, layer1, layer2):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, layer1)  # First hidden layer
            self.fc2 = nn.Linear(layer1, layer2)  # Second hidden layer
            self.fc3 = nn.Linear(layer2, output_size)  # Output layer
            self.act = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            x = torch.sigmoid(x)
            return x

    # Input and output dimensions based on the shapes you provided
    input_size = X.shape[1]  # Number of features
    output_size = 1  # Number of target variables
    num_epochs = 20  # Adjust as needed
    batch_size = 16

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))

    kf = KFold(n_splits=n_fold, shuffle=True)

    f1_scores = list()
    auc_prs = list()
    auc_rocs = list()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # print(f"Fold {fold + 1}")
        # Training loop

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

        model = SimpleNN(input_size, output_size, *layer_sizes)
        criterion = nn.BCELoss()  # Mean Squared Error for regression tasks
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer with learning rate 0.001

        # print("Training the fold")
        for epoch in range(num_epochs):
            print()
            model.train()  # Set the model to training mode
            running_loss = 0.0

            for batch_inputs, batch_targets in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_inputs)
                loss = criterion(outputs.view(-1), batch_targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print average loss for the epoch

            model.eval()
            val_preds = list()
            val_labels = list()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
            roc_auc_score(val_labels, val_preds)

            # Compute the AUC-ROC score
            auc_roc = roc_auc_score(val_labels,
                                    val_preds)  # Use the probabilities for the positive class

            # Compute the AUC-PR (precision-recall curve area)
            auc_pr = average_precision_score(val_labels, val_preds)

            # Compute the F1-score

            f1 = f1_score(val_labels, np.round(np.concatenate(val_preds)))
            print(
                f"Epoch {epoch+1}, Loss: {running_loss:.2f}, LR, ROC AUC: {auc_roc:.2f}, PR AUC: {auc_pr:.2f}, F1: {f1:.2f}"
            )

        auc_prs.append(auc_pr)
        auc_rocs.append(auc_roc)
        f1_scores.append(f1)

    # Print and return the scores
    print(f'Average ROC AUC: {np.array(auc_rocs).mean()}')
    print(f'Average PR AUC: {np.array(auc_prs).mean()}')
    print(f'Average F1 score: {np.array(f1_scores).mean()}')

    return np.array(auc_rocs).mean(), np.array(auc_prs).mean(), np.array(f1_scores).mean()


def evaluate_network_ray(X,
                         y,
                         lr=0.001,
                         batch_size=16,
                         layer_sizes=[128, 64],
                         dropout_rate=0.2,
                         weight_decay=1e-5,
                         test_size=0.3,
                         lr_decay_step=5,
                         lr_decay_gamma=0.1):
    # Define the neural network
    class SimpleNN(nn.Module):

        def __init__(self, input_size, output_size, layer_sizes, dropout_rate):
            super(SimpleNN, self).__init__()
            self.layers = nn.ModuleList()
            # Input layer
            self.layers.append(nn.Linear(input_size, layer_sizes[0]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

            # Hidden layers
            for i in range(1, len(layer_sizes)):
                self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_rate))

            # Output layer
            self.layers.append(nn.Linear(layer_sizes[-1], output_size))
            self.layers.append(nn.Sigmoid())

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    # Input and output dimensions based on the shapes you provided
    input_size = X.shape[1]  # Number of features
    output_size = 1  # Number of target variables
    num_epochs = 20  # Adjust as needed

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))

    f1_scores = list()
    auc_prs = list()
    auc_rocs = list()

    # Generate dataset indices
    indices = np.random.permutation(len(y))
    split_point = int(np.floor(test_size * len(y)))
    train_idx = indices[split_point:]
    val_idx = indices[:split_point]

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = SimpleNN(input_size=input_size,
                     output_size=output_size,
                     layer_sizes=layer_sizes,
                     dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    # print("Training the fold")
    for epoch in range(num_epochs):
        print()
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_inputs, batch_targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs.view(-1), batch_targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        # Print average loss for the epoch

        model.eval()
        val_preds = list()
        val_labels = list()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

            roc_auc_score(val_labels, val_preds)
            auc_roc = roc_auc_score(val_labels, val_preds)
            auc_pr = average_precision_score(val_labels, val_preds)
            f1 = f1_score(val_labels, np.round(np.concatenate(val_preds)))

            train.report({
                "epoch": epoch,
                "loss": running_loss,
                "roc_auc": auc_roc,
                "pr_auc": auc_pr,
                "f1_score": f1,
                "learning_rate":
                    scheduler.get_last_lr()[0]  # Report current learning rate
            })
        print(
            f"Epoch {epoch+1}, Loss: {running_loss:.2f}, LR: {scheduler.get_last_lr()[0]:.2f}, ROC AUC: {auc_roc:.2f}, PR AUC: {auc_pr:.2f}, F1: {f1:.2f}"
        )

    auc_prs.append(auc_pr)
    auc_rocs.append(auc_roc)
    f1_scores.append(f1)

    return auc_roc, auc_pr, f1
