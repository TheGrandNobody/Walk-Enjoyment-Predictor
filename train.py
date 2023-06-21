from lstm import LSTMRegressor 
import optuna
import pandas as pd
import torch
import shap
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss

def evaluate(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # No need to track gradients
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            output = model(inputs)
            # Compute loss
            loss = criterion(output, targets)
            # Accumulate the test loss
            test_loss += loss.item()
            
    # Compute the average test loss
    avg_test_loss = test_loss / len(test_loader)

    return avg_test_loss

# Define a function to calculate validation loss
def validate(model, valid_loader, criterion):
    model.eval()
    with torch.no_grad():
        valid_loss = sum(criterion(model(inputs.to(device)), targets.to(device)).item() for inputs, targets in valid_loader)
    return valid_loss / len(valid_loader)

def objective(trial):
    df = pd.read_csv('final_5s.csv')
    inp = torch.Tensor(df.drop('Average PACES', axis = 1).values)
    inp = inp.view(1927, 1, 56)
    lab = torch.Tensor(df['Average PACES'].values)
    lab = lab.view(1927, 1)
    hidden_dim = trial.suggest_int("hidden_dim", 75, 125)
    num_layers = trial.suggest_int("num_layers", 10, 50)
    lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e-5)

    model = LSTMRegressor(input_dim=56, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=7, verbose=True)
    # First split the data into a training set and a temporary set using 80/20 split
    inpu,inputs_temp,labe,targets_temp = train_test_split(inp, lab, test_size=0.2, random_state=42)
    # Then split the temporary set into validation and test sets using 50/50 split
    vali,_,labels_vali,_ = train_test_split(inputs_temp, targets_temp, test_size=0.5, random_state=42)
    # Create Tensor datasets
    train_data = TensorDataset(inpu, labe)
    valid_data = TensorDataset(vali, labels_vali)
    # Create Data loaders
    batch_size = 32  # Choose as appropriate
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    while True:
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)
                val_loss += loss.item()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    trial.set_user_attr('model_state_dict', model.state_dict())
    return early_stopping.val_loss_min


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')  # Create a new study.
    study.optimize(objective, n_trials=1)  # Invoke optimization of the objective function.
    best_trial = study.best_trial

    # Now you can save the model parameters from the best trial
    model = LSTMRegressor(input_dim=56, hidden_dim=study.best_trial.params['hidden_dim'], output_dim=1, num_layers=study.best_trial.params['num_layers'])
    model.load_state_dict(study.best_trial.user_attrs['model_state_dict'])

    # Save the best model parameters
    torch.save(model.state_dict(), 'best.pth')

    print(f"Best trial: score {best_trial.value}, params {best_trial.params}")
    model.to(device)
    df = pd.read_csv('final_5s.csv')
    inputs = torch.Tensor(df.drop('Average PACES', axis = 1).values)
    inputs = inputs.view(1927, 1, 56)
    labels = torch.Tensor(df['Average PACES'].values)
    labels = labels.view(1927, 1)
    # First split the data into a training set and a temporary set using 80/20 split
    inputs,inputs_temp,labels,targets_temp = train_test_split(inputs, labels, test_size=0.2, random_state=42)
    # Then split the temporary set into validation and test sets using 50/50 split
    _,test,_,test_labels = train_test_split(inputs_temp, targets_temp, test_size=0.5, random_state=42)
    test_d = TensorDataset(test, test_labels)
    test_loader = DataLoader(test_d, shuffle=True, batch_size=32, drop_last=True)

    # we assume you have a trained PyTorch LSTM model called model
    model.eval()  # set the model to evaluation mode
    # Usage:
    test_loss = evaluate(model, test_loader, torch.nn.MSELoss())
    print("Test Loss: {:.6f}".format(test_loss))

    # we create a DeepExplainer object that can calculate shap values
    explainer = shap.DeepExplainer(model, inputs[:100])  # use a subset of training data as background
    test_inputs, test_targets = next(iter(test_loader))
    test_inputs = test_inputs.view(test_inputs.size(0), 1, -1)
    test_inputs = test_inputs.to(device)
    # we calculate shap values for the test examples
    shap_values = explainer.shap_values(test_inputs)

    # Combine the output-specific SHAP arrays into one array with an added output dimension
    shap_values_combined = np.stack(shap_values, axis=0)

    # Reshape test_inputs for compatibility with SHAP plotting function
    test_inputs_reshaped = test_inputs.view(test_inputs.shape[0], -1)

    print("Shape of shap_values_combined: ", shap_values_combined.shape)
    print("Shape of test_inputs_reshaped: ", test_inputs_reshaped.shape)

    # Plot the SHAP values
    shap.summary_plot(shap_values_combined, test_inputs_reshaped.cpu().numpy())



    # Print the average absolute SHAP values for each feature
    print(torch.mean(torch.abs(torch.tensor(shap_values)), axis=0))
