import shap
import torch
from lstm import LSTMRegressor
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor(input_dim=56, hidden_dim=75, output_dim=1, num_layers=10)
model.load_state_dict(torch.load('best.pth'))
model.to(device)
df = pd.read_csv('final_5s.csv')
inputs = torch.Tensor(df.drop('Average PACES', axis = 1).values)
inputs = inputs.view(1927, 1, 56)
# we also assume inputs is a tensor of your input data
inputs = inputs.to(device)

# we assume you have a trained PyTorch LSTM model called model
model.eval()  # set the model to evaluation mode

# we create a DeepExplainer object that can calculate shap values
explainer = shap.DeepExplainer(model, inputs)

# we calculate shap values for the first 10 examples in the training set
shap_values = explainer.shap_values(inputs[:10])

# Average attributions over the sequence length dimension
shap_values = shap_values.mean(axis=1)

# Plot the feature attributions. Convert inputs to 2D for the plot
shap.summary_plot(shap_values, inputs[:10].view(-1, inputs.size(-1)))

# Print the average absolute SHAP values for each feature
print(shap_values)
