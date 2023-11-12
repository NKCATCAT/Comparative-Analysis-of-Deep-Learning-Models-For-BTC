#%%
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import shap
# %%
# Multi-head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim  # Embedding dimension
        self.n_heads = n_heads  # Number of attention heads
        self.head_dim = emb_dim // n_heads  # Dimension of each head

        # Linear layers for queries, keys, and values
        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)

        # Fully connected output layer
        self.fc_out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Project input tensors (query, key, value) to linear layers
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape query, key, and value tensors to (batch_size, num_heads, seq_length, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads,
                self.head_dim).transpose(1,2)
        K = K.view(batch_size, -1, self.n_heads,
                self.head_dim).transpose(1,2)
        V = V.view(batch_size, -1, self.n_heads,
                self.head_dim).transpose(1,2)

        # Calculate the attention scores by performing a scaled dot-product between query and key
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)

        # Apply the mask if provided (used in the decoder to prevent attending to future positions)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))

        # calculate the attention weights by applying softmax to the energy scores
        attention = F.softmax(attention_scores, dim=-1)

        # apply dropout to the attention weights and compute the weighted sum of the value vectors using the attention weights
        x = torch.matmul(self.dropout(attention), V)

        # Rearrange dimensions to (batch_size, seq_length, num_heads, head_dim)
        '''
        Sometimes, certain operations such as transpose, permute, or view can lead to a tensor that does not adhere to this 
        standard storage order. When the actual physical storage of a tensor's data does not match its logical layout, the tensor is considered non-contiguous. 
        '''
        x = x.transpose(1,2).contiguous()

        # Merge the attention heads back into a single tensor
        x = x.view(batch_size, -1, self.emb_dim)

        # Apply the final fully connected layer
        return self.fc_out(x)
# Position-wise Feed-Forward Network
# This part consists of two fully connected (linear layers) with a ReLU activation function in betweem.
# The purpose of this network is to apply a non_linear transformation to the input embeddings, which helps the model learn more complex patterns in the data.

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()

        # First fully connected layer, with input size emb_dim and output size ff_dim
        self.fc1 = nn.Linear(emb_dim, ff_dim)

        # Second fully connected layer, with input size ff_dim and output size emb_dim
        self.fc2 = nn.Linear(ff_dim, emb_dim)

        # Dropout layer with dropout probability dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pass the input "X" through the first fully connected layer, and apply ReLU activation function
        x = self.dropout(F.relu(self.fc1(x)))

        # Pass the output of the first layer through the second fully connected layer
        x = self.fc2(x)
        return x

# Positional Encoding
# The Positional Encoding class implements a method for adding positional information to the input embeddings.
# Positional encoding is used to inject positional information into the model by adding sinusoidal functions of different
# frequencies to the input embeddings.

# The forward function adds the positional encoding to the input and applies dropout before returning the results
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Dropout layer with dropout probability dropout
        self.dropout = nn.Dropout(p=dropout)

        # Initialize a positional encoding matrix of size (max_len, emb_dim)
        pe = torch.zeros(max_len, emb_dim)
        # Create a tensor of positions (0, 1, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the scaling factor for the sinusoidal functions
        div_term = torch.exp(torch.arange(
            0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        # Apply the sine function to the even indices of the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cosine function to the odd indices of the positional encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)
        # Transpose and unsqueeze the positional encoding matrix to match input dimensions
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register the positional encoding matrix as a buffer, so it's not considered a model parameter
        self.register_buffer('pe', pe)
    def forward(self, x):
        # Add the positional encoding to the input x
        x = x + self.pe[:x.size(0), :]
        # Apply dropout and return the result
        return self.dropout(x)
# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()

        # Multi-head Self-Attention layers
        self.self_attn = MultiHeadAttention(emb_dim, n_heads, dropout)

        # Position-Wise Feedforawrd layer
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            emb_dim, ff_dim, dropout)

        # Layer normalization 1
        self.norm1 = nn.LayerNorm(emb_dim)

        # Layer normalization 2
        self.norm2 = nn.LayerNorm(emb_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):

        # Multi-head Self-Attention
        _src = self.self_attn(src, src, src, src_mask)

        # Add & Normalize: Add the original input (residual connection) and normalize
        src = self.norm1(src + self.dropout(_src))

        # Position-wise Feedforward
        _src = self.positionwise_feedforward(src)

        # Add & Normalize: Add the output from the previous step (residual connection) and normalize
        src = self.norm2(src + self.dropout(_src))
        return src
class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, n_outputs, emb_dim, n_heads, n_layers, ff_dim, dropout):
        super(TimeSeriesTransformer, self).__init__()

        self.positional_encoding = PositionalEncoding(emb_dim, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(emb_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])

        self.fc_out = nn.Linear(emb_dim, n_outputs)

    def forward(self, src, src_mask=None):
        # Add positional encoding to the source input
        src = self.positional_encoding(src)
        # Pass the source input through each encoder layer
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Take the output from the last time step
        src = src[:, -1, :]

        # Pass the average encoder outputs through the fully connected output layer to get predictions
        return self.fc_out(src)
#%%
def group_process(group, features, history_points,location_dummies_columns):  
    scaler = StandardScaler()  
    locations = group[location_dummies_columns]
    scaled_features = scaler.fit_transform(group[features].drop(columns=location_dummies_columns, axis = 1).values)                              
    scaled_features = np.concatenate((scaled_features, locations.values), axis=1)
    X = []  
    y = []  
    dates = []
    returns = []
    closes = []
    for i in range(history_points, len(scaled_features)):  
        X.append(scaled_features[i-history_points:i])  
        y.append(group['direction'][i])   
        dates.append(group['time'][i])
        returns.append(group['returns'][i])
        closes.append(group['close'][i])
    X = np.array(X)
    y = np.array(y)
    return X, y, dates, returns, closes
def train_test_split(X_tensor, y_tensor, dates, returns,closes, n_splits=4):
    def split_process(X_tensor, y_tensor, n_splits=4):
        # Define the step size for splitting the data
        step_size = len(X_tensor) // n_splits
        # Split the data using sliding window approach
        for i in range(step_size, len(X_tensor), (step_size // 6)):
            # Split the features
            X_train = X_tensor[i-step_size:i, :]
            X_test = X_tensor[i:i+ (step_size // 6), :]  # Reverted to get a single row of test data

            # Split the targets
            y_train = y_tensor[i-step_size:i]
            y_test = y_tensor[i:i+ (step_size // 6)]  

            # return dates and returns
            test_dates = dates[i:i+ (step_size // 6)]  
            test_returns = returns[i:i+ (step_size // 6)]  
            test_closes = closes[i:i+(step_size // 6)]
            yield X_train, X_test, y_train, y_test, test_dates, test_returns,test_closes
    splits = []
    for X_train, X_test, y_train, y_test, test_dates, test_returns, test_closes in split_process(X_tensor, y_tensor, n_splits=4):
        split =  {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'test_dates': test_dates,
            'test_returns': test_returns,
            'test_closes': test_closes
        }
        if len(y_test) >= 400:
            splits.append(split)
    return splits
def prepare_dataset(batch_size, history_points):
    data = pd.read_csv(
    r"../dataset/dataset.csv")
    data['date'] = data.index
    data['direction'] = data['direction'].replace(1, 2)
    data['direction'] = data['direction'].replace(0, 1)
    data['direction'] = data['direction'].replace(-1, 0)
    data['direction'] = data['direction'].astype(int)  
    location_dummies = pd.get_dummies(data['Location'], prefix='location')
    data = pd.concat([data, location_dummies], axis=1)
    features = ['ema5', 'roc5', 'cci5', 'eom5', 'skewness', 'filtered percentiles']
    features.extend(location_dummies.columns.tolist())
    feature_names = features
    data = group_process(data, features, history_points, location_dummies.columns)
    X = data[0]
    y = data[1]
    dates = data[2]
    returns = data[3]
    closes = data[4]
    train_loaders = []
    test_loaders = []
    dates_list = []
    returns_list = []
    closes_list = []
    splits = train_test_split(X, y, dates, returns,closes, n_splits=4)
    for item in splits:
        X_train = item['X_train']
        y_train = item['y_train']
        X_val = item['X_test']
        y_val = item['y_test']
        test_dates = item['test_dates']
        test_returns = item['test_returns']
        test_closes = item['test_closes']
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)  
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        train_loaders.append(train_loader)
        test_loaders.append(val_loader)
        dates_list.append(test_dates)
        returns_list.append(test_returns)
        closes_list.append(test_closes)
    return train_loaders, test_loaders, dates_list, returns_list, closes_list, feature_names
#%%
def train_and_test(batch_size, num_epochs,n_features, n_outputs, emb_dim, n_heads, n_layers, ff_dim, dropout, history_points):
    train_loaders, test_loaders, dates_list, returns_list, closes_list, feature_names = prepare_dataset(batch_size, history_points)
    mean_acc_score_list, mean_f1_score_list = [], []
    for idx, (train_loader, val_loader) in enumerate(zip(train_loaders, test_loaders)):
        coin_name = "bitcoin"
        dates = dates_list[idx]
        returns = returns_list[idx]
        closes = closes_list[idx]
        # CPU or GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize Model
        model = TimeSeriesTransformer(n_features, n_outputs, emb_dim, n_heads, n_layers, ff_dim, dropout).to(device)
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        # Training
        train_loss_list, train_acc_list, test_acc_list, train_f1_list, test_f1_list= [], [], [], [], []
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (data, target_train) in enumerate(train_loader):
                data, target_train = data.to(device), target_train.to(device)
                output = model(data)
                loss = criterion(output, target_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted = output.argmax(dim=1)
                train_correct = (predicted == target_train).sum().item()
                train_f1_score = f1_score(target_train.cpu(), output.argmax(dim=1).cpu(), average='macro') 

            train_f1_list.append(train_f1_score)
            train_loss_list.append(loss.item())
            train_acc_list.append(train_correct / target_train.size(0))

            # 验证模型性能
            model.eval()
            with torch.no_grad():
                predictions = []
                for batch_idx, (data, target_test) in enumerate(val_loader):
                    data, target_test = data.to(device), target_test.to(device)
                    output = model(data)
                    predicted = output.argmax(dim=1)
                    predicteds = predicted.tolist()
                    for i in predicteds:
                        predictions.append(i)
                    test_correct = (predicted == target_test).sum().item()
                    test_f1_score = f1_score(target_test.cpu(),output.argmax(dim=1).cpu(), average='macro')
                test_f1_list.append(test_f1_score)
                test_acc_list.append(test_correct / target_test.size(0))
            print(f"Epoch: {epoch + 1}")
            print(f"Train acc: {train_correct / target_train.size(0)}")
            print(f"Train f1 macro: {train_f1_score}")
            print(f"Test acc: {test_correct / target_test.size(0)}")
            print(f"Test f1 macro: {test_f1_score}")
            '''
            # Save Model
            save_epochs = [49, 99, 199, 299, 399, 499, 599, 799, 999, 1199]

            # Check if the directory exists, if not, create it
            if not os.path.exists("../state_dict"):
                os.makedirs("../state_dict")

            # Save the model at specific epochs
            if epoch in save_epochs:
                torch.save(model.state_dict(), f'../state_dict/model_{coin_name}_epoch{epoch + 1}.pth')
            '''
        mean_acc_score, mean_f1_score= np.mean(test_acc_list), np.mean(test_f1_score)
        mean_acc_score_list.append(mean_acc_score)
        mean_f1_score_list.append(mean_f1_score)
        '''
        # SHAP
        if not os.path.exists("../shap/transformerencoder"):
            os.makedirs("../shap/transformerencoder")
        val_samples = torch.cat([data.to(device) for data, _ in val_loader], dim=0)
        explainer = shap.DeepExplainer(model, val_samples)
        shap_values = explainer.shap_values(val_samples)
        shap_plot_file = f'shap_values_model_{coin_name}.png'
        shap.image_plot(shap_values, val_samples, feature_names=feature_names,
                        save_path=os.path.join("../shap/transformerencoder", shap_plot_file))
        '''
        # Prediction_results
        if not os.path.exists("../prediction_results"):
            os.makedirs("../prediction_results")
        prediction_results = pd.DataFrame({  
                        'Coin': [coin_name] * len(dates),  
                        'Date': dates,
                        'Prediction': predictions,
                        'Returns': returns,
                        'Prices': closes
                    })  
        prediction_results.to_csv(f'../prediction_results/prediction_results_{idx}_{coin_name}_transformer.csv', index=False)
        # Plot
        if not os.path.exists("../metrics/transformerencoder"):
            os.makedirs("../metrics/transformerencoder")
        plt.figure(figsize=(16, 16), dpi = 400)
        plt.suptitle(f"{coin_name}")
        plt.subplot(3, 1, 1)
        plt.plot(train_loss_list, label="Train Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(3, 1, 2)
        plt.plot(train_acc_list, label="Train Acc")
        plt.plot(train_f1_list, label="Train F1")
        plt.legend()
        plt.title("Accuracy")

        plt.subplot(3, 1, 3)
        plt.plot(test_acc_list, label="Test Acc")
        plt.plot(test_f1_list, label="Test F1")
        plt.legend()
        plt.title("F1 macro")
        plot_file = f'{coin_name}_{idx}_metrics_plot.png'
        plt.savefig(os.path.join("../metrics/transformerencoder", plot_file))
    print(f"Mean acc scores: {mean_acc_score_list},"
          f"Mean f1 scores: {mean_f1_score_list}")
    return np.mean(mean_acc_score_list), np.mean(mean_f1_score_list)
#%%
batch_size, num_epochs,n_features, n_outputs, emb_dim, n_heads, n_layers, ff_dim, dropout, history_points = 512, 150, 22, 3, 22, 2, 2, 64, 0.3, 5
acc, f1 = train_and_test(batch_size, num_epochs,n_features, n_outputs, emb_dim, n_heads, n_layers, ff_dim, dropout, history_points)
print(acc, f1)
#%%
'''
history_points_acc_effect, history_points_f1_effect= [], []
for history_points in range(5, 100):
    batch_size, num_epochs,n_features, n_outputs, emb_dim, n_heads, n_layers, ff_dim, dropout = 512, 150, 22, 3, 22, 2, 2, 64, 0.3
    acc, f1 = train_and_test(batch_size, num_epochs,n_features, n_outputs, emb_dim, n_heads, n_layers, ff_dim, dropout, history_points)
    history_points_acc_effect.append(acc)
    history_points_f1_effect.append(f1)
plt.figure(figsize=(16, 11), dpi = 400)
plt.subplot(2, 1, 1)
plt.plot(history_points_acc_effect, label="window-accuracy")
plt.legend()
plt.title("Accuracy under Different Windows")
plt.xlabel("Windows (4-hours / window)")
plt.subplot(2, 1, 2)
plt.plot(history_points_f1_effect, label="window-f1")
plt.legend()
plt.title("F1 under Different Windows")
plot_file = 'window_metrics_plot.png'
plt.savefig(os.path.join("../metrics/transformerencoder", plot_file))
'''