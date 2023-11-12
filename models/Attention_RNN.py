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
#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)
        self.fc_out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1,2)
        attention_score = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float("-1e20"))
        attention = F.softmax(attention_score, dim = -1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.transpose(1,2).contiguous()
        x = x.view(batch_size, -1, self.emb_dim)
        return self.fc_out(x)
class RNNcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNcell, self).__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
    def forward(self, input, hidden):
        combined = self.input_to_hidden(input) + self.hidden_to_hidden(hidden)
        new_hidden = self.tanh(combined)
        return new_hidden
class AttentionRNN(nn.Module):
    def __init__(self, input_size, output_size, emb_dim, n_layers, hidden_size, dropout, n_heads):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.attention = MultiHeadAttention(emb_dim, n_heads, dropout)
        self.n_layers = n_layers
        self.layers = nn.ModuleList([RNNcell(input_size if i == 0 else hidden_size, hidden_size) for i in range(n_layers)])
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        device = torch.device("cuda")
        x = x.to(device)
        atten_out = self.attention(x, x, x, mask=None)
        batch_size, seq_len, _ = atten_out.size()
        h = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.n_layers)]
        outputs = []

        for t in range(seq_len):
            x_t = atten_out[:, t, :]
            new_h = []
            for i, layer in enumerate(self.layers):
                h_i = layer(x_t, h[i])
                new_h.append(h_i)
                x_t = h_i
            h = new_h
            outputs.append(x_t.unsqueeze(1))

        out = torch.cat(outputs, dim=1)
        out = self.dropout(out[:, -1, :])  # Use the last time step for classification
        return self.fc(out)

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
    splits = train_test_split(X, y, dates, returns,closes ,n_splits=4)
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
def train_and_test(batch_size, num_epochs,input_size, output_size, emb_dim, n_heads, n_layers, hidden_size, dropout, history_points):
    train_loaders, test_loaders, dates_list, returns_list, closes_list,feature_names = prepare_dataset(batch_size, history_points)
    mean_acc_score_list, mean_f1_score_list = [], []
    for idx, (train_loader, val_loader) in enumerate(zip(train_loaders, test_loaders)):
        coin_name = "bitcoin"
        dates = dates_list[idx]
        returns = returns_list[idx]
        closes = closes_list[idx]
        # CPU or GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize Model
        model = AttentionRNN(input_size, output_size, emb_dim, n_layers, hidden_size, dropout, n_heads).to(device)
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        # Training
        train_loss_list, train_acc_list, test_acc_list, train_f1_list, test_f1_list= [], [], [], [], []
        for epoch in range(num_epochs):
            torch.autograd.set_detect_anomaly(True)
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
        prediction_results.to_csv(f'../prediction_results/prediction_results_{idx}_{coin_name}_rnn.csv', index=False)
        # Plot
        if not os.path.exists("../metrics/attentionRNN"):
            os.makedirs("../metrics/attentionRNN")
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
        plt.savefig(os.path.join("../metrics/attentionRNN", plot_file))
    
    print(f"Mean acc scores: {mean_acc_score_list},"
          f"Mean f1 scores: {mean_f1_score_list}")
    return np.mean(mean_acc_score_list), np.mean(mean_f1_score_list)
batch_size, num_epochs,input_size, output_size, emb_dim, n_heads, n_layers, hidden_size, dropout, history_points = 512, 150, 22, 3, 22, 2, 2, 128, 0.45, 5
acc, f1 = train_and_test(batch_size, num_epochs,input_size, output_size, emb_dim, n_heads, n_layers, hidden_size, dropout, history_points)
print(acc, f1)
# %%
'''
history_points_acc_effect, history_points_f1_effect= [], []
for history_points in range(5, 50):
    batch_size, num_epochs,input_size, output_size, emb_dim, n_heads, n_layers, hidden_size, dropout = 512, 200, 22, 3, 22, 2, 2, 128, 0.45
    acc, f1 = train_and_test(batch_size, num_epochs,input_size, output_size, emb_dim, n_heads, n_layers, hidden_size, dropout, history_points)
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
if not os.path.exists("../metrics/attentionRNN"):
    os.makedirs("../metrics/attentionRNN")
plt.savefig(os.path.join("../metrics/attentionRNN", plot_file))
'''