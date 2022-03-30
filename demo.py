from cgi import print_directory
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

LR = 0.0001
EPOCH = 10
TRAIN_END = 3000
DAYS_BEFORE = 30
TOTAL_SIZE = 3457

def load_train_data():
    df=pd.read_csv('./lstm_data.csv')
    series = df['price'].copy()
    train_end = TRAIN_END
    days_before = DAYS_BEFORE
    train_series, test_series = series[:train_end], series[train_end - days_before:]
    train_data = pd.DataFrame()
    for i in range(days_before):
        train_data['c%d' % i] = train_series.tolist()[i: -days_before + i]
            
    train_data['y'] = train_series.tolist()[days_before:]
    # print(train_data)
    # here form a matrix
    # using the 30 days before to pridict the next day
    return train_data, series, df.index.tolist()

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1, 
            batch_first=True)
        
        self.out = nn.Sequential(
            nn.Linear(64,1))
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])      
        
        return out

class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)




def train_model():
    train_data, all_series, df_index = load_train_data()
    all_series = np.array(all_series.tolist())
    plt.figure(figsize=(12,8))
    plt.plot(df_index, all_series, label='real-data')
    plt.savefig('out.png')

    train_data_numpy = np.array(train_data)
    train_mean = np.mean(train_data_numpy)
    train_std  = np.std(train_data_numpy)
    train_data_numpy = (train_data_numpy - train_mean) / train_std
    train_data_tensor = torch.Tensor(train_data_numpy)

    train_set = TrainSet(train_data_tensor)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

    rnn = LSTM()

    if torch.cuda.is_available():
        rnn = rnn.cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    for step in range(EPOCH):
        for tx, ty in train_loader:
            
            if torch.cuda.is_available():
                tx = tx.cuda()
                ty = ty.cuda()       
            
            output = rnn(torch.unsqueeze(tx, dim=2))
            loss = loss_func(torch.squeeze(output), ty)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
        print(step, loss.cpu())
        if step % 10:
            torch.save(rnn, 'rnn.pkl')
    torch.save(rnn, 'rnn.pkl')

def eval_model():
    rnn = LSTM()
    rnn = torch.load('./rnn.pkl')
    rnn = rnn.eval()
    train_data, all_series, df_index = load_train_data()
    generate_data_train = []
    generate_data_test_real = []
    generate_data_test_predict = []

    # 测试数据开始的索引
    test_start = len(all_series) + TRAIN_END

    # 对所有的数据进行相同的归一化
    train_data_numpy = np.array(train_data)
    train_mean = np.mean(train_data_numpy)
    train_std  = np.std(train_data_numpy)
    train_data_numpy = (train_data_numpy - train_mean) / train_std
    train_data_tensor = torch.Tensor(train_data_numpy)

    all_series = (all_series - train_mean) / train_std
    all_series = torch.Tensor(all_series)

    prediction_array = 0

    # in this iteration
    # we calculate the next day's price using the real data
    for i in range(DAYS_BEFORE, TOTAL_SIZE):
        x = all_series[i - DAYS_BEFORE:i]
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2)
        
        if torch.cuda.is_available():
            x = x.cuda()

        y = rnn(x)

        if i < TRAIN_END:
            generate_data_train.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)
        else:
            generate_data_test_real.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)


    for i in range(DAYS_BEFORE, TOTAL_SIZE):
        if i <= TRAIN_END:
            x = all_series[i - DAYS_BEFORE:i]
        else:
            x = prediction_array.reshape(DAYS_BEFORE)
        
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2)
        
        if torch.cuda.is_available():
            x = x.cuda()

        y = rnn(x)
        if i >= TRAIN_END:
            prediction_array = torch.cat((x[:, 1:], y.reshape(1, 1, 1)), dim=1).detach()

        if i >= TRAIN_END:
            generate_data_test_predict.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)
            
    plt.figure(figsize=(12,8))
    plt.plot(df_index, all_series.clone().numpy()* train_std + train_mean, 'r', label='real_data')
    plt.plot(df_index[DAYS_BEFORE: TRAIN_END], generate_data_train, 'b', label='generate_train', )
    plt.plot(df_index[TRAIN_END:], generate_data_test_real, 'k', label='test_real')
    plt.plot(df_index[TRAIN_END:], generate_data_test_predict, 'y', label='generate_predict')
    plt.legend()
    plt.savefig('predict.png')


train_model()
eval_model()