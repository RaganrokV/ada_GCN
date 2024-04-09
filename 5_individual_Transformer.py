# -*- coding: utf-8 -*-
import math
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from torch import nn
import time
import torch.utils.data as Data
from matplotlib import pyplot as plt
from My_utils.evaluation_scheme import evaluation
import torch
from torch_geometric.typing import PairTensor  # noqa

#%%
with open('10-ada_GCN/smoothed_data.pkl', 'rb') as file:
    irregular_data = pickle.load(file)
#%%
A=[]
for i in range(44):
    split_indices = {}
    # 遍历字典中的每个 DataFrame
    for iscid, df in irregular_data.items():
        # 获取 'CYCLENGTH' 列的累积和
        cumulative_sum = df['CYCLENGTH'].cumsum()

        # 寻找划分索引，使累积和首次超过阈值
        train_end_index = (cumulative_sum >= (i+1)*24 * 60 * 60).idxmax()

        # 存储划分索引
        split_indices[iscid] = {
            'train_end_index': train_end_index}
    A.append(split_indices)

Daily_division = []

for idx in range(44):
    split_indices = {}
    train_data = {}
    train_data_tensors = {}

    if idx == 0:
        prev_split_info = {iscid: {'train_end_index': None} for iscid in irregular_data.keys()}
    else:
        prev_split_info = A[idx - 1]

    for iscid, df in irregular_data.items():
        cumulative_sum = df['CYCLENGTH'].cumsum()
        train_end_index = (cumulative_sum >= (idx + 1) * 24 * 60 * 60).idxmax()

        split_indices[iscid] = {'train_end_index': train_end_index}

        if idx == 0:
            train_df = df.iloc[:train_end_index + 1].copy()
        else:
            split_info1 = prev_split_info[iscid]
            train_df = df.iloc[split_info1['train_end_index']:train_end_index + 1].copy()

        train_data[iscid] = train_df
        train_data_tensors[iscid] = torch.tensor(train_df.values, dtype=torch.float32)

    Daily_division.append(train_data_tensors)

#%%
Train = Daily_division[:30]
Valid = Daily_division[30:34]
Test = Daily_division[34:]

train_data_tensors = {}

for iscid, df in irregular_data.items():
    train_data_tensors[iscid] = torch.cat([d[iscid] for d in Train])


val_data_tensors = {}

for iscid, df in irregular_data.items():
    val_data_tensors[iscid] = torch.cat([d[iscid] for d in Valid])

test_data_tensors = {}

for iscid, df in irregular_data.items():
    test_data_tensors[iscid] = torch.cat([d[iscid] for d in Test])
#%%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class TransformerTS(nn.Module):
    def __init__(self,
                 input_dim,
                 dec_seq_len,
                 out_seq_len,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 activation,
                 custom_encoder=None,
                 custom_decoder=None):
        r"""
        Args:
            input_dim: dimision of imput series
            d_model: the number of expected features in the encoder/decoder inputs (default=512).
            nhead: the number of heads in the multiheadattention models (default=8).
            num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
            num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
            custom_encoder: custom encoder (default=None).
            custom_decoder: custom decoder (default=None).


        """
        super(TransformerTS, self).__init__()
        self.transform = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
        )
        self.pos = PositionalEncoding(d_model)
        self.enc_input_fc = nn.Linear(input_dim, d_model)
        self.dec_input_fc = nn.Linear(input_dim, d_model)
        # self.out_fc = nn.Linear(dec_seq_len * d_model, out_seq_len)
        self.out_fc = nn.Linear(d_model, out_seq_len)
        self.dec_seq_len=dec_seq_len

    def forward(self, x):
        # embedding
        embed_encoder_input = self.pos(self.enc_input_fc(x))
        embed_decoder_input = self.dec_input_fc(x[:,-dec_seq_len:, :])
        # transform
        x = self.transform(embed_encoder_input, embed_decoder_input)

        x = self.out_fc(x)


        return x/ 150

#%%
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

start_time = time.time()
RULTS=[]
# Hyper Parameters
input_dim = 1  # 输入大小
dec_seq_len = 1
out_seq_len = 1
lr = 0.001
epochs=50

"all results"
for iscid, data in train_data_tensors.items():

    timestep=5

    train_windows = data[:,0].unfold(dimension=0, size=timestep+1, step=1)
    trainX, trainY=train_windows[:,:timestep].unsqueeze(2),train_windows[:,-1].unsqueeze(1)
    train_dataset = Data.TensorDataset(trainX, trainY)

    val_windows = val_data_tensors[iscid][:,0].unfold(dimension=0, size=timestep+1, step=1)
    valX, valY=val_windows[:,:timestep].unsqueeze(2),val_windows[:,-1].unsqueeze(1)
    val_dataset = Data.TensorDataset(valX, valY)

    # put into liader
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_model = TransformerTS(input_dim,
                            dec_seq_len,
                            out_seq_len,
                            d_model=8,
                            nhead=4,
                            num_encoder_layers=2,
                            num_decoder_layers=2,
                            dim_feedforward=64,
                            dropout=0.1,
                            activation='relu',
                            custom_encoder=None,
                            custom_decoder=None).to(device)

    optimizer = torch.optim.AdamW(T_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
    loss_func = nn.MSELoss()

    #  训练
    train_loss_all = []
    val_loss_all = []
    best_val_loss = float("inf")
    best_model = None
    T_model.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(epochs):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pre_y = T_model(x)

            loss = loss_func(pre_y, y/150)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(T_model.parameters(), 0.1)  # 梯度裁剪，放backward和step直接

            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)

            total_loss += loss.item()

            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('-' * 89)
            print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
            print('-' * 89)

        train_loss_all.append(train_loss / train_num)

        # 验证阶段
        T_model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = T_model(data)
                val_loss += loss_func(output, target/150).item()

        val_loss /= len(val_loader.dataset)
        val_loss_all.append(val_loss)

        scheduler.step(val_loss)

        T_model.train()  # 将模型设置回train()模式

        print('Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = T_model
            print('best_Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, best_val_loss))


    # 绘制loss曲线
    plt.figure()
    adj_val_loss = [num * (train_loss_all[0]/val_loss_all[0]-1) for num in val_loss_all]
    plt.plot(range(1, epochs + 1), train_loss_all, label='Train Loss')
    plt.plot(range(1, epochs + 1), adj_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    best_model = best_model.eval()  # 转换成测试模式

    test_windows = test_data_tensors[iscid][:,0].unfold(dimension=0, size=timestep+1, step=1)
    testX, testY=test_windows[:,:timestep].unsqueeze(2),test_windows[:,-1].unsqueeze(1)
    # test_dataset = Data.TensorDataset(testX, testY)
    with torch.no_grad():
        output = best_model(testX.to(device))
        predictions = output.squeeze().to('cpu').numpy()
        targets = testY.squeeze().to('cpu').numpy()

    Metric = np.array(evaluation(targets.reshape(-1)
                                  , 150 *predictions.reshape(-1)))
    print(Metric)
    RULTS.append(np.array(Metric))

    del T_model




R=np.array(RULTS)
print(R)

end_time = time.time()

# 计算代码运行时间
total_time = end_time - start_time
print("代码运行时间为：", total_time, "秒")
tf_mean = np.mean(R, axis=0)
# 打印按列求的平均值
print("The mean of TF_ARRAY by column:\n", tf_mean)