
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
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.lstm = nn.GRU(
            input_size=seq_len,               # 输入纬度   记得加逗号
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=True)                  #是否双向Bi-GRU
        self.out = nn.Linear(hidden_size*2, pre_len)

    def forward(self, x):
        temp, _ = self.lstm(x)
        s, b, h = temp.size()
        temp = temp.view(s * b, h)
        outs = self.out(temp)
        lstm_out = outs.view(s, b, -1)
        return lstm_out/100
#%%
start_time = time.time()
RULTS=[]
# Hyper Parameters
seq_len = 5  #
hidden_size = 32
pre_len = 1
num_layers = 3
lr = 0.0001
epochs = 50
"all results"
for iscid, data in train_data_tensors.items():
    # if iscid != 29:
    #     continue  # 跳过iscid不为29的情况

    timestep=5

    train_windows = data[:,0].unfold(dimension=0, size=timestep+1, step=1)
    trainX, trainY=train_windows[:,:timestep].unsqueeze(1),train_windows[:,-1].unsqueeze(1)
    train_dataset = Data.TensorDataset(trainX, trainY)

    val_windows = val_data_tensors[iscid][:,0].unfold(dimension=0, size=timestep+1, step=1)
    valX, valY=val_windows[:,:timestep].unsqueeze(1),val_windows[:,-1].unsqueeze(1)
    val_dataset = Data.TensorDataset(valX, valY)

    # put into liader
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRU_model = GRU().to(device)

    optimizer = torch.optim.AdamW(GRU_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
    loss_func = nn.MSELoss()

    #  训练
    train_loss_all = []
    val_loss_all = []
    best_val_loss = float("inf")
    best_model = None
    GRU_model.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(epochs):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pre_y = GRU_model(x)

            loss = loss_func(pre_y, y/100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(GRU_model.parameters(), 0.1)  # 梯度裁剪，放backward和step直接

            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)

            total_loss += loss.item()

            optimizer.step()

        if (epoch + 1) % 1 == 0:
            print('-' * 89)
            print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
            print('-' * 89)

        train_loss_all.append(train_loss / train_num)

        # 验证阶段
        GRU_model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = GRU_model(data)
                val_loss += loss_func(output, target/100).item()

        val_loss /= len(val_loader.dataset)
        val_loss_all.append(val_loss)

        scheduler.step(val_loss)

        GRU_model.train()  # 将模型设置回train()模式

        print('Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = GRU_model
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
    testX, testY=test_windows[:,:timestep].unsqueeze(1),test_windows[:,-1].unsqueeze(1)
    # test_dataset = Data.TensorDataset(testX, testY)
    with torch.no_grad():
        output = best_model(testX.to(device))
        predictions = output.squeeze().to('cpu').numpy()
        targets = testY.squeeze().to('cpu').numpy()

    Metric = np.array(evaluation(targets.reshape(-1)
                                  , 100 *predictions.reshape(-1)))
    print(Metric)
    RULTS.append(np.array(Metric))

    del GRU_model




R=np.array(RULTS)
print(R)

end_time = time.time()

# 计算代码运行时间
total_time = end_time - start_time
print("代码运行时间为：", total_time, "秒")
tf_mean = np.mean(R, axis=0)
# 打印按列求的平均值
print("The mean of TF_ARRAY by column:\n", tf_mean)