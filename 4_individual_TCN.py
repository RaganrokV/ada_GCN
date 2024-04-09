
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
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs=5, n_outputs=1, kernel_size=20, stride=1, dilation=None, padding=3, dropout=0.1):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        if dilation is None:
            dilation = 2
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """

        return self.network(x) / 150
#%%
start_time = time.time()
RULTS=[]
# Hyper Parameters
input_size = 5  # 输入大小
epochs=50
lr=0.001

"all results"
for iscid, data in train_data_tensors.items():

    timestep=5

    train_windows = data[:,0].unfold(dimension=0, size=timestep+1, step=1)
    trainX, trainY=train_windows[:,:timestep].unsqueeze(1).transpose(1, 2),train_windows[:,-1].unsqueeze(1)
    train_dataset = Data.TensorDataset(trainX, trainY)

    val_windows = val_data_tensors[iscid][:,0].unfold(dimension=0, size=timestep+1, step=1)
    valX, valY=val_windows[:,:timestep].unsqueeze(1).transpose(1, 2),val_windows[:,-1].unsqueeze(1)
    val_dataset = Data.TensorDataset(valX, valY)

    # put into liader
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TCN_model = TemporalConvNet(num_inputs=input_size, num_channels=[32, 32, 1],
                                kernel_size=4, dropout=0.1).to(device)

    optimizer = torch.optim.AdamW(TCN_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
    loss_func = nn.MSELoss()

    #  训练
    train_loss_all = []
    val_loss_all = []
    best_val_loss = float("inf")
    best_model = None
    TCN_model.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(epochs):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pre_y = TCN_model(x)

            loss = loss_func(pre_y, y/150)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(TCN_model.parameters(), 0.1)  # 梯度裁剪，放backward和step直接

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
        TCN_model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = TCN_model(data)
                val_loss += loss_func(output, target/150).item()

        val_loss /= len(val_loader.dataset)
        val_loss_all.append(val_loss)

        scheduler.step(val_loss)

        TCN_model.train()  # 将模型设置回train()模式

        print('Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = TCN_model
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
    testX, testY=test_windows[:,:timestep].unsqueeze(1).transpose(1, 2),test_windows[:,-1].unsqueeze(1)
    # test_dataset = Data.TensorDataset(testX, testY)
    with torch.no_grad():
        output = best_model(testX.to(device))
        predictions = output.squeeze().to('cpu').numpy()
        targets = testY.squeeze().to('cpu').numpy()

    Metric = np.array(evaluation(targets.reshape(-1)
                                  , 150 *predictions.reshape(-1)))
    print(Metric)
    RULTS.append(np.array(Metric))

    del best_model




R=np.array(RULTS)
print(R)

end_time = time.time()

# 计算代码运行时间
total_time = end_time - start_time
print("代码运行时间为：", total_time, "秒")
tf_mean = np.mean(R, axis=0)
# 打印按列求的平均值
print("The mean of TF_ARRAY by column:\n", tf_mean)
