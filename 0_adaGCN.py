#%%
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import bisect
from torch.nn.utils import weight_norm
import pickle
import warnings
warnings.filterwarnings("ignore")
import torch
from torch_geometric.nn.dense.linear import Linear
from My_utils.evaluation_scheme import evaluation
from torch.nn.utils.rnn import pad_sequence
#%%
with open('./10-ada_GCN/smoothed_data.pkl', 'rb') as file:
    irregular_data = pickle.load(file)
# with open('smoothed_data.pkl', 'rb') as file:
#     irregular_data = pickle.load(file)
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

val_data_tensors = {}

for iscid, df in irregular_data.items():
    val_data_tensors[iscid] = torch.cat([d[iscid] for d in Valid])

#%%

class adaGCNConv(nn.Module):
    def __init__(self, num_nodes: int, timespan: int, out_channels: int,drop):

        super(adaGCNConv, self).__init__()

        self.num_nodes = num_nodes

        self.timespan = timespan
        self.out_channels = out_channels
                   
        self.droupout=torch.nn.Dropout(p=drop)
        self.conv=weight_norm(nn.Conv1d(num_nodes,int(num_nodes//2),kernel_size=7))
        

        self.lin = Linear(num_nodes, num_nodes, bias=True,
                          weight_initializer="kaiming_uniform")
        self.out_lin = Linear(num_nodes, out_channels, bias=True,
                          weight_initializer="kaiming_uniform")
        self.conv_lin = Linear(150, 21, bias=True,
                          weight_initializer="kaiming_uniform")

    def Time_Calibration(self, cumulative_sum, time_intervals):
        closest_indices = []

        cumulative_sum = cumulative_sum.tolist()  # 将张量转换为Python列表

        for val in time_intervals:
            closest_idx = bisect.bisect_left(cumulative_sum, val)
            if closest_idx == 0:
                closest_indices.append(0)
            elif closest_idx == len(cumulative_sum):
                closest_indices.append(len(cumulative_sum) - 1)
            else:
                if abs(cumulative_sum[closest_idx] - val) < abs(cumulative_sum[closest_idx - 1] - val):
                    closest_indices.append(closest_idx)
                else:
                    closest_indices.append(closest_idx - 1)

        return closest_indices

    def Ada_Calibration(self, train_data_tensors, timespan):

        """
        :param train_data_tensors: dict of input, [ N, *].
        :param timespan: equal to timestep, int.
        :return:
            X,Y: formatted data,list [ l, D].
        """
        X_list = []
        Y_list = []

        # 定义阈值
        idx = int(timespan / 60)

        # 遍历 train_data_tensors 字典中的每个张量
        for iscid, tensor in train_data_tensors.items():

            cumulative_sum = torch.cumsum(tensor[:, 1], dim=0)
            time_intervals = [i * 60 for i in range(0, int(cumulative_sum[-1]/60)+1)]
            closest_indices = self.Time_Calibration(cumulative_sum, time_intervals)
            X = []
            Y = []
            # for idx in closest_indices:
            for i in range(len(closest_indices) - idx):
                X.append(tensor[closest_indices[i]:closest_indices[i + idx], 0])
                Y.append(tensor[closest_indices[i + idx]:closest_indices[i + idx] + 1, 0])

            X_list.append(X)
            Y_list.append(Y)

        return X_list, Y_list

    def Ada_message_agg(self, X_list, Y_list, i):
        """
        :param inputs: X_list, [ N, *].
        :return:
            AGG_info , [ N,N, D].
        """

        A = [x_data[i] for x_data in X_list]

        max_length = max(len(tensor) for tensor in A)
        #
        padded_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=0) for tensor in A]

        # 使用pad_sequence将张量列表转换为一个张量
        result_tensor = pad_sequence(padded_tensors, batch_first=True)

        feature_result = torch.matmul(result_tensor, result_tensor.t())  # 特征矩阵乘以其转置
        # 对每行进行归一化，将最小值映射到 0，最大值映射到 1
        min_vals = feature_result.min(dim=1, keepdim=True).values
        max_vals = feature_result.max(dim=1, keepdim=True).values
        # 使用最小值和最大值进行归一化
        epsilon = 1e-2  # 小的常数，用于避免除以零
        feature_result_normalized = (feature_result - min_vals) / (max_vals - min_vals + epsilon)
        GT = torch.vstack([Y_data[i] for Y_data in Y_list])

        return feature_result_normalized, GT  #tensor (21,21)  and (21,1)

    def calculate_edge_weights(self,dist_matrix, scale=1.0):
        # 计算指数函数中的指数部分
        exponents = -scale * dist_matrix

        # 计算边的权重，这里使用指数函数
        edge_weights = torch.exp(exponents)

        return edge_weights

    # def calculate_edge_weights(self, dist_matrix, sigma=1, epsilon=0.01):
    #     # 计算指数函数中的指数部分
    #     squared_distances = dist_matrix ** 2
    #     exponents = - squared_distances / (sigma ** 2)
    #
    #     # 应用阈值
    #     exponents[exponents < np.log(epsilon)] = np.log(epsilon)
    #
    #     # 计算边的权重，这里使用指数函数
    #     edge_weights = torch.exp(exponents)
    #
    #     return edge_weights

    def calculate_laplacian(self, matrix, edge_weights):
        matrix = matrix + torch.eye(matrix.size(0), device=device)
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )

        # 将权重矩阵乘以拉普拉斯矩阵
        weighted_laplacian = edge_weights * normalized_laplacian

        return weighted_laplacian

    def forward(self, train_data_tensors, graph,dist_matrix):
        """
        :param inputs: input features, [ N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [ N, D].
        """

        X_list, Y_list = self.Ada_Calibration(train_data_tensors, self.timespan)

        edge_weights=self.calculate_edge_weights(dist_matrix)
        laplacian = self.calculate_laplacian(graph, edge_weights)

        trans_out_list = []
        GT_list = []
        min_length = min(len(lst) for lst in X_list) #can be conducted more
        for i in range(min_length):
            agg_info, GT = self.Ada_message_agg(X_list, Y_list, i)  # agg_info(23,23),GT(23,1)

            AX=torch.mm(laplacian,agg_info)
            AXW=AX.mm(self.lin.weight)

            ax_drop=self.droupout(AXW)
            
            trans_out=F.leaky_relu(ax_drop, 1/5.5)

            # input = torch.vstack([torch.mean(x_data[i]) for x_data in X_list]) #(21,1)

            input = torch.vstack([x_data[i][-1] if len(x_data[i]) > 0 else torch.mean(x_data[i]) for x_data in X_list])  # (21,1)
            input[input != input] = 0

            trans_out = self.out_lin(trans_out)+input  #skip connect # 小的常数，用于避免除以零

            # trans_out=self.conv(trans_out.unsqueeze(0).permute(0,2,1))
            # trans_out = self.conv_lin(trans_out.view(1,-1)).view(-1,1)

            trans_out_list.append(trans_out.unsqueeze(0))
            GT_list.append(GT.unsqueeze(0))

        trans_out_stacked = torch.vstack(trans_out_list)  # (循环次数, 原本维度)
        GT_stacked = torch.vstack(GT_list)

        return trans_out_stacked/100, GT_stacked/100

#%%
adj=pd.read_csv(r'./10-ada_GCN/NEW_ADJ.csv')
# adj=pd.read_csv(r'NEW_ADJ.csv')
adj_matrix = torch.tensor(adj.values, dtype=torch.float32)

"""dist_weight"""
dist_adj=pd.read_csv(r'./10-ada_GCN/NEW_DIS_ADJ.csv')
# dist_adj=pd.read_csv(r'NEW_DIS_ADJ.csv')
dist_matrix = torch.tensor(dist_adj.values, dtype=torch.float32)

"""paras"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

#%%
start_time = time.time()
ada_G=adaGCNConv(num_nodes=21,timespan=5*60,out_channels=1,drop=0.1).to(device)
optimizer = torch.optim.AdamW(ada_G.parameters(), lr=0.0001,
                              betas=(0.9, 0.9999), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=5, factor=0.95)
criterion = nn.MSELoss()

train_loss_all = []
val_loss_all = []
best_val_loss = float("inf")
best_model = None
ada_G.train()  # Turn on the train mode
# total_loss = 0.

epochs =50
for epoch in range(epochs):
    train_loss = 0
    train_num = 0

    for train_data_tensors in Train:

        optimizer.zero_grad()

        train_data_tensors = {k: v.to(device) for k, v in train_data_tensors.items()}

        pred_y, y = ada_G(train_data_tensors, adj_matrix.to(device), dist_matrix.to(device))

        loss = criterion(pred_y.squeeze(), y.squeeze())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(ada_G.parameters(), 0.1)  # 梯度裁剪，放backward和step直接

        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print('-' * 89)
        print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
        print('-' * 89)

    ada_G.eval()

    with torch.no_grad():
        val_data_tensors = {k: v.to(device) for k, v in val_data_tensors.items()}
        data, target = ada_G(val_data_tensors, adj_matrix.to(device), dist_matrix.to(device))
        val_loss = criterion(data.squeeze(), target.squeeze())

    scheduler.step(val_loss)

    ada_G.train()  # 将模型设置回train()模式

    print('Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, val_loss))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = ada_G
        print('best_Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, best_val_loss))

    train_loss_all.append(loss.item())
    val_loss_all.append(val_loss.item())



# 绘制loss曲线
plt.figure()
adj_val_loss = [num * (train_loss_all[0] / val_loss_all[0]) for num in val_loss_all]
plt.plot(range(1, epochs + 1), train_loss_all, label='Train Loss')
plt.plot(range(1, epochs + 1), adj_val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%
# hyperparameters_str = f"lr_{lr}_span_{t}"
# model_path = f"./ada_GCN/model/AdaGCN_{hyperparameters_str}.pth"
# torch.save(best_model, model_path)

# test_data_tensors = {}
#
# for iscid, df in irregular_data.items():
#     test_data_tensors[iscid] = torch.cat([d[iscid] for d in Test])


ada_G.eval()  # 进入评估模式
with torch.no_grad():
    A=[]
    B=[]
    for test_data_tensors in Test:
        test_data_tensors = {k: v.to(device) for k, v in test_data_tensors.items()}
        pred_y, y = ada_G(test_data_tensors, adj_matrix.to(device), dist_matrix.to(device))
        predictions = pred_y.squeeze().to('cpu').numpy()
        targets = y.squeeze().to('cpu').numpy()
        A.append(predictions)
        B.append(targets)

Pred=np.vstack(A)
GT=np.vstack(B)

TF=[]
for i in range(21):

    Metric_TF = np.array(evaluation(100 * GT[:, i].reshape(-1, 1),
                                        100 * Pred[:, i].reshape(-1, 1)))
    TF.append(Metric_TF)

TF_ARRAY=np.vstack(TF)

# 打印 TF_ARRAY 和 SGN_ARRAY
print("TF_ARRAY:\n", TF_ARRAY)
# 计算按列求平均值
tf_mean = np.mean(TF_ARRAY, axis=0)
# 打印按列求的平均值
print("The mean of TF_ARRAY by column:\n", tf_mean)
#%%


end_time = time.time()

# 计算代码运行时间
total_time = end_time - start_time
print("代码运行时间为：", total_time, "秒")
    # print(hyperparameters_str)

#%%
"""load model and test"""
ada_G = torch.load("10-ada_GCN/model/AdaGCN_lr_0.0001_span_300.pth").to(device)
# ada_G = torch.load("10-ada_GCN/model/AdaGCN_lr_0.001_span_300.pth").to(device)
# 设置模型为评估模式
ada_G.eval()
with torch.no_grad():
    A=[]
    B=[]
    for test_data_tensors in Test:
        test_data_tensors = {k: v.to(device) for k, v in test_data_tensors.items()}
        pred_y, y = ada_G(test_data_tensors, adj_matrix.to(device), dist_matrix.to(device))
        predictions = pred_y.squeeze().to('cpu').numpy()
        targets = y.squeeze().to('cpu').numpy()
        A.append(predictions)
        B.append(targets)

Pred=np.vstack(A)
GT=np.vstack(B)

TF=[]
for i in range(21):

    Metric_TF = np.array(evaluation(100 * GT[:, i].reshape(-1, 1),
                                        100 * Pred[:, i].reshape(-1, 1)))
    TF.append(Metric_TF)

TF_ARRAY=np.vstack(TF)

# 打印 TF_ARRAY 和 SGN_ARRAY
print("TF_ARRAY:\n", TF_ARRAY)
# 计算按列求平均值
tf_mean = np.mean(TF_ARRAY, axis=0)
# 打印按列求的平均值
print("The mean of TF_ARRAY by column:\n", tf_mean)

















        

        





        
        






