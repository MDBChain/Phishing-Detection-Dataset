from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,precision_recall_curve,average_precision_score
import warnings
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv,GraphConv
from torch_geometric.data import Data
from matplotlib import pyplot as plt
import time
import os
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
warnings.filterwarnings("ignore")
data1_load = pd.read_csv('allnodes_enhancement.csv')
data2_load = pd.read_csv('Enhangce_adjacency_matrix.csv')

adj_matrix = data2_load.iloc[:, 1:].values

# feature = data1_load.iloc[:, 1:19].values  # 从第0列到第5列
# 读取第一列和第3到23列的数据
feature = data1_load.iloc[:, [0] + list(range(4, 23))].values

label = data1_load.iloc[:, 23].values

# print("feature",feature)
# print("feature",feature.shape)
# print("label",label)
# print("label",label.shape)
feature = torch.FloatTensor(feature)
# print("feature",feature)
# print("feature",feature.shape)
# feature = feature.to(torch.double)
# print(feature.dtype)

label = torch.LongTensor(label)


feature_train, feature_test, label_train, label_test = train_test_split(feature, label,
                                                                        test_size=0.3, random_state=11)
feature_train, feature_test, label_train, label_test = feature_train.to(device), feature_test.to(device), \
                                                       label_train.to(device), label_test.to(device)
train_dataset = TensorDataset(feature_train, label_train)
test_dataset = TensorDataset(feature_test, label_test)

# 创建 DataLoader 对象，设置 batch size 为 89
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# breakpoint()
# label = label.double()
adj_matrix = torch.LongTensor(adj_matrix)
adj_matrix = adj_matrix.to(device)

# adj_matrix = adj_matrix.to(torch.int64)
# adj_matrix = adj_matrix.double()
print("adj_matrix\n",adj_matrix)
print("adj_matrix",adj_matrix.shape)
# breakpoint()
# print("feature",feature)
# print("feature",feature.shape)#feature torch.Size([890, 20])
# print("label",label)
# print("label",label.shape)#torch.Size([890])

# breakpoint()
# print("adjacency_matrix",adjacency_matrix)
# print("adjacency_matrix",adjacency_matrix.shape)#adjacency_matrix torch.Size([890, 890])
# 将邻接矩阵转换为边索引
# edge_index = adj_matrix.nonzero().t()
# edge_index = edge_index.to(torch.int64)

# print("edge_index",edge_index)
# print("edge_index",edge_index.shape)#edge_index torch.Size([2, 792100])
# breakpoint()
# 创建 PyTorch Geometric Data 对象
class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)#####记得调用一下

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
class TemporalFeatureNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalFeatureNetwork, self).__init__()
        self.conv1d_blocks = nn.Sequential(
            Conv1DBlock(input_dim, hidden_dim, kernel_size=1),
            # Assuming there might be more Conv1D blocks here
            Conv1DBlock(hidden_dim, input_dim, kernel_size=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Adaptive average pooling to reduce the dimension

    def forward(self, x):
        x = self.conv1d_blocks(x)
        x = self.avg_pool(x)
        # Flattening the output for the attention layer
        x = torch.flatten(x, start_dim=1)
        return x

# Adjusting the LSTM-based Attention Layer to include an attention mechanism
class GRUAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUAttentionLayer, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, input_dim, batch_first=True)

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        # LSTM expects input shape (batch, seq_len, features), ensuring x matches this shape
        x = x.unsqueeze(1)  # Adding a sequence length of 1
        # print(x.size())
        gru_out, _ = self.gru1(x)
        # MultiheadAttention expects input shape (batch, seq_len, features), lstm_out matches this shape
        attn_output, _ = self.attention(gru_out, gru_out, gru_out)
        gru_out, _ = self.gru2(attn_output)

        gru_out = gru_out.squeeze(1)  # Remove the sequence length dimension
        return gru_out


class GcnEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GcnEncoder, self).__init__()
        self.sageconv1 = SAGEConv(input_dim, hidden_dim)
        self.sageconv2 = SAGEConv(hidden_dim, output_dim)
        self.fc = nn.Linear(54, 2)
        self.gru_attention = GRUAttentionLayer(8, hidden_dim)
        # self.conv1d_block = Conv1DBlock(8,hidden_dim,1)
        self.conv1d_block = TemporalFeatureNetwork(8,hidden_dim)

        # self.sf = nn.Softmax(dim=1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        basic = x[:, :11] # torch.Size([256, 11])
        timeseries = x[:, 11:] # torch.Size([256, 8])
        # 多头注意力GRU模块
        gru_timeseries = self.gru_attention(timeseries)#torch.Size([256, 8])

        # 一维卷积模块
        conv1d_timeseries = timeseries.unsqueeze(1)
        conv1d_timeseries = conv1d_timeseries.permute(0, 2, 1)
        conv1d_timeseries = self.conv1d_block(conv1d_timeseries)#torch.Size([256, 8])

        # 总序列模块特征提取
        cat_timeseries = torch.cat((gru_timeseries, conv1d_timeseries), dim=1)#torch.Size([256, 16])

        # 图的数据输入
        graph_input = torch.cat((cat_timeseries, basic), dim=1)#torch.Size([256, 27])

        # 执行两个SAGEConv层的前向传播
        sage_basic = self.sageconv1(graph_input, edge_index)
        sage_basic = torch.relu(sage_basic)
        sage_basic = self.sageconv2(sage_basic, edge_index)#torch.Size([256, 27])

        #总特征
        sum_features = torch.cat((cat_timeseries,sage_basic, basic), dim=1)#torch.Size([256, 54])
        classify = self.fc(sum_features)
        # classify = self.sf(classify)
        return x, classify

# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
num_features = 27
epochs = 500
compare = 0.3
train_acc, train_loss,test_acc, test_loss = [], [],[], []
# 创建一个 GcnEncoder 实例，并传入相应的维度参数
encoder = GcnEncoder(input_dim=num_features, hidden_dim=128, output_dim=num_features).to(device)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
for j in range (epochs):
    epoch_loss = 0.0
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1sum = 0.0
    f = 0.0
    roc = 0.0

    encoder.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("batch_idx",batch_idx)
        # print("data",data)
        # print("data", data.shape)
        # print("target",target)
        # print("target", target.shape)
        data_features = data[:, 1:]  # 选择第 1 到最后一列的特征
        selected_nodes = data[:, 0]
        selected_nodes = selected_nodes.to(torch.long)
        adj_selected = adj_matrix[selected_nodes][:, selected_nodes]
        edge_index = adj_selected.nonzero().t()
        edge_index = edge_index.to(torch.int64)
        data = Data(x=data_features, edge_index=edge_index)
        # output_train__dir = "BoT-IoT_output_train_images"
        encoder.train()

        # label_batch = torch.squeeze(label_batch)
        # print("label_batch", label_batch)
        optimizer.zero_grad()
        output,classify = encoder(data)
        classify_pred = torch.argmax(classify, dim=1).cpu().numpy()
        # print("output_pred",classify_pred)
        # print(output)
        # print(output.shape)
        # print("pred_y",classify)
        # label = label.unsqueeze(1)
        acc = accuracy_score(target, classify_pred)
        pr = precision_score(target, classify_pred)
        f1 = f1_score(target, classify_pred)
        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(target, classify_pred)

        # 计算 ROC 曲线下的面积（AUC）
        roc_auc = auc(fpr, tpr)
        # if acc > 0.95:
        #     print("classify_pred",classify_pred)
        #     print("target",target)
        #     breakpoint()

        loss = criterion(classify, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        accuracy += acc
        precision += pr
        f1sum += f1
        roc += roc_auc
        f += 1
    train_acc.append(accuracy/f)
    train_loss.append(epoch_loss/f)
    # if (j+1) % 5 == 0:
    #     print("epoch, loss", j + 1, epoch_loss/f)
    #     print("acc", accuracy/f)
    #     print("precision", precision/f)
    #     print("f1", f1sum/f)
    #     print("roc_auc", roc/f)
    #     print("\n")

    test_epoch_loss = 0.0
    test_accuracy = 0.0
    test_precision = 0.0
    test_recall = 0.0
    test_f1sum = 0.0
    test_f = 0.0
    test_roc = 0.0

    y_test_true = []
    y_test_pre = []

    output_test__dir = "SAGEConv_CM"
    encoder.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data_features = data[:, 1:]  # 选择第 1 到最后一列的特征
            selected_nodes = data[:, 0]
            selected_nodes = selected_nodes.to(torch.long)
            adj_selected = adj_matrix[selected_nodes][:, selected_nodes]
            edge_index = adj_selected.nonzero().t()
            edge_index = edge_index.to(torch.int64)
            data = Data(x=data_features, edge_index=edge_index)

            output, classify = encoder(data)
            classify_pred = torch.argmax(classify, dim=1).cpu().numpy()

            y_test_pre.extend(classify_pred)
            y_test_true.extend(target)

            loss = criterion(classify, target)
            test_epoch_loss += loss.item()
            test_f += 1

        epoch_loss = test_epoch_loss / test_f
        acc1 = accuracy_score(y_test_true, y_test_pre)
        test_acc.append(acc1)
        test_loss.append(epoch_loss)
        # if (j + 1) % 50 == 0:
        if epoch_loss < compare:
            compare = epoch_loss

            acc = accuracy_score(y_test_true, y_test_pre)
            pr = precision_score(y_test_true, y_test_pre)
            recall1 = recall_score(y_test_true, y_test_pre)
            f1 = f1_score(y_test_true, y_test_pre)

            fpr, tpr, thresholds = roc_curve(y_test_true, y_test_pre)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test_true, y_test_pre)
            ap = average_precision_score(y_test_true, y_test_pre)

            cm = confusion_matrix(y_test_true, y_test_pre)
            timestamp = str(int(time.time()))
            file_path = f'SAGEConv_data/epoch{j + 1}_{timestamp}_SAGEConv_train_test_acc_loss.npz'
            np.savez(file_path, fpr=fpr, tpr=tpr, precision=precision, recall=recall, auc=roc_auc, ap=ap,
                     acc = acc,pr = pr,recall1 = recall1,f1 = f1,cm = cm,compare = compare,y_test_true = y_test_true,
                     y_test_pre = y_test_pre)

            file_path_model = f'SAGEConv_model/epoch{j + 1}_{timestamp}_SAGEConv.pth'
            torch.save(encoder.state_dict(), file_path_model)
            plt.matshow(cm, cmap=plt.cm.Greens)
            plt.colorbar()
            for i in range(len(cm)):
                for k in range(len(cm)):
                    plt.annotate(cm[i, k], xy=(i, k), horizontalalignment='center', verticalalignment='center',
                                 fontsize=5)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            # filename = os.path.join(loss__dir, f"loss_epoch_{epoch}_{timestamp}.png")
            filename = os.path.join(output_test__dir, f"output_epoch_{j+1}_{timestamp}.jpg")
            plt.savefig(filename, dpi=300)
            # plt.show()
            # 清除当前图像
            plt.clf()
            print("test_epoch, loss", j + 1, epoch_loss)
            print("test_acc", acc)
            print("test_precision", pr)
            print("test_f1", f1)
            print("test_recall", recall1)
            print("test_roc_auc", roc_auc)
            print("test_ap", ap)
            print("timestamp", timestamp)

            print("\n")
            # if test_epoch_loss / test_f < 0.25:
            # if test_f > 0:
            # if test_roc / test_f > 0.94:
        # if (j + 1) % 50 == 0:
        #     print("test_epoch, loss", j + 1, epoch_loss)
        #     print("test_acc", acc)
        #     print("test_precision", pr)
        #     print("test_f1", f1)
        #     print("test_recall", recall1)
        #     print("test_roc_auc", roc_auc)
        #     print("test_ap", ap)
        #     print("timestamp", timestamp)

timestamp = str(int(time.time()))
file_path = f'SAGEConv_data/SAGEConv_train_test_acc_loss_epoch_{j + 1}_{timestamp}.npz'
np.savez(file_path, train_acc = train_acc, train_loss = train_loss,
         test_acc = test_acc, test_loss = test_loss)



