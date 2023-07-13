import torch
from torch import nn
import torch.nn.functional as F
from .Network_Layer.attention_Layer import Channel_AttentionLayer, SpatialSELayer, Self_attentionLayer
from .Network_Layer.Transformer_Layer import PositionalEncoding
from .Network_Layer.Fusion_Layer import Weighted_Fusion, Weighted_Fusion3, Weighted_Fusion4, ElementWiseFusion
from .Network_Layer.DenseCNN_layer import DenseCNN


class Definition_Network(nn.Module):
    def __init__(self, time_step, pre_len, station_num, device, N, hidden_cha, hidden_lay):
        super().__init__()
        self.time_step = time_step
        self.pre_len = pre_len
        self.station_num = station_num
        self.device = device

        # Early-fusion
        self.lstm0 = nn.LSTM(input_size=time_step * 3, hidden_size=15, num_layers=1, bidirectional=True).to(self.device)
        self.DenseCNN = DenseCNN(in_channle=time_step * 3, Dense_layer=3, outsize=station_num).to(self.device)
        self.linear_EF = nn.Linear(in_features= time_step * 3 * station_num, out_features=N).to(self.device)

        # Branch 1
        self.Pos_Enco = PositionalEncoding(time_lag=time_step * 3, station_num=station_num).to(self.device)
        self.Multihead_Attention_1 = nn.MultiheadAttention(embed_dim=time_step * 3, num_heads=1).to(device).to(self.device)
        self.OD_Layer_Norm = nn.LayerNorm(normalized_shape=[self.station_num, self.station_num], eps=1e-5).to(self.device)
        self.POI_Layer_Norm = nn.LayerNorm(normalized_shape=[self.station_num, 23], eps=1e-5).to(self.device)
        self.lstm = nn.LSTM(input_size=time_step * 3, hidden_size=hidden_lay, num_layers=1, bidirectional=True).to(self.device)    # Num_lay 可以是超参数
        self.linear_B1_1 = nn.Linear(in_features=self.station_num * hidden_lay * 2, out_features=N * 2).to(self.device)
        self.linear_B1_2 = nn.Linear(in_features=N * 2, out_features=N).to(self.device)

        # Branch 2
        self.Spatial_SE = SpatialSELayer(num_channels=self.time_step * 3).to(self.device)  # 生成通道注意力
        self.Channel_Attention = Channel_AttentionLayer(channel=self.time_step * 3).to(self.device)  # 给原矩阵加上通道注意力
        self.Depthwise_CNN_1 = nn.Conv2d(in_channels=time_step * 3, out_channels=time_step * 3, kernel_size=3, padding=1, groups=time_step * 3).to(self.device)  # 深度分离卷积
        self.Pointwise_CNN_1 = nn.Conv2d(in_channels=time_step * 3, out_channels=hidden_cha, kernel_size=1, padding=0, groups=1).to(self.device)  # 逐点卷积
        self.Depthwise_CNN_2 = nn.Conv2d(in_channels=hidden_cha, out_channels=hidden_cha, kernel_size=3, padding=1, groups=hidden_cha).to(self.device)  # 深度分离卷积
        self.Pointwise_CNN_2 = nn.Conv2d(in_channels=hidden_cha, out_channels=1, kernel_size=1, padding=0, groups=1).to(self.device)  # 逐点卷积
        self.linear_B2_1 = nn.Linear(in_features=7225, out_features=N*2).to(self.device)  # 2-2178   3-3267   1-Layer 20480  1-Layer 8192   3-Layer 4096   dataset1 4096, detaset2 7225
        self.linear_B2_2 = nn.Linear(in_features=N*2, out_features=N).to(self.device)


        # Feature Fusion
        self.self_att_1 = Self_attentionLayer(23, station_num, 23).to(self.device)
        self.self_att_2 = Self_attentionLayer(23, station_num, 23).to(self.device)
        self.self_att_3 = Self_attentionLayer(23, station_num, 23).to(self.device)
        self.linear_POI_1 = nn.Linear(in_features=23, out_features=time_step * 3).to(self.device)
        self.linear_POI_2 = nn.Linear(in_features=23, out_features=station_num).to(self.device)
        self.linear_POI_3 = nn.Linear(in_features=23, out_features=station_num).to(self.device)
        self.Early_Fusion = Weighted_Fusion(in_features=2, out_features=2).to(self.device)
        self.Front_Fusion_L = Weighted_Fusion3(in_features=3, out_features=3).to(self.device)
        self.Front_Fusion_R = Weighted_Fusion4(in_features=4, out_features=4).to(self.device)
        self.Deep_Fusion = Weighted_Fusion3(in_features=3, out_features=3).to(self.device)

        self.linear_d = nn.Linear(in_features=N, out_features=station_num * pre_len).to(self.device)
        self.linear_L = nn.Linear(in_features=N, out_features=station_num * pre_len).to(self.device)
        self.linear_R = nn.Linear(in_features=N, out_features=station_num * pre_len).to(self.device)

        self.Late_Fusion = ElementWiseFusion(in_features=3, station_num=station_num, pre_len=pre_len).to(self.device)


    def forward(self, inflow, OD, POI):    # 注 batch_size=20，timestep=30
        # 输入数据整理  Inflow 数据、OD数据、POI数据
        inflow = inflow.type(torch.FloatTensor).to(self.device)  # (20, Num_Station, 30)
        OD = OD.type(torch.FloatTensor).to(self.device)  # (20, 30, Num_Station, Num_Station)
        OD = OD/torch.max(OD)  # OD 需要标准化
        POI = POI.type(torch.FloatTensor).to(self.device)  # (20, Num_Station, 23)
        POI = POI/torch.max(POI)  # OD 需要标准化
        # POI = POI.unsqueeze(1)  # (20, 1, Num_Station, 23)

        # Inflow 多时间模式合并  Multi-stack
        inflow_week = inflow[:, :, 0:self.time_step]                        # (20, Num_Station, 10)
        inflow_day = inflow[:, :, self.time_step:self.time_step * 2]        # (20, Num_Station, 10)
        inflow_time = inflow[:, :, self.time_step * 2:self.time_step * 3]   # (20, Num_Station, 10)
        inflow = torch.cat([inflow_week, inflow_day, inflow_time], dim=2)   # (20, Num_Station, 30)

        # 前融合  Early-fusion for Inflow and OD data
        # Inflow processed by Bi-LSTM / RHN
        Early_F1, (ht, ct) = self.lstm0(inflow)  # 接入LSTM层   torch.Size([20, Num_Station, 30])
        # OF processed by DenseCNN
        Early_F2 = self.DenseCNN(OD).permute(0, 2, 1)    # ([20, Num_Station, 30])
        Early_F = self.Early_Fusion(X1=Early_F1, X2=Early_F2)  # 加权融合
        Early_F = self.linear_EF(Early_F.reshape(Early_F.size()[0], -1))   # ([20, N / 512])


        # 左 Matrix Operation （OD Features → Inflow data）
        OD_features = torch.sum(input=OD, dim=2)   # (20, 30, 1, Num_Station)   自动简化为  (20, 30, Num_Station)
        OD_features = OD_features.permute(0, 2, 1)   # (20, Num_Station, 30)  转置 - 才能fusion
        # # 左 Attention Fusion
        # X_Branch1 = self.Front_Fusion_L(X1=inflow, X2=OD_features)   # 加权融合
        
        # 右 Matrix Operation （Inflow Features → OD data）
        inflow_sum = torch.sum(input=inflow, dim=1)  # (20, 30)   # 即每个时间步下整个网络的进站人数
        inflow_sum = inflow_sum.unsqueeze(1)   # (20, 1, 30)
        inflow_features = torch.div(inflow, inflow_sum)  # ([20, Num_Station, 30])   # 当前时间步下，每个车站进站人数在整个网络的进站人数所占比例
        inflow_features = torch.where(torch.isnan(inflow_features), torch.full_like(inflow_features, 0), inflow_features)   # 前面的inflow_sun有0的情况，所以div后出现的nan值需要去除
        inflow_features = inflow_features.permute(0, 2, 1)  # ([20, 30, Num_Station]) 转置 - 才能fusion
        inflow_features = inflow_features.unsqueeze(3)  # ([20, 30, Num_Station, 1])
        # # 右 Attention Fusion
        # X_Branch2 = self.Front_Fusion_R(X1=OD, X2=inflow_features)   # 加权融合

        # 左 POI数据接入
        POI_att1 = self.self_att_1(POI)  # (20, Num_Station, 23)
        POI_inflow = F.relu(self.linear_POI_1(POI_att1))  # (20, Num_Station, 1)   /   (20, Num_Station, 64)
        X_Branch1 = self.Front_Fusion_L(X1=inflow, X2=OD_features, X3=POI_inflow)  # 加权融合 (Inflow, OD_Feature, POI_Inflow)


        # 右 POI数据接入
        POI_att2 = self.self_att_2(POI).unsqueeze(1)                 # (20, 1, Num_Station, 23)
        POI_OD_1 = F.relu(self.linear_POI_2(POI_att2))  # (20, 1, Num_Station, 1)  / (20, 1, Num_Station, 64)
        POI_att3 = self.self_att_3(POI).unsqueeze(1)               # (20, 1, Num_Station, 23)
        POI_OD_2 = F.relu(self.linear_POI_3(POI_att3))  # (20, 1, Num_Station, 1)  / (20, 1, Num_Station, 64)
        POI_OD_2 = POI_OD_2.permute(0, 1, 3, 2)  # (20, 1, 1, Num_Station)  /  (20, 1, 64, Num_Station)
        X_Branch2 = self.Front_Fusion_R(X1=OD, X2=inflow_features, X3=POI_OD_1, X4=POI_OD_2)  # 加权融合

        # 左 Transformer LSTM 层
        X_Branch1 = self.Pos_Enco(X_Branch1)  # ([20, 64, 30])    # 位置编码
        X_Branch1, attn_output_weights = self.Multihead_Attention_1(query=X_Branch1, key=X_Branch1, value=X_Branch1)
        X_Branch1 = X_Branch1 + inflow    # 残差连接
        # # FeedForward 前馈使用LSTM网络
        X_Branch1, (ht, ct) = self.lstm(X_Branch1)   # 接入LSTM层   torch.Size([20, Num_Station, 5])
        X_Branch1 = X_Branch1.reshape(X_Branch1.size()[0], -1)  # (batch_size, Num_Station*hidden_lay)   相当于flatten
        X_Branch1 = F.relu(self.linear_B1_1(X_Branch1))   # 全连接层1
        X_Branch1 = F.relu(self.linear_B1_2(X_Branch1))   # 全连接层2  # (batch_size, 512)

        # 右 深度注意力模块
        X_Branch2 = self.Spatial_SE(X_Branch2)  # (20, 30, Num_Station, Num_Station)
        X_Branch2 = self.Channel_Attention(X_Branch2)  # (20, 30, Num_Station, Num_Station)
        X_Branch2 = self.Depthwise_CNN_1(X_Branch2)  # 深度分离卷积
        X_Branch2 = self.Pointwise_CNN_1(X_Branch2)  # 逐点卷积
        X_Branch2 = self.Depthwise_CNN_2(X_Branch2)  # 深度分离卷积
        X_Branch2 = self.Pointwise_CNN_2(X_Branch2)  # 逐点卷积
        X_Branch2 = X_Branch2.reshape(X_Branch2.size()[0], -1)  # (20, 8192)   flatten
        X_Branch2 = F.relu(self.linear_B2_1(X_Branch2))  # (batch_size, 1024)
        X_Branch2 = F.relu(self.linear_B2_2(X_Branch2))  # (batch_size, 512)

        # Deep Fusion Layer (Feature-wise)
        Deep_F = self.Deep_Fusion(X1=X_Branch1, X2=X_Branch2, X3=Early_F)  # (batch_size, 512)
        Deep_F = F.relu(self.linear_d(Deep_F))     # (batch_size, 64*pre_len)   Decision 1


        # 后融合 Late Fusion (Decision-wise)
        X_Branch1 = F.relu(self.linear_L(X_Branch1))   # (batch_size, 64*pre_len)    Decision 2
        X_Branch2 = F.relu(self.linear_R(X_Branch2))  # (batch_size, 64*pre_len)    Decision 3
        out_put = self.Late_Fusion(X1=X_Branch1, X2=X_Branch2, X3=Deep_F)
        out_put = out_put.reshape(out_put.size()[0], self.station_num, self.pre_len)

        return out_put
