import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """
    簡化版圖卷積：D^{-1} A X W
    
    加入度數正規化（D^{-1}A）防止鄰集後訊號清長，
    改善深層模型中的梯度流。
    """
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x:   [batch, num_nodes, features]
        # adj: [num_nodes, num_nodes] – 已由 dataset 做 D^{-1/2} A D^{-1/2} 正規化
        #      此處再加 D^{-1} 正規化减少模型內訊號振盪
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [N, 1]
        adj_norm = adj / deg                                  # D^{-1} A
        out = torch.matmul(adj_norm, x)  # Aggregate from neighbors
        out = self.fc(out)               # Feature transformation
        return F.relu(out)

class STGCN_Prototype(nn.Module):
    """
    簡化版 STGCN：Temporal Conv → Spatial Graph Conv → Temporal Conv → FC

    改善項目：
      - BatchNorm2d 幫助訓練穩定、加快收斂
      - dropout 降至 0.1 防止 mock data 量少時的 underfitting
      - hidden_channels 提升到 64 增加容量
    """
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, time_steps, dropout=0.1):
        super(STGCN_Prototype, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)

        # Temporal Conv 1
        self.t_conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels,
            kernel_size=(1, 3), padding=(0, 1)
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)  # 穩定訓練

        # Spatial Graph Conv (Simplified)
        self.g_conv = GraphConv(hidden_channels, hidden_channels)

        # Temporal Conv 2
        self.t_conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(1, 3), padding=(0, 1)
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)  # 穩定訓練

        # Fully Connected for prediction
        self.fc = nn.Linear(hidden_channels * time_steps, out_channels)

    def forward(self, x, adj):
        # x shape: [batch, in_channels, num_nodes, time_steps]

        # ── Temporal Conv 1 ───────────────────────────────────
        out = self.t_conv1(x)    # [batch, hidden, num_nodes, time]
        out = self.bn1(out)      # BatchNorm 穩定訓練
        out = F.relu(out)
        out = self.dropout(out)

        # ── Spatial Graph Conv ────────────────────────────────
        batch, hidden, nodes, time = out.shape
        out = out.permute(0, 3, 2, 1).contiguous()  # [batch, time, nodes, hidden]
        g_out_list = []
        for t in range(time):
            g_out_list.append(self.g_conv(out[:, t, :, :], adj))
        out = torch.stack(g_out_list, dim=1)         # [batch, time, nodes, hidden]
        out = out.permute(0, 3, 2, 1).contiguous()  # [batch, hidden, nodes, time]
        out = self.dropout(out)

        # ── Temporal Conv 2 ───────────────────────────────────
        out = self.t_conv2(out)  # [batch, hidden, nodes, time]
        out = self.bn2(out)      # BatchNorm 穩定訓練
        out = F.relu(out)
        out = self.dropout(out)

        # ── Flatten 與預測 ─────────────────────────────────────
        out = out.permute(0, 2, 1, 3).contiguous()  # [batch, nodes, hidden, time]
        out = out.view(batch, nodes, -1)             # [batch, nodes, hidden * time]
        out = self.fc(out)                           # [batch, nodes, out_channels]
        return out
