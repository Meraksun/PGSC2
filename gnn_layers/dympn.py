import torch
import torch.nn as nn
import torch.nn.functional as F


class DyMPNLayer(nn.Module):
    """
    实现论文中的Dynamic Message Passing Network（DyMPN）层
    适配20-50节点辐射型配电网：简化消息传递步数为1~3步，聚焦局部拓扑关系捕捉
    核心逻辑：随机消息传递步数 + 基于邻接矩阵的特征聚合 + 无效填充节点截断
    """

    def __init__(self, d_in: int = 4, d_model: int = 64):
        """
        初始化DyMPN层

        Args:
            d_in: 输入节点特征维度（节点矩阵为4列，故默认4，对应P_load、Q_load、V、θ的标幺值）
            d_model: 嵌入后特征维度（默认64，适配20-50节点的小规模配电网，平衡参数规模与表达能力）
        """
        super().__init__()

        # 1. 线性嵌入层：将原始4维节点特征映射到高维空间d_model（论文中用于提升表征能力）
        self.embed = nn.Linear(in_features=d_in, out_features=d_model)

        # 2. 消息传递线性层：对聚合后的邻居特征进行线性变换（论文中μ函数的简化实现）
        self.mp_linear = nn.Linear(in_features=d_model, out_features=d_model)

        # 3. 激活函数：ReLU（论文中用于引入非线性，增强特征表达）
        self.relu = nn.ReLU()

    def forward(
            self,
            node_feat: torch.Tensor,
            adj: torch.Tensor,
            node_count: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：执行动态消息传递，输出处理后的节点特征

        Args:
            node_feat: 带掩码的输入节点特征，形状为(B, N, d_in)
                       B=Batch大小，N=Batch内最大节点数，d_in=输入特征维度（默认4）
            adj: 拓扑邻接矩阵，形状为(B, N, N)
                 辐射型配电网无闭环，矩阵为稀疏树状结构（非连接位置为0，连接位置为线路阻抗模值倒数）
            node_count: 每个场景的真实节点数，形状为(B, )
                       用于截断无效填充节点（避免填充的0值参与消息传递）

        Returns:
            h: 动态消息传递后的节点特征，形状为(B, N, d_model)
        """
        # -------------------------- 步骤1：节点特征嵌入 --------------------------
        # 将原始d_in维特征映射到d_model维，为后续消息传递提供更丰富的表征
        # 形状变化：(B, N, d_in) → (B, N, d_model)
        h = self.embed(node_feat)

        # -------------------------- 步骤2：随机采样消息传递步数k --------------------------
        # 论文中k∈[1,5]，此处简化为k∈[1,3]（小节点规模无需更多步数，避免冗余计算）
        # 同一Batch内所有样本使用相同k，保证Batch维度一致性
        k = torch.randint(low=1, high=4, size=(1,), device=node_feat.device).item()  # 1/2/3中随机取一个

        # -------------------------- 步骤3：k次消息传递迭代 --------------------------
        for _ in range(k):
            # 核心消息传递公式：h = ReLU( adj @ h @ W + b )
            # 1. 邻接矩阵与节点特征相乘：基于拓扑聚合邻居特征（论文中"消息聚合"步骤）
            # 形状变化：(B, N, N) @ (B, N, d_model) → (B, N, d_model)
            aggregated = torch.bmm(adj, h)  # 批次矩阵乘法，适配Batch维度

            # 2. 线性变换：对聚合后的特征进行参数化映射（论文中μ函数）
            # 形状变化：(B, N, d_model) → (B, N, d_model)
            aggregated = self.mp_linear(aggregated)

            # 3. 非线性激活：引入非线性，增强特征表达能力
            h = self.relu(aggregated)

            # -------------------------- 关键处理：截断无效填充节点 --------------------------
            # 遍历每个样本，将超出"真实节点数"的填充节点特征置0（避免填充值干扰计算）
            for batch_idx in range(h.shape[0]):
                real_node_num = node_count[batch_idx].item()  # 当前样本的真实节点数
                # 填充节点索引：从real_node_num到N-1，将这些节点的特征置0
                h[batch_idx, real_node_num:, :] = 0.0

        # 返回动态消息传递后的节点特征（已捕捉局部拓扑关系）
        return h

# -------------------------- 单元测试示例（注释形式，取消注释可运行） --------------------------
# if __name__ == "__main__":
#     # 1. 配置测试参数（模拟20-50节点辐射型配电网场景）
#     batch_size = 2  # Batch大小
#     max_node = 50   # Batch内最大节点数（适配50节点上限）
#     d_in = 4        # 输入特征维度（P_load、Q_load、V、θ）
#     d_model = 64    # 嵌入维度
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 2. 生成模拟输入数据
#     # 模拟带掩码的节点特征（值范围0~1，符合标幺值特性）
#     node_feat = torch.rand(batch_size, max_node, d_in, device=device)
#     # 模拟辐射型邻接矩阵（稀疏树状结构：随机生成10~20条边，避免闭环）
#     adj = torch.zeros(batch_size, max_node, max_node, device=device)
#     for b in range(batch_size):
#         real_node = torch.randint(20, 51, size=(1,)).item()  # 每个样本的真实节点数（20~50）
#         # 生成树状边（real_node-1条边，无闭环）
#         for i in range(1, real_node):
#             parent = torch.randint(0, i, size=(1,)).item()  # 父节点索引（小于当前节点，避免闭环）
#             impedance模值 = torch.rand(1).item() + 0.1  # 线路阻抗模值（0.1~1.1）
#             adj[b, i, parent] = 1.0 / impedance模值  # 邻接矩阵值=阻抗模值倒数
#             adj[b, parent, i] = 1.0 / impedance模值  # 邻接矩阵对称（无向图）
#     # 模拟真实节点数（20~50）
#     node_count = torch.randint(20, 51, size=(batch_size,), device=device, dtype=torch.int32)
#
#     # 3. 初始化DyMPN层并执行前向传播
#     dympn_layer = DyMPNLayer(d_in=d_in, d_model=d_model).to(device)
#     output = dympn_layer(node_feat=node_feat, adj=adj, node_count=node_count)
#
#     # 4. 打印测试结果（验证输出形状与逻辑正确性）
#     print("=" * 50)
#     print("DyMPN层单元测试结果")
#     print("=" * 50)
#     print(f"输入node_feat形状: {node_feat.shape}")  # 预期: (2, 50, 4)
#     print(f"输入adj形状: {adj.shape}")              # 预期: (2, 50, 50)
#     print(f"输入node_count: {node_count.tolist()}") # 预期: [20~50间的整数, 20~50间的整数]
#     print(f"输出特征形状: {output.shape}")          # 预期: (2, 50, 64)
#     print(f"填充节点特征是否为0: {torch.all(output[0, node_count[0]:, :] == 0.0).item()}")  # 预期: True
#     print("=" * 50)