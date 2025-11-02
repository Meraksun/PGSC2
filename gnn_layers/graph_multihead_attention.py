import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMultiHeadAttention(nn.Module):
    """
    实现论文中的Graphical Multi-head Attention（图多头注意力）
    核心逻辑：基于DyMPN输出的局部特征，通过多头注意力捕捉全局拓扑依赖
    适配20-50节点辐射型配电网：头数默认4，参数规模轻量化，避免过拟合
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        """
        初始化图多头注意力层

        Args:
            d_model: 输入/输出特征维度（需与DyMPN输出维度一致，默认64）
            n_heads: 注意力头数（默认4，需满足d_model % n_heads == 0，适配小节点规模）
            dropout: 注意力权重的dropout概率（默认0.1，缓解过拟合）

        Raises:
            ValueError: 若d_model无法被n_heads整除，抛出参数不合法错误
        """
        super().__init__()

        # 校验参数合法性：d_model必须能被n_heads整除（确保每个头的维度一致）
        if d_model % n_heads != 0:
            raise ValueError(
                f"输入维度d_model={d_model}必须能被头数n_heads={n_heads}整除，"
                f"当前d_model//n_heads={d_model // n_heads}，存在余数"
            )

        # 核心参数定义
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个注意力头的特征维度（默认64//4=16）

        # 1. 定义Q、K、V线性变换层（论文中用于将输入特征映射到注意力空间）
        # 输入输出维度均为d_model：拆分后每个头处理d_k维度，总维度保持d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # 2. 定义输出线性变换层（论文中用于合并多头注意力结果）
        self.out_linear = nn.Linear(d_model, d_model)

        # 3. 定义dropout层（论文中用于正则化，降低过拟合风险）
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            h: torch.Tensor,
            mask: torch.Tensor,
            node_count: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：通过多头注意力捕捉全局拓扑依赖，输出增强后的节点特征

        Args:
            h: DyMPN输出的局部特征，形状为(B, N, d_model)
               B=Batch大小，N=Batch内最大节点数，d_model=特征维度
            mask: 节点电压掩码矩阵，形状为(B, N, 2)
               仅标记节点第2、3列（电压幅值、相角）的缺失情况（1=缺失，0=正常）
            node_count: 每个场景的真实节点数，形状为(B, )
               用于截断无效填充节点，避免其干扰全局注意力计算

        Returns:
            out: 全局注意力增强后的节点特征，形状为(B, N, d_model)
        """
        # -------------------------- 步骤1：生成Q、K、V特征 --------------------------
        # 通过线性层将DyMPN的局部特征映射到注意力空间（Q=查询，K=键，V=值）
        # 形状变化：(B, N, d_model) → (B, N, d_model)
        q = self.q_linear(h)
        k = self.k_linear(h)
        v = self.v_linear(h)

        # -------------------------- 步骤2：多头特征拆分 --------------------------
        # 将Q、K、V按头数拆分，使每个头独立捕捉不同维度的全局依赖
        # 1. 形状转换：(B, N, d_model) → (B, N, n_heads, d_k)
        #    （将d_model维度拆分为"头数×单头维度"）
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_k)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_k)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_k)

        # 2. 转置调整维度顺序：适配注意力计算的批量矩阵乘法
        #    Q/V形状：(B, n_heads, N, d_k)（便于按头计算注意力）
        #    K形状：(B, n_heads, d_k, N)（需转置为"头数×单头维度×节点数"，匹配Q的乘法维度）
        q = q.transpose(1, 2)  # (B, n_heads, N, d_k)
        k = k.transpose(1, 2).transpose(2, 3)  # (B, n_heads, d_k, N)
        v = v.transpose(1, 2)  # (B, n_heads, N, d_k)

        # -------------------------- 步骤3：计算注意力分数 --------------------------
        # 核心公式：Attention(Q,K,V) = softmax( QK^T / sqrt(d_k) )V
        # 1. 批量矩阵乘法计算QK^T（捕捉节点间的全局关联度）
        #    形状变化：(B, n_heads, N, d_k) × (B, n_heads, d_k, N) → (B, n_heads, N, N)
        score = torch.matmul(q, k)

        # 2. 缩放因子：除以sqrt(d_k)，缓解分数过大导致的softmax梯度消失（论文标准操作）
        scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=score.device))
        score = score / scale

        # -------------------------- 步骤4：注意力掩码处理 --------------------------
        # 目标：屏蔽电压数据缺失的节点和无效填充节点，避免注意力聚焦于无效信息
        batch_size, n_heads, max_node, _ = score.shape

        # 1. 处理电压掩码（mask：B×N×2 → 转换为节点级掩码B×N）
        #    若节点的任意一列电压数据缺失（mask=1），则标记该节点为"需屏蔽"
        node_mask = mask.any(dim=-1)  # (B, N)：1=需屏蔽节点，0=正常节点
        #    扩展为注意力掩码形状（B×1×N×N）：每个头共享同一掩码
        #    逻辑：若query节点i或key节点j需屏蔽，则score[i][j]设为-1e9（softmax后权重≈0）
        #    使用广播：node_mask.unsqueeze(1).unsqueeze(-1) -> (B, 1, N, 1)
        #             node_mask.unsqueeze(1).unsqueeze(2) -> (B, 1, 1, N)
        #             | 操作会广播到 (B, 1, N, N)
        node_mask_query = node_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        node_mask_key = node_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        attn_mask = (node_mask_query | node_mask_key).bool()  # (B, 1, N, N)，转换为布尔类型
        #    掩码值替换：需屏蔽位置设为-1e9（softmax后趋近于0），正常位置设为0（不影响）
        score = score.masked_fill(attn_mask, -1e9)

        # 2. 处理无效填充节点（通过node_count截断）
        #    对每个样本，将"真实节点数之后的填充节点"对应的注意力分数设为-1e9
        for b in range(batch_size):
            real_node = node_count[b].item()  # 当前样本的真实节点数
            if real_node < max_node:
                # 屏蔽填充节点的query和key对应的分数
                score[b, :, real_node:, :] = -1e9  # 填充节点作为query
                score[b, :, :, real_node:] = -1e9  # 填充节点作为key

        # -------------------------- 步骤5：Softmax与Dropout --------------------------
        # 1. Softmax：将注意力分数转换为权重（每行和为1，体现节点间的关注程度）
        score = F.softmax(score, dim=-1)  # 对key维度（最后一维）做softmax
        # 2. Dropout：随机失活部分注意力权重，降低过拟合风险（论文中正则化手段）
        score = self.dropout(score)

        # -------------------------- 步骤6：计算注意力输出 --------------------------
        # 用注意力权重加权求和V，得到每个头的全局特征
        # 形状变化：(B, n_heads, N, N) × (B, n_heads, N, d_k) → (B, n_heads, N, d_k)
        attn_out = torch.matmul(score, v)

        # -------------------------- 步骤7：多头特征合并 --------------------------
        # 将多个头的特征重新合并为d_model维度（逆操作步骤2）
        # 1. 转置调整维度：(B, n_heads, N, d_k) → (B, N, n_heads, d_k)
        attn_out = attn_out.transpose(1, 2)
        # 2. 合并头维度：(B, N, n_heads, d_k) → (B, N, d_model)（n_heads×d_k=d_model）
        attn_out = attn_out.contiguous().view(attn_out.size(0), attn_out.size(1), self.d_model)
        # 3. 输出线性变换：融合多头特征，保证输出维度与输入h一致（便于残差连接）
        attn_out = self.out_linear(attn_out)

        # -------------------------- 步骤8：残差连接 --------------------------
        # 论文中核心设计：将全局注意力特征与DyMPN的局部特征相加，保留局部信息的同时补充全局依赖
        # 形状变化：(B, N, d_model) + (B, N, d_model) → (B, N, d_model)
        out = attn_out + h

        return out

# -------------------------- 示例：组合DyMPN与GraphMultiHeadAttention --------------------------
# if __name__ == "__main__":
#     """
#     验证流程：模拟20-50节点辐射型配电网数据 → DyMPN提取局部特征 → GraphMultiHeadAttention捕捉全局依赖
#     预期输出：最终特征形状与输入DyMPN的特征形状一致（B, N, d_model）
#     """
#     import sys
#     # 导入同目录下的DyMPNLayer（需确保dympn.py与当前文件在同一文件夹）
#     sys.path.append(".")
#     from dympn import DyMPNLayer
#
#     # 1. 配置模拟参数（适配20-50节点配电网）
#     batch_size = 2       # Batch大小
#     max_node = 50        # Batch内最大节点数（覆盖50节点上限）
#     d_in = 4             # DyMPN输入特征维度（节点矩阵4列：P_load、Q_load、V、θ）
#     d_model = 64         # 特征嵌入维度（与DyMPN、注意力层一致）
#     n_heads = 4          # 注意力头数（64//4=16，单头维度合理）
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 2. 生成模拟输入数据
#     # 2.1 模拟DyMPN输入：带掩码的节点特征（符合标幺值范围0~1）
#     node_feat = torch.rand(batch_size, max_node, d_in, device=device)  # (2,50,4)
#     # 2.2 模拟辐射型邻接矩阵（稀疏树状结构，无闭环）
#     adj = torch.zeros(batch_size, max_node, max_node, device=device)  # (2,50,50)
#     for b in range(batch_size):
#         real_node = torch.randint(20, 51, size=(1,)).item()  # 每个样本的真实节点数（20~50）
#         # 生成树状边（real_node-1条边，确保无闭环）
#         for i in range(1, real_node):
#             parent = torch.randint(0, i, size=(1,)).item()  # 父节点索引小于当前节点（避免闭环）
#             impedance模值 = torch.rand(1).item() + 0.1  # 线路阻抗模值（0.1~1.1）
#             adj[b, i, parent] = 1.0 / impedance模值  # 邻接矩阵值=阻抗模值倒数（用户数据集定义）
#             adj[b, parent, i] = 1.0 / impedance模值  # 无向图，邻接矩阵对称
#     # 2.3 模拟电压掩码（B,N,2）：随机掩码30%的非平衡节点电压数据
#     mask = torch.zeros(batch_size, max_node, 2, device=device)  # (2,50,2)
#     for b in range(batch_size):
#         real_node = torch.randint(20, 51, size=(1,)).item()
#         non_balance_idx = torch.arange(1, real_node)  # 非平衡节点（1~real_node-1，0为平衡节点）
#         # 对非平衡节点的电压列随机掩码（30%概率）
#         mask_prob = torch.full((len(non_balance_idx), 2), 0.3, device=device)
#         mask[b, non_balance_idx] = torch.bernoulli(mask_prob)
#     # 2.4 模拟真实节点数（20~50）
#     node_count = torch.randint(20, 51, size=(batch_size,), device=device, dtype=torch.int32)  # (2,)
#
#     # 3. 初始化并串联模型
#     # 3.1 DyMPN层：提取局部拓扑特征
#     dympn_layer = DyMPNLayer(d_in=d_in, d_model=d_model).to(device)
#     # 3.2 GraphMultiHeadAttention层：捕捉全局拓扑依赖
#     gatt_layer = GraphMultiHeadAttention(d_model=d_model, n_heads=n_heads).to(device)
#
#     # 4. 执行前向传播
#     # 4.1 DyMPN输出局部特征
#     local_feat = dympn_layer(node_feat=node_feat, adj=adj, node_count=node_count)
#     print(f"DyMPN输出局部特征形状: {local_feat.shape}")  # 预期：(2, 50, 64)
#     # 4.2 注意力层输出全局增强特征
#     global_feat = gatt_layer(h=local_feat, mask=mask, node_count=node_count)
#     print(f"GraphMultiHeadAttention输出全局特征形状: {global_feat.shape}")  # 预期：(2, 50, 64)
#
#     # 5. 验证关键逻辑正确性
#     # 5.1 残差连接验证：输出特征与输入特征维度一致
#     assert global_feat.shape == local_feat.shape, "残差连接维度不匹配！"
#     # 5.2 填充节点特征验证：填充节点特征应为0（无效数据不参与计算）
#     for b in range(batch_size):
#         real_node = node_count[b].item()
#         fill_feat = global_feat[b, real_node:, :]
#         assert torch.allclose(fill_feat, torch.zeros_like(fill_feat), atol=1e-6), \
#             f"第{b}个样本的填充节点特征未置0！"
#     print("\n✅ 模型串联验证通过：维度匹配+填充节点处理正确")