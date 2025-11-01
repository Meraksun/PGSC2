import torch
from typing import Optional, List


def add_self_loop(
        adj: torch.Tensor,
        node_count: torch.Tensor
) -> torch.Tensor:
    """
    为配电网邻接矩阵的有效节点添加自环（对角线置1），填充节点不添加
    论文逻辑适配：增强节点自身特征权重，适配GNN消息传递（🔶1-62、🔶1-73）

    Args:
        adj: 输入邻接矩阵（shape: [B, N, N]，B=Batch，N=最大节点数）
        node_count: 每个场景的真实节点数（shape: [B]，用于确定有效节点范围）

    Returns:
        adj_with_self_loop: 添加自环后的邻接矩阵（shape与adj一致）

    Raises:
        ValueError: adj非3维方阵，或node_count长度与Batch大小不一致
    """
    # 输入参数校验
    if adj.dim() != 3:
        raise ValueError(f"adj需为3维张量[B, N, N]，当前维度：{adj.dim()}")
    if adj.shape[1] != adj.shape[2]:
        raise ValueError(f"adj的节点维度需为方阵，当前形状：{adj.shape[1:]}")
    if len(node_count) != adj.shape[0]:
        raise ValueError(f"node_count长度（{len(node_count)}）与Batch大小（{adj.shape[0]}）不一致")

    adj_with_self_loop = adj.clone()  # 避免修改原矩阵
    batch_size, max_node, _ = adj.shape

    # 遍历每个场景，为有效节点添加自环
    for b in range(batch_size):
        real_count = node_count[b].item()
        # 仅对前real_count个有效节点的对角线置1（自环）
        adj_with_self_loop[b, :real_count, :real_count].diagonal().fill_(1.0)

    return adj_with_self_loop


def check_radial(adj: torch.Tensor, node_count: Optional[int] = None) -> bool:
    """
    判断单个配电网邻接矩阵是否为辐射型（树状无环、连通）
    论文逻辑适配：辐射型网络无闭环，符合配电网拓扑特性（🔶1-119、🔶1-183）

    Args:
        adj: 单个场景的邻接矩阵（shape: [N, N]，无Batch维度）
        node_count: 真实节点数（可选，默认使用所有非孤立节点；若指定，需<=N）

    Returns:
        is_radial: True=辐射型（树状无环），False=非辐射型（有环或不连通）

    Raises:
        ValueError: adj非2维方阵，或node_count超出合理范围
    """
    # 输入参数校验
    if adj.dim() != 2:
        raise ValueError(f"adj需为2维方阵[N, N]（单个场景），当前维度：{adj.dim()}")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj需为方阵，当前形状：{adj.shape}")
    N = adj.shape[0]
    if node_count is not None and (node_count < 2 or node_count > N):
        raise ValueError(f"node_count需在2~{N}范围内，当前输入：{node_count}")

    # 1. 确定有效节点（排除孤立节点，或按node_count截断）
    if node_count is not None:
        valid_nodes = set(range(node_count))
    else:
        # 孤立节点：行和为0的节点（无任何连接）
        row_sums = adj.sum(dim=1)
        valid_nodes = set(i for i in range(N) if row_sums[i] > 1e-6)
    valid_nodes = list(valid_nodes)
    M = len(valid_nodes)
    if M < 2:
        return False  # 至少2个有效节点才可能构成辐射型网络

    # 2. 构建有效节点的邻接表（无向图，避免重复边）
    adj_table: List[List[int]] = [[] for _ in valid_nodes]
    node_to_idx = {node: idx for idx, node in enumerate(valid_nodes)}  # 有效节点→局部索引
    edge_count = 0

    for i in valid_nodes:
        for j in valid_nodes:
            if i < j and adj[i, j] > 1e-6:  # 仅统计上三角非零元素（无向图去重）
                adj_table[node_to_idx[i]].append(node_to_idx[j])
                adj_table[node_to_idx[j]].append(node_to_idx[i])
                edge_count += 1

    # 3. 树的判定条件1：边数 = 节点数 - 1
    if edge_count != M - 1:
        return False

    # 4. 树的判定条件2：所有节点连通（DFS遍历）
    visited = [False] * M
    stack = [0]  # 从第一个有效节点开始遍历
    visited[0] = True
    visited_count = 1

    while stack:
        current = stack.pop()
        for neighbor in adj_table[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                visited_count += 1
                stack.append(neighbor)

    # 所有有效节点均被访问 → 连通
    return visited_count == M

# -------------------------- 函数示例（注释形式，取消注释可运行） --------------------------
# if __name__ == "__main__":
#     # 模拟输入：Batch=2，最大节点数=50，真实节点数=[20,30]
#     batch_size = 2
#     max_node = 50
#     node_count = torch.tensor([20, 30], dtype=torch.int32)
#
#     # 1. add_self_loop 示例
#     # 生成模拟邻接矩阵（辐射型，仅上三角有非零值）
#     adj = torch.zeros(batch_size, max_node, max_node)
#     for b in range(batch_size):
#         real_count = node_count[b].item()
#         for i in range(1, real_count):
#             parent = torch.randint(0, i, (1,)).item()  # 父节点<当前节点（无环）
#             adj[b, i, parent] = 1.0 / (torch.rand(1).item() + 0.1)  # 阻抗模值倒数
#             adj[b, parent, i] = adj[b, i, parent]  # 无向图对称
#     # 添加自环
#     adj_with_loop = add_self_loop(adj, node_count)
#     print("=== add_self_loop 示例结果 ===")
#     print(f"输入adj形状：{adj.shape}")
#     print(f"添加自环后，第1个场景有效节点（0~19）的对角线值：{adj_with_loop[0, :5, :5].diagonal()}（预期均为1.0）")
#     print(f"添加自环后，第1个场景填充节点（20~49）的对角线值：{adj_with_loop[0, 20, 20]}（预期为0.0）")
#
#     # 2. check_radial 示例
#     # 生成辐射型邻接矩阵（20节点，树状无环）
#     radial_adj = torch.zeros(20, 20)
#     for i in range(1, 20):
#         parent = torch.randint(0, i, (1,)).item()
#         radial_adj[i, parent] = 1.0
#         radial_adj[parent, i] = 1.0
#     # 生成非辐射型邻接矩阵（添加额外边造成环）
#     non_radial_adj = radial_adj.clone()
#     non_radial_adj[2, 5] = 1.0
#     non_radial_adj[5, 2] = 1.0
#
#     is_radial1 = check_radial(radial_adj, node_count=20)
#     is_radial2 = check_radial(non_radial_adj, node_count=20)
#     print("\n=== check_radial 示例结果 ===")
#     print(f"辐射型邻接矩阵判定结果：{is_radial1}（预期True）")
#     print(f"非辐射型邻接矩阵（有环）判定结果：{is_radial2}（预期False）")