import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class PhysicsInformedLoss(nn.Module):
    """
    适配20-50节点辐射型配电网的物理知情损失函数
    核心逻辑：预测误差损失（MSE） + λ×物理约束损失（功率平衡 + 线路潮流一致性）
    严格遵循论文"物理知情正则化"思想，适配标幺值数据与小节点规模场景（🔶1-36、🔶1-82）
    """

    def __init__(self, lambda_: float = 0.5, V_base: float = 12.66, Z_base: float = 16.03):
        """
        初始化物理知情损失函数

        Args:
            lambda_: 物理约束损失的权重系数（默认0.5，平衡预测误差与物理约束）
            V_base: 电压基准值（kV，默认12.66，标幺值下实际生效为1）
            Z_base: 阻抗基准值（Ω，默认16.03，标幺值下实际生效为1）
        """
        super().__init__()
        self.lambda_ = lambda_
        # 标幺值转换系数（用户数据已标幺化，此处保留参数便于扩展）
        self.V_base = V_base
        self.Z_base = Z_base

    def _build_line_node_mapping(self, adj: torch.Tensor, node_count: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """
        辅助函数：为每个场景构建"线路索引→连接节点对(i,j)"的映射（i<j，避免重复）
        基于邻接矩阵的非零元素确定线路，适配辐射型网络无闭环特性

        Args:
            adj: 邻接矩阵（B, N, N），N=Batch内最大节点数
            node_count: 每个场景的真实节点数（B, ）

        Returns:
            line_node_mapping: 线路-节点对映射列表，每个场景对应[(i1,j1), (i2,j2), ...]
        """
        batch_size, max_node, _ = adj.shape
        line_node_mapping = []

        for b in range(batch_size):
            real_node = node_count[b].item()
            adj_b = adj[b, :real_node, :real_node]  # 截断填充节点，仅保留真实节点的邻接关系
            line_pairs = []

            # 遍历邻接矩阵上三角（i<j），非零元素即为线路（辐射型网络无重复边）
            for i in range(real_node):
                for j in range(i + 1, real_node):
                    if adj_b[i, j] != 0:  # 存在线路连接
                        line_pairs.append((i, j))

            line_node_mapping.append(line_pairs)
        return line_node_mapping

    def calc_power_balance(
            self,
            pred_node: torch.Tensor,
            pred_line: List[torch.Tensor],
            adj: torch.Tensor,
            node_count: torch.Tensor
    ) -> torch.Tensor:
        """
        计算节点功率平衡误差：P_inj - 线路有功潮流和，Q_inj - 线路无功潮流和
        平衡节点（1号，tensor索引0）与填充节点误差置0（🔶1-82、🔶1-87）

        Args:
            pred_node: 预测节点特征（B, N, 4），列0=P_load, 1=Q_load, 2=V, 3=θ（均为标幺值）
            pred_line: 预测线路潮流列表（B个元素，每个元素为(b, 4)，列2=P, 3=Q）
            adj: 邻接矩阵（B, N, N）
            node_count: 每个场景的真实节点数（B, ）

        Returns:
            power_balance_err: 功率平衡误差（B, N, 2），列0=有功误差，1=无功误差
        """
        batch_size, max_node, _ = pred_node.shape
        power_balance_err = torch.zeros((batch_size, max_node, 2), device=pred_node.device)

        # 1. 构建线路-节点对映射（确定每条线路对应的节点i,j）
        line_node_mapping = self._build_line_node_mapping(adj, node_count)

        for b in range(batch_size):
            real_node = node_count[b].item()
            line_pairs = line_node_mapping[b]  # 当前场景的线路-节点对
            pred_line_b = pred_line[b]  # 当前场景的预测线路潮流（b_line, 4）
            pred_node_b = pred_node[b, :real_node, :]  # 截断填充节点

            # 2. 计算每个节点的注入功率（P_inj, Q_inj）
            # 非平衡节点（2~a号，tensor索引1~real_node-1）：P_inj = -P_load，Q_inj = -Q_load
            # 平衡节点（1号，索引0）：注入功率=全网负荷和，无需计算（误差置0）
            P_inj = torch.zeros(real_node, device=pred_node.device)
            Q_inj = torch.zeros(real_node, device=pred_node.device)
            non_balance_idx = torch.arange(1, real_node, device=pred_node.device)
            P_inj[non_balance_idx] = -pred_node_b[non_balance_idx, 0]  # 列0=P_load
            Q_inj[non_balance_idx] = -pred_node_b[non_balance_idx, 1]  # 列1=Q_load

            # 3. 计算每个节点的线路潮流和（有功P_sum，无功Q_sum）
            P_sum = torch.zeros(real_node, device=pred_node.device)
            Q_sum = torch.zeros(real_node, device=pred_node.device)

            for line_idx, (i, j) in enumerate(line_pairs):
                # 线路潮流：列2=P_ij，列3=Q_ij（i<j方向）
                P_ij = pred_line_b[line_idx, 2]
                Q_ij = pred_line_b[line_idx, 3]

                # 节点i：潮流流入（+P_ij, +Q_ij）；节点j：潮流流出（-P_ij, -Q_ij）
                P_sum[i] += P_ij
                Q_sum[i] += Q_ij
                P_sum[j] -= P_ij
                Q_sum[j] -= Q_ij

            # 4. 计算功率平衡误差（仅非平衡节点有效，平衡节点误差置0）
            P_err = P_inj - P_sum
            Q_err = Q_inj - Q_sum
            P_err[0] = 0.0  # 平衡节点（索引0）误差置0
            Q_err[0] = 0.0

            # 5. 赋值到总误差矩阵（填充节点已默认置0）
            power_balance_err[b, :real_node, 0] = P_err
            power_balance_err[b, :real_node, 1] = Q_err

        return power_balance_err

    def calc_line_flow_constraint(
            self,
            pred_node: torch.Tensor,
            line_param: List[torch.Tensor],
            pred_line: List[torch.Tensor],
            adj: torch.Tensor,
            node_count: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        计算线路潮流约束误差：基于简化公式 V_i² - V_j² ≈ 2(RP + XQ)（标幺值下）
        误差 = 理论值（2(RP+XQ)） - 实际值（V_i² - V_j²）（🔶1-88、🔶1-93）

        Args:
            pred_node: 预测节点特征（B, N, 4），列2=V（电压幅值标幺值）
            line_param: 线路参数列表（B个元素，每个元素为(b_line, 4)，列0=R, 1=X）
            pred_line: 预测线路潮流列表（B个元素，每个元素为(b_line, 4)，列2=P, 3=Q）
            adj: 邻接矩阵（B, N, N）
            node_count: 每个场景的真实节点数（B, ）

        Returns:
            line_flow_err_list: 线路潮流误差列表（B个元素，每个元素为(b_line, 2)，列0=P误差，1=Q误差）
        """
        batch_size = pred_node.shape[0]
        line_node_mapping = self._build_line_node_mapping(adj, node_count)
        line_flow_err_list = []

        for b in range(batch_size):
            real_node = node_count[b].item()
            line_pairs = line_node_mapping[b]
            line_param_b = line_param[b]  # 当前场景线路参数（R, X）
            pred_line_b = pred_line[b]  # 当前场景预测潮流（P, Q）
            pred_node_b = pred_node[b, :real_node, :]  # 截断填充节点
            b_line = len(line_pairs)
            line_err = torch.zeros((b_line, 2), device=pred_node.device)

            for line_idx, (i, j) in enumerate(line_pairs):
                # 1. 提取线路参数与预测潮流（标幺值）
                R = line_param_b[line_idx, 0]  # 列0=R
                X = line_param_b[line_idx, 1]  # 列1=X
                P = pred_line_b[line_idx, 2]  # 列2=P
                Q = pred_line_b[line_idx, 3]  # 列3=Q

                # 2. 提取节点i,j的电压幅值（标幺值）
                V_i = pred_node_b[i, 2]
                V_j = pred_node_b[j, 2]

                # 3. 计算理论值与实际值
                theoretical = 2 * (R * P + X * Q)  # 2(RP + XQ)（公式简化版）
                actual = torch.pow(V_i, 2) - torch.pow(V_j, 2)  # V_i² - V_j²

                # 4. 潮流约束误差（P、Q共用同一误差逻辑，因公式耦合）
                line_err[line_idx, 0] = theoretical - actual  # 有功相关误差
                line_err[line_idx, 1] = theoretical - actual  # 无功相关误差

            line_flow_err_list.append(line_err)
        return line_flow_err_list

    def forward(
            self,
            pred_node: torch.Tensor,
            gt_node: torch.Tensor,
            pred_line: List[torch.Tensor],
            gt_line: List[torch.Tensor],
            adj: torch.Tensor,
            line_param: List[torch.Tensor],
            node_count: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算总损失 = 预测误差损失 + λ×物理约束损失（🔶1-36、🔶1-82）

        Args:
            pred_node: 预测节点特征（B, N, 4）
            gt_node: 真实节点特征（B, N, 4）
            pred_line: 预测线路潮流列表（B个元素，每个元素为(b_line, 4)）
            gt_line: 真实线路潮流列表（B个元素，每个元素为(b_line, 4)）
            adj: 邻接矩阵（B, N, N）
            line_param: 线路参数列表（B个元素，每个元素为(b_line, 4)）
            node_count: 每个场景的真实节点数（B, ）

        Returns:
            total_loss: 总损失
            pred_loss: 预测误差损失（MSE）
            physics_loss: 物理约束损失
        """
        batch_size, max_node, _ = pred_node.shape

        # -------------------------- 步骤1：计算预测误差损失（MSE） --------------------------
        # 节点特征MSE（仅计算真实节点部分，屏蔽填充节点）
        node_mse = 0.0
        for b in range(batch_size):
            real_node = node_count[b].item()
            node_mse += F.mse_loss(pred_node[b, :real_node, :], gt_node[b, :real_node, :])
        node_mse /= batch_size  # 平均到每个场景

        # 线路潮流MSE（每条线路独立计算）
        line_mse = 0.0
        total_line = 0
        for b in range(batch_size):
            pred_line_b = pred_line[b]
            gt_line_b = gt_line[b]
            line_mse += F.mse_loss(pred_line_b, gt_line_b)
            total_line += 1
        line_mse /= total_line if total_line > 0 else 1  # 平均到每个场景

        # 平均预测损失
        pred_loss = (node_mse + line_mse) / 2

        # -------------------------- 步骤2：计算物理约束损失 --------------------------
        # 2.1 节点功率平衡误差
        power_balance_err = self.calc_power_balance(pred_node, pred_line, adj, node_count)
        # 屏蔽填充节点的误差（仅保留真实节点部分）
        power_err = 0.0
        for b in range(batch_size):
            real_node = node_count[b].item()
            power_err += torch.mean(torch.square(power_balance_err[b, :real_node, :]))
        power_err /= batch_size

        # 2.2 线路潮流约束误差
        line_flow_err_list = self.calc_line_flow_constraint(pred_node, line_param, pred_line, adj, node_count)
        line_err = 0.0
        total_line = 0
        for b in range(batch_size):
            line_err_b = line_flow_err_list[b]
            line_err += torch.mean(torch.square(line_err_b))
            total_line += 1
        line_err /= total_line if total_line > 0 else 1

        # 平均物理约束损失
        physics_loss = (power_err + line_err) / 2

        # -------------------------- 步骤3：计算总损失 --------------------------
        total_loss = pred_loss + self.lambda_ * physics_loss

        return total_loss, pred_loss, physics_loss

# -------------------------- 示例：验证损失函数计算逻辑 --------------------------
# if __name__ == "__main__":
#     """
#     模拟20-50节点辐射型配电网数据，验证损失函数计算流程
#     预期输出：总损失、预测损失、物理损失的具体数值，且逻辑符合物理约束
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     batch_size = 2  # 模拟2个场景
#     max_node = 50   # Batch内最大节点数
#     d_node = 4      # 节点特征维度
#
#     # 1. 生成模拟输入数据
#     def generate_sim_data(batch_size, max_node, device):
#         data = {
#             "pred_node": [], "gt_node": [], "pred_line": [], "gt_line": [],
#             "adj": [], "line_param": [], "node_count": []
#         }
#
#         for b in range(batch_size):
#             # 随机生成真实节点数（20~50）
#             real_node = torch.randint(20, 51, size=(1,), device=device).item()
#             # 随机生成线路数（辐射型网络：line_num = real_node - 1）
#             line_num = real_node - 1
#
#             # 1.1 节点特征（标幺值：P_load/Q_load∈[0.0045,0.05], V∈[0.8,1.2], θ∈[-0.1,0.1]）
#             pred_node_b = torch.rand(real_node, d_node, device=device)
#             pred_node_b[:, 0] = pred_node_b[:, 0] * 0.0455 + 0.0045  # P_load: 0.0045~0.05
#             pred_node_b[:, 1] = pred_node_b[:, 1] * 0.28 + 0.02       # Q_load: 0.02~0.3
#             pred_node_b[:, 2] = pred_node_b[:, 2] * 0.4 + 0.8        # V: 0.8~1.2
#             pred_node_b[:, 3] = pred_node_b[:, 3] * 0.2 - 0.1        # θ: -0.1~0.1
#             # 真实节点特征（在预测值基础上加小噪声）
#             gt_node_b = pred_node_b + torch.randn_like(pred_node_b) * 0.01
#             # 填充到max_node维度
#             pred_node_pad = torch.zeros(max_node, d_node, device=device)
#             pred_node_pad[:real_node, :] = pred_node_b
#             gt_node_pad = torch.zeros(max_node, d_node, device=device)
#             gt_node_pad[:real_node, :] = gt_node_b
#             data["pred_node"].append(pred_node_pad)
#             data["gt_node"].append(gt_node_pad)
#
#             # 1.2 线路数据（R∈[0.005,0.1], X∈[0.0025,0.1125], P/Q∈[-0.05,0.05]）
#             line_param_b = torch.rand(line_num, 4, device=device)
#             line_param_b[:, 0] = line_param_b[:, 0] * 0.095 + 0.005  # R: 0.005~0.1（标幺值）
#             line_param_b[:, 1] = line_param_b[:, 1] * 0.11 + 0.0025  # X: 0.0025~0.1125
#             # 预测线路潮流（在真实值基础上加噪声）
#             gt_line_b = torch.randn(line_num, 4, device=device) * 0.01
#             gt_line_b[:, 2:4] = gt_line_b[:, 2:4].clamp(-0.05, 0.05)  # P/Q: -0.05~0.05
#             pred_line_b = gt_line_b + torch.randn_like(gt_line_b) * 0.005
#             data["pred_line"].append(pred_line_b)
#             data["gt_line"].append(gt_line_b)
#             data["line_param"].append(line_param_b)
#
#             # 1.3 邻接矩阵（辐射型树状结构，i<j有边，值=1/阻抗模值）
#             adj_b = torch.zeros(max_node, max_node, device=device)
#             for i in range(1, real_node):
#                 parent = torch.randint(0, i, size=(1,), device=device).item()
#                 impedance = torch.sqrt(torch.pow(line_param_b[i-1, 0], 2) + torch.pow(line_param_b[i-1, 1], 2))
#                 adj_b[i, parent] = 1.0 / impedance
#                 adj_b[parent, i] = 1.0 / impedance
#             data["adj"].append(adj_b)
#
#             # 1.4 真实节点数
#             data["node_count"].append(torch.tensor(real_node, device=device, dtype=torch.int32))
#
#         # 转换为Batch张量
#         data["pred_node"] = torch.stack(data["pred_node"], dim=0)
#         data["gt_node"] = torch.stack(data["gt_node"], dim=0)
#         data["adj"] = torch.stack(data["adj"], dim=0)
#         data["node_count"] = torch.stack(data["node_count"], dim=0)
#         return data
#
#     # 生成模拟数据
#     sim_data = generate_sim_data(batch_size, max_node, device)
#
#     # 2. 初始化损失函数
#     physics_loss_fn = PhysicsInformedLoss(lambda_=0.5).to(device)
#
#     # 3. 计算损失
#     total_loss, pred_loss, physics_loss = physics_loss_fn(
#         pred_node=sim_data["pred_node"],
#         gt_node=sim_data["gt_node"],
#         pred_line=sim_data["pred_line"],
#         gt_line=sim_data["gt_line"],
#         adj=sim_data["adj"],
#         line_param=sim_data["line_param"],
#         node_count=sim_data["node_count"]
#     )
#
#     # 4. 打印结果
#     print("=" * 60)
#     print("物理知情损失函数计算示例结果")
#     print("=" * 60)
#     print(f"Batch大小: {batch_size}")
#     print(f"Batch内最大节点数: {max_node}")
#     print(f"每个场景真实节点数: {sim_data['node_count'].tolist()}")
#     print(f"\n总损失 (Total Loss): {total_loss.item():.6f}")
#     print(f"预测误差损失 (Prediction Loss): {pred_loss.item():.6f}")
#     print(f"物理约束损失 (Physics Loss): {physics_loss.item():.6f}")
#     print(f"\n损失构成验证: Total ≈ Pred + λ×Physics → {total_loss.item():.6f} ≈ {pred_loss.item() + 0.5*physics_loss.item():.6f}")
#     print("=" * 60)