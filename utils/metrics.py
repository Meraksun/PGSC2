import torch
from typing import Optional


def calc_nrmse(
        pred: torch.Tensor,
        gt: torch.Tensor,
        node_count: torch.Tensor
) -> float:
    """
    计算节点/线路特征的标准化均方根误差（NRMSE），适配Batch处理与填充节点屏蔽
    论文逻辑适配：标幺值下的误差评估，屏蔽无效填充节点（🔶1-137、🔶1-140）

    Args:
        pred: 预测值张量（shape: [B, N, *] 或 [B, b, *]，B=Batch，N=最大节点数，b=最大线路数）
        gt: 真实值张量（shape与pred完全一致）
        node_count: 每个场景的真实节点/线路数（shape: [B]，用于截断填充数据）

    Returns:
        nrmse: 平均NRMSE值（标量），计算逻辑：NRMSE = RMSE / (gt_max - gt_min)

    Raises:
        ValueError: pred与gt形状不匹配，或node_count长度与Batch大小不一致
    """
    # 输入参数校验
    if pred.shape != gt.shape:
        raise ValueError(f"pred与gt形状不匹配：pred={pred.shape}, gt={gt.shape}")
    if len(node_count) != pred.shape[0]:
        raise ValueError(f"node_count长度（{len(node_count)}）与Batch大小（{pred.shape[0]}）不一致")

    batch_size = pred.shape[0]
    total_rmse = 0.0
    gt_range = gt.max() - gt.min()  # 标幺值下通常为0.4（1.2-0.8），也可按场景单独计算

    # 遍历每个场景，截断填充节点/线路
    for b in range(batch_size):
        real_count = node_count[b].item()
        # 截断到真实数量（排除填充数据）
        pred_b = pred[b, :real_count, ...].flatten()  # 展平为1维便于计算
        gt_b = gt[b, :real_count, ...].flatten()

        # 计算单个场景的RMSE
        mse = torch.mean(torch.square(pred_b - gt_b))
        rmse = torch.sqrt(mse)
        total_rmse += rmse.item()

    # 计算平均RMSE与NRMSE（避免gt_range为0导致除零）
    avg_rmse = total_rmse / batch_size
    nrmse = avg_rmse / gt_range if gt_range > 1e-6 else 0.0
    return nrmse


def calc_physics_satisfaction(
        power_balance_err: torch.Tensor,
        node_count: torch.Tensor,
        threshold: float = 0.025  # 2.5%标幺值误差阈值（🔶1-137）
) -> float:
    """
    计算功率平衡约束满足率：误差小于阈值的有效节点数 / 总有效节点数
    论文逻辑适配：物理约束满足性评估，仅统计非填充节点（🔶1-137、🔶1-186）

    Args:
        power_balance_err: 功率平衡误差张量（shape: [B, N, 2]，2=有功/无功误差）
        node_count: 每个场景的真实节点数（shape: [B]，用于截断填充节点）
        threshold: 误差阈值（标幺值，默认0.025=2.5%）

    Returns:
        satisfaction_rate: 功率平衡约束满足率（标量，0~1）

    Raises:
        ValueError: power_balance_err维度不合法，或node_count长度与Batch大小不一致
    """
    # 输入参数校验
    if power_balance_err.dim() != 3 or power_balance_err.shape[2] != 2:
        raise ValueError(f"power_balance_err需为[B, N, 2]维度，当前形状：{power_balance_err.shape}")
    if len(node_count) != power_balance_err.shape[0]:
        raise ValueError(f"node_count长度（{len(node_count)}）与Batch大小（{power_balance_err.shape[0]}）不一致")

    batch_size = power_balance_err.shape[0]
    total_satisfied = 0
    total_valid_nodes = 0

    # 遍历每个场景，统计有效节点的满足情况
    for b in range(batch_size):
        real_count = node_count[b].item()
        # 截断到真实节点数（排除填充节点），并计算误差绝对值
        err_b = torch.abs(power_balance_err[b, :real_count, :])  # [real_count, 2]
        # 满足条件：有功和无功误差均小于阈值（逻辑与）
        satisfied_nodes = torch.logical_and(err_b[:, 0] < threshold, err_b[:, 1] < threshold)
        # 累计满足节点数与总有效节点数
        total_satisfied += satisfied_nodes.sum().item()
        total_valid_nodes += real_count

    # 计算满足率（避免总有效节点数为0）
    satisfaction_rate = total_satisfied / total_valid_nodes if total_valid_nodes > 0 else 0.0
    return satisfaction_rate

# -------------------------- 函数示例（注释形式，取消注释可运行） --------------------------
# if __name__ == "__main__":
#     # 模拟输入：Batch=2，最大节点数=50，真实节点数=[20,30]
#     batch_size = 2
#     max_node = 50
#     node_count = torch.tensor([20, 30], dtype=torch.int32)
#
#     # 1. calc_nrmse 示例
#     # 模拟预测值（标幺值：电压0.8~1.2，潮流-0.05~0.05）
#     pred = torch.rand(batch_size, max_node, 4) * 0.4 + 0.8  # 节点特征示例
#     gt = pred + torch.randn_like(pred) * 0.02  # 真实值=预测值+小噪声
#     nrmse = calc_nrmse(pred, gt, node_count)
#     print("=== calc_nrmse 示例结果 ===")
#     print(f"Batch大小：{batch_size}，最大节点数：{max_node}，真实节点数：{node_count.tolist()}")
#     print(f"NRMSE值：{nrmse:.4f}（预期接近0.05~0.1）")
#
#     # 2. calc_physics_satisfaction 示例
#     # 模拟功率平衡误差（标幺值：大部分<0.025，少数超标）
#     power_balance_err = torch.rand(batch_size, max_node, 2) * 0.05  # 0~0.05
#     satisfaction = calc_physics_satisfaction(power_balance_err, node_count)
#     print("\n=== calc_physics_satisfaction 示例结果 ===")
#     print(f"误差阈值：0.025（2.5%标幺值）")
#     print(f"功率平衡约束满足率：{satisfaction:.2%}（预期接近50%~60%）")