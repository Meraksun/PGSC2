import torch
from typing import Optional


def generate_voltage_mask(
        node_count: int,
        mask_ratio: float = 0.3,
        balance_node_idx: int = 1  # 节点编号（1-based），对应tensor索引为0
) -> torch.Tensor:
    """
    生成符合配电网规则的节点电压掩码矩阵（适配辐射型网络）
    论文逻辑适配：仅掩码非平衡节点的电压数据（第2、3列），平衡节点电压不掩码（🔶1-78、🔶1-104）

    Args:
        node_count: 单个场景的真实节点数（需>=2，至少需要一个平衡节点和一个非平衡节点）
        mask_ratio: 非平衡节点电压数据的掩码比例（0~1，默认0.3）
        balance_node_idx: 平衡节点的编号（1-based，默认1，对应tensor索引0）

    Returns:
        mask: 掩码矩阵（shape: [node_count, 4]），1=掩码，0=保留
              仅非平衡节点的第2列（电压幅值）、第3列（电压相角）可能为1，其余列/节点均为0

    Raises:
        ValueError: 输入参数不合法时抛出（如node_count<2，mask_ratio超出0~1）
    """
    # 输入参数校验
    if node_count < 2:
        raise ValueError(f"节点数node_count必须>=2（至少需要一个平衡节点和一个非平衡节点），当前输入：{node_count}")
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"掩码比例mask_ratio必须在[0,1]范围内，当前输入：{mask_ratio}")
    if balance_node_idx < 1 or balance_node_idx > node_count:
        raise ValueError(f"平衡节点编号balance_node_idx（1-based）需在1~{node_count}范围内，当前输入：{balance_node_idx}")

    # 转换平衡节点编号为tensor索引（1-based → 0-based）
    balance_idx = balance_node_idx - 1

    # 1. 初始化掩码矩阵（全0，shape: [node_count, 4]）
    mask = torch.zeros((node_count, 4), dtype=torch.float32)

    # 2. 确定非平衡节点索引（排除平衡节点）
    non_balance_indices = torch.tensor([i for i in range(node_count) if i != balance_idx], dtype=torch.long)
    if len(non_balance_indices) == 0:
        raise ValueError("场景节点数需大于1，否则无是非平衡节点可掩码")

    # 3. 生成非平衡节点的电压列（第2、3列）掩码（按mask_ratio随机采样）
    # 生成掩码概率矩阵（shape: [非平衡节点数, 2]）
    mask_prob = torch.full((len(non_balance_indices), 2), mask_ratio, dtype=torch.float32)
    # 伯努利采样生成0/1掩码
    voltage_mask = torch.bernoulli(mask_prob)

    # 4. 赋值掩码到对应位置（仅非平衡节点的第2、3列）
    mask[non_balance_indices[:, None], [2, 3]] = voltage_mask

    return mask

# -------------------------- 函数示例（注释形式，取消注释可运行） --------------------------
# if __name__ == "__main__":
#     # 模拟输入：20节点场景，平衡节点编号1（索引0），掩码比例0.3
#     node_count = 20
#     mask_ratio = 0.3
#     balance_node_idx = 1
#
#     # 生成掩码
#     mask = generate_voltage_mask(node_count, mask_ratio, balance_node_idx)
#
#     # 验证结果
#     print("=== generate_voltage_mask 示例结果 ===")
#     print(f"输入：节点数={node_count}, 掩码比例={mask_ratio}, 平衡节点编号={balance_node_idx}")
#     print(f"掩码矩阵形状：{mask.shape}")
#     print(f"平衡节点（索引0）的电压列（2、3）掩码值：{mask[0, 2]}, {mask[0, 3]}（预期均为0）")
#     print(f"非平衡节点（索引1）的电压列掩码值：{mask[1, 2]}, {mask[1, 3]}（预期0或1）")
#     print(f"非电压列（0、1）掩码值：{mask[1, 0]}, {mask[1, 1]}（预期均为0）")
#     # 统计掩码比例（非平衡节点电压列）
#     non_balance_voltage_mask = mask[[i for i in range(node_count) if i != 0], [2, 3]]
#     actual_mask_ratio = non_balance_voltage_mask.mean().item()
#     print(f"非平衡节点电压列实际掩码比例：{actual_mask_ratio:.3f}（预期接近{mask_ratio}）")