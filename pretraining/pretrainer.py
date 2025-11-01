import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于训练进度可视化（可选，提升体验）

# 导入自定义模块（需确保各模块路径正确）
from gnn_layers import DyMPNLayer, GraphMultiHeadAttention
from physics_loss import PhysicsInformedLoss


class GTransformerPretrain(nn.Module):
    """
    基于论文的Graph Transformer预训练模型（PPGT简化版）
    核心结构：堆叠"DyMPN局部特征提取 + GraphMultiHeadAttention全局依赖捕捉"层
    适配20-50节点辐射型配电网，聚焦掩码电压预测任务（🔶1-20、🔶1-22、🔶1-75）
    """

    def __init__(
            self,
            d_in: int = 4,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2
    ):
        """
        初始化GTransformer预训练模型

        Args:
            d_in: 输入节点特征维度（默认4，对应P_load、Q_load、V、θ标幺值）
            d_model: 嵌入/中间特征维度（默认64，与DyMPN、注意力层一致）
            n_heads: 注意力头数（默认4，需满足d_model % n_heads == 0，适配小节点）
            n_layers: GTransformer堆叠层数（默认2，小节点规模无需深层结构）
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # 1. 输入嵌入层：将原始4维特征映射到d_model维（与DyMPN输入匹配）
        self.input_embed = nn.Linear(d_in, d_model)

        # 2. 堆叠GTransformer层：每层=DyMPN + GraphMultiHeadAttention + 残差连接 + 层归一化
        self.gtransformer_layers = nn.ModuleList()
        for _ in range(n_layers):
            dympn = DyMPNLayer(d_in=d_model, d_model=d_model)  # DyMPN输入为d_model（嵌入后特征）
            gatt = GraphMultiHeadAttention(d_model=d_model, n_heads=n_heads)
            norm = nn.LayerNorm(d_model)  # 层归一化（论文中用于稳定训练，🔶1-72）
            self.gtransformer_layers.append(nn.ModuleDict({
                "dympn": dympn,
                "gatt": gatt,
                "norm": norm
            }))

        # 3. 节点特征预测头：输出完整节点特征（P_load、Q_load、V、θ，4维）
        # 论文中预训练目标为"补全掩码特征"，此处直接预测所有节点特征（🔶1-78）
        self.node_pred_head = nn.Linear(d_model, 4)

    def forward(
            self,
            node_feat: torch.Tensor,
            adj: torch.Tensor,
            node_count: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：从带掩码的节点特征预测完整节点特征

        Args:
            node_feat: 带掩码的输入节点特征（B, N, d_in），B=Batch，N=最大节点数
            adj: 拓扑邻接矩阵（B, N, N）
            node_count: 每个场景的真实节点数（B, ）

        Returns:
            pred_node: 预测的完整节点特征（B, N, 4）
        """
        # 步骤1：输入特征嵌入（d_in→d_model）
        # 形状变化：(B, N, d_in) → (B, N, d_model)
        x = self.input_embed(node_feat)

        # 步骤2：经过n_layers层GTransformer
        for layer in self.gtransformer_layers:
            # 2.1 残差连接备份（论文中用于缓解梯度消失，🔶1-72）
            residual = x

            # 2.2 DyMPN：提取局部拓扑特征
            local_feat = layer["dympn"](node_feat=x, adj=adj, node_count=node_count)

            # 2.3 GraphMultiHeadAttention：捕捉全局拓扑依赖
            # 注意力掩码：从node_feat的掩码推导（非0即非掩码，0为掩码）
            # 掩码逻辑：节点特征中电压列（2、3列）为0 → 该节点需屏蔽注意力
            mask = (node_feat[:, :, 2:4] == 0).any(dim=-1, keepdim=True)  # (B, N, 1)
            global_feat = layer["gatt"](h=local_feat, mask=mask, node_count=node_count)

            # 2.4 残差连接 + 层归一化
            x = layer["norm"](residual + global_feat)

        # 步骤3：预测完整节点特征（d_model→4）
        pred_node = self.node_pred_head(x)

        # 步骤4：屏蔽填充节点的预测结果（填充节点特征置0，避免干扰损失计算）
        for b in range(pred_node.shape[0]):
            real_node = node_count[b].item()
            pred_node[b, real_node:, :] = 0.0

        return pred_node


def pretrain_loop(
        model: GTransformerPretrain,
        data_loader: DataLoader,
        loss_fn: PhysicsInformedLoss,
        optimizer: optim.Optimizer,
        epochs: int = 50,
        device: Optional[torch.device] = None,
        save_path: str = "./pretrained_weights.pth"
) -> None:
    """
    GTransformer自监督预训练循环（基于掩码电压预测任务）
    核心逻辑：论文"物理知情自监督预训练"简化版，仅保留掩码特征预测（🔶1-23、🔶1-75）

    Args:
        model: GTransformerPretrain实例
        data_loader: 数据加载器（返回带掩码的节点特征、真实标签等）
        loss_fn: 物理知情损失函数（PhysicsInformedLoss实例）
        optimizer: PyTorch优化器（默认Adam，lr=1e-3）
        epochs: 预训练轮数（默认50，小数据集无需多轮）
        device: 训练设备（自动检测cpu/cuda）
        save_path: 预训练权重保存路径
    """
    # 1. 设备自动检测（优先级：用户指定 > 自动检测）
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)
    print(f"=== 开始预训练 | 设备: {device} | 总轮次: {epochs} | 权重保存路径: {save_path} ===")

    # 2. 预训练主循环
    for epoch in range(1, epochs + 1):
        model.train()  # 切换训练模式（启用dropout、BatchNorm等）
        total_epoch_loss = 0.0
        total_epoch_pred_loss = 0.0
        total_epoch_physics_loss = 0.0

        # 遍历DataLoader（带进度条）
        with tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar, 1):
                # 2.1 数据移至设备
                # Batch键说明：来自data_loader.SceneDataset的__getitem__
                node_feat = batch["node_matrix"].to(device)  # 带掩码的节点特征 (B, N, 4)
                adj = batch["adj_matrix"].to(device)  # 邻接矩阵 (B, N, N)
                gt_node = batch["node_matrix"].to(device)  # 真实节点特征（标签）(B, N, 4)
                gt_line = batch["line_matrix"].to(device)  # 真实线路潮流（简化用，🔶1-88）
                line_param = batch["line_matrix"].to(device)  # 线路参数 (B个元素，每个(b_line,4))
                node_count = batch["node_matrix"].to(device)  # 真实节点数 (B,)

                # 2.2 前向传播：预测完整节点特征
                pred_node = model(node_feat=node_feat, adj=adj, node_count=node_count)

                # 2.3 计算损失（论文"预测误差+物理约束"损失，🔶1-82）
                # 简化处理：线路潮流预测暂用真实值（gt_line），仅优化节点特征预测
                total_loss, pred_loss, physics_loss = loss_fn(
                    pred_node=pred_node,
                    gt_node=gt_node,
                    pred_line=gt_line,  # 简化：用真实线路潮流替代预测（聚焦电压预测）
                    gt_line=gt_line,
                    adj=adj,
                    line_param=line_param,
                    node_count=node_count
                )

                # 2.4 反向传播与参数更新
                optimizer.zero_grad()  # 清空梯度
                total_loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数

                # 2.5 累计损失（用于日志）
                batch_size = node_feat.shape[0]
                total_epoch_loss += total_loss.item() * batch_size
                total_epoch_pred_loss += pred_loss.item() * batch_size
                total_epoch_physics_loss += physics_loss.item() * batch_size

                # 2.6 每10个Batch打印日志（🔶1-140中训练监控逻辑）
                if batch_idx % 10 == 0:
                    avg_loss = total_epoch_loss / (batch_idx * batch_size)
                    avg_pred_loss = total_epoch_pred_loss / (batch_idx * batch_size)
                    avg_physics_loss = total_epoch_physics_loss / (batch_idx * batch_size)
                    pbar.set_postfix({
                        "总损失": f"{avg_loss:.6f}",
                        "预测损失": f"{avg_pred_loss:.6f}",
                        "物理损失": f"{avg_physics_loss:.6f}"
                    })

        # 3. 每5轮保存一次模型权重（避免过拟合，便于中断后恢复）
        if epoch % 5 == 0:
            save_path_epoch = save_path.replace(".pth", f"_epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path_epoch)
            print(f"✅ Epoch {epoch} 权重已保存至: {save_path_epoch}")

    # 4. 预训练结束：保存最终权重
    torch.save(model.state_dict(), save_path)
    print(f"\n=== 预训练完成 | 最终权重保存至: {save_path} ===")
    # 计算并打印最终平均损失
    avg_final_loss = total_epoch_loss / len(data_loader.dataset)
    avg_final_pred_loss = total_epoch_pred_loss / len(data_loader.dataset)
    avg_final_physics_loss = total_epoch_physics_loss / len(data_loader.dataset)
    print(
        f"📊 最终平均损失：总损失={avg_final_loss:.6f}, 预测损失={avg_final_pred_loss:.6f}, 物理损失={avg_final_physics_loss:.6f}")

# -------------------------- 预训练启动示例（注释形式，取消注释可运行） --------------------------
# if __name__ == "__main__":
#     """
#     示例：初始化数据集、模型、损失函数，启动预训练
#     依赖模块：data_loader（SceneDataset、get_data_loader）、physics_informed_loss（PhysicsInformedLoss）
#     """
#     # 1. 导入依赖模块（需确保模块路径正确）
#     from data_loader import SceneDataset, get_data_loader
#
#     # 2. 配置预训练参数
#     PRETRAIN_CONFIG = {
#         "data_root": "./Dataset",       # 数据集路径（用户已准备）
#         "mask_ratio": 0.3,              # 电压掩码比例（论文默认0.3，🔶1-78）
#         "batch_size": 8,                # Batch大小（适配小节点，避免GPU内存不足）
#         "epochs": 50,                   # 预训练轮次
#         "lr": 1e-3,                     # 学习率（Adam默认，🔶1-128）
#         "save_path": "./pretrained_weights.pth",  # 权重保存路径
#         "d_in": 4,                      # 输入特征维度
#         "d_model": 64,                  # 嵌入维度
#         "n_heads": 4,                   # 注意力头数
#         "n_layers": 2                   # GTransformer层数
#     }
#
#     # 3. 初始化数据集与DataLoader
#     print("=== 加载数据集 ===")
#     dataset = SceneDataset(
#         data_root=PRETRAIN_CONFIG["data_root"],
#         mask_ratio=PRETRAIN_CONFIG["mask_ratio"]
#     )
#     data_loader = get_data_loader(
#         dataset=dataset,
#         batch_size=PRETRAIN_CONFIG["batch_size"],
#         shuffle=True,
#         num_workers=2
#     )
#     print(f"数据集加载完成：共{len(dataset)}个场景，Batch大小={PRETRAIN_CONFIG['batch_size']}")
#
#     # 4. 初始化模型、损失函数、优化器
#     print("\n=== 初始化模型与训练组件 ===")
#     # 4.1 模型
#     model = GTransformerPretrain(
#         d_in=PRETRAIN_CONFIG["d_in"],
#         d_model=PRETRAIN_CONFIG["d_model"],
#         n_heads=PRETRAIN_CONFIG["n_heads"],
#         n_layers=PRETRAIN_CONFIG["n_layers"]
#     )
#     # 4.2 物理知情损失函数（λ=0.5，平衡预测与物理约束）
#     loss_fn = PhysicsInformedLoss(lambda_=0.5)
#     # 4.3 优化器（Adam，论文中使用的优化器类型，🔶1-128）
#     optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_CONFIG["lr"])
#     print(f"模型参数总数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#
#     # 5. 启动预训练
#     print("\n=== 启动预训练 ===")
#     pretrain_loop(
#         model=model,
#         data_loader=data_loader,
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#         epochs=PRETRAIN_CONFIG["epochs"],
#         save_path=PRETRAIN_CONFIG["save_path"]
#     )