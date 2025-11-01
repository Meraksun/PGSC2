import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List, Tuple
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# 导入自定义模块（基于论文及前文实现）
from pretraining import GTransformerPretrain
from physics_loss import PhysicsInformedLoss
from utils import calc_nrmse, calc_physics_satisfaction  # 评估指标（论文🔶1-137指标适配）


class GTransformerFinetune(nn.Module):
    """
    基于预训练GTransformer的微调模型（适配用户20-50节点辐射型配电网）
    核心目标：从电压缺失的节点特征中预测完整潮流（节点电压+线路潮流）
    论文逻辑适配：预训练权重迁移+下游任务专用预测头（🔶1-113、🔶1-140）
    """

    def __init__(
            self,
            pretrain_path: str,
            d_in: int = 4,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2
    ):
        """
        初始化微调模型：加载预训练权重+新增线路潮流预测头

        Args:
            pretrain_path: 预训练GTransformer权重路径（.pth文件）
            d_in: 输入节点特征维度（默认4，同预训练）
            d_model: 嵌入/中间特征维度（默认64，同预训练）
            n_heads: 注意力头数（默认4，同预训练）
            n_layers: GTransformer堆叠层数（默认2，同预训练）

        Raises:
            FileNotFoundError: 预训练权重文件不存在时抛出
        """
        super().__init__()
        # 1. 初始化预训练GTransformer骨干网络（复用预训练特征提取能力）
        self.backbone = GTransformerPretrain(
            d_in=d_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )

        # 2. 加载预训练权重（仅加载与backbone匹配的参数，忽略线路预测头参数）
        if not torch.isfile(pretrain_path):
            raise FileNotFoundError(f"预训练权重文件不存在：{pretrain_path}")
        pretrain_state = torch.load(pretrain_path, map_location="cpu")
        # 过滤权重：仅保留backbone中存在的参数（排除线路预测头的随机初始化参数）
        backbone_state = {
            k: v for k, v in pretrain_state.items()
            if k in self.backbone.state_dict()
        }
        self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"✅ 成功加载预训练权重：共加载{len(backbone_state)}/{len(pretrain_state)}个参数")

        # 3. 新增线路潮流预测头（论文🔶1-113：下游任务专用预测层）
        # 设计逻辑：辐射型网络线路数b = N-1（N为节点数），先对节点特征做"线路级聚合"
        # 聚合方式：对每条线路连接的两个节点特征取平均，得到线路级特征（b, d_model）
        self.line_agg = lambda node_feat, line_node_mapping: torch.stack([
            (node_feat[:, i, :] + node_feat[:, j, :]) / 2  # (B, d_model)
            for i, j in line_node_mapping
        ], dim=1)  # 输出：(B, b, d_model)
        # 线路潮流预测层：输入线路级特征，输出4维潮流（R/X用真实值，仅P/Q生效）
        self.line_pred_head = nn.Linear(d_model, 4)

    def _get_line_node_mapping(self, adj: torch.Tensor, node_count: torch.Tensor) -> List[Tuple[int, int]]:
        """
        辅助函数：基于邻接矩阵获取线路-节点对映射（适配辐射型网络b=N-1）
        同物理损失函数中的线路映射逻辑（确保一致性，🔶1-88）

        Args:
            adj: 单个场景的邻接矩阵（N, N）
            node_count: 单个场景的真实节点数（int）

        Returns:
            line_node_mapping: 线路-节点对列表（[(i1,j1), (i2,j2), ...]，i<j）
        """
        real_node = node_count.item()
        adj_trim = adj[:real_node, :real_node]
        line_pairs = []
        for i in range(real_node):
            for j in range(i + 1, real_node):
                if adj_trim[i, j] != 0:  # 辐射型网络无闭环，非零即线路
                    line_pairs.append((i, j))
        return line_pairs

    def forward(
            self,
            node_feat: torch.Tensor,
            adj: torch.Tensor,
            node_count: torch.Tensor,
            line_param: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播：预测完整节点电压+线路潮流（论文🔶1-140微调逻辑）

        Args:
            node_feat: 带电压缺失的输入节点特征（B, N, 4）
            adj: 拓扑邻接矩阵（B, N, N）
            node_count: 每个场景的真实节点数（B, ）
            line_param: 真实线路参数列表（B个元素，每个元素为(b, 4)，列0=R, 1=X）

        Returns:
            pred_node: 补全的节点特征（B, N, 4）
            pred_line: 预测的线路潮流列表（B个元素，每个元素为(b, 4)，R/X=真实值，P/Q=预测值）
        """
        batch_size, max_node, _ = node_feat.shape
        pred_line = []

        # 步骤1：调用预训练骨干网络补全节点特征（电压缺失修复）
        pred_node = self.backbone(node_feat=node_feat, adj=adj, node_count=node_count)
        # 截断填充节点（避免无效数据干扰）
        for b in range(batch_size):
            real_node = node_count[b].item()
            pred_node[b, real_node:, :] = 0.0

        # 步骤2：预测线路潮流（R/X用真实值，仅P/Q为预测值）
        for b in range(batch_size):
            real_node = node_count[b].item()
            # 2.1 获取当前场景的线路-节点对映射（b = real_node - 1）
            line_pairs = self._get_line_node_mapping(adj[b], node_count[b])
            b_line = len(line_pairs)
            # 2.2 提取当前场景的真实线路参数（R/X）和骨干网络输出的节点特征
            line_param_b = line_param[b]  # (b_line, 4)
            node_feat_b = pred_node[b, :real_node, :]  # (real_node, d_model=64)

            # 2.3 线路级特征聚合（连接节点特征平均）
            line_feat = self.line_agg(node_feat_b.unsqueeze(0), line_pairs)  # (1, b_line, d_model)
            line_feat = line_feat.squeeze(0)  # (b_line, d_model)

            # 2.4 预测线路潮流（4维：R, X, P, Q）
            line_pred_b = self.line_pred_head(line_feat)  # (b_line, 4)

            # 2.5 替换R/X为真实值（仅P/Q保留预测值，符合用户需求）
            line_pred_b[:, 0] = line_param_b[:, 0]  # 真实R
            line_pred_b[:, 1] = line_param_b[:, 1]  # 真实X

            pred_line.append(line_pred_b)

        return pred_node, pred_line


def finetune_loop(
        model: GTransformerFinetune,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: PhysicsInformedLoss,
        optimizer: optim.Optimizer,
        epochs: int = 30,
        device: Optional[torch.device] = None,
        save_path: str = "./finetuned_weights.pth",
        unfreeze_epoch: int = 10,  # 10轮后解冻所有层（论文🔶1-141参数微调策略）
        patience: int = 5  # 早停机制：验证损失连续5轮不下降则停止
) -> None:
    """
    配电网潮流计算任务微调循环（聚焦电压缺失场景）
    论文逻辑适配：预训练权重迁移+部分参数冻结+物理约束监督（🔶1-140、🔶1-141）

    Args:
        model: GTransformerFinetune实例
        train_loader: 训练集DataLoader（80个场景）
        val_loader: 验证集DataLoader（15个场景）
        loss_fn: 物理知情损失函数（确保预测符合物理规律，🔶1-82）
        optimizer: Adam优化器（lr=5e-4，弱于预训练避免破坏权重）
        epochs: 微调轮次（默认30，小数据集适配）
        device: 训练设备（自动检测cpu/cuda）
        save_path: 最优微调权重保存路径
        unfreeze_epoch: 解冻底层参数的轮次（默认10）
        patience: 早停机制耐心值（默认5）
    """
    # 1. 设备自动检测与模型部署
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)
    print(f"=== 开始微调 | 设备: {device} | 总轮次: {epochs} | 解冻轮次: {unfreeze_epoch} ===")

    # 2. 初始参数冻结（仅训练顶层GTransformer和线路预测头，🔶1-141）
    def freeze_bottom_layers(freeze: bool):
        """冻结/解冻GTransformer前1层DyMPN参数"""
        for layer_idx in range(min(1, model.backbone.n_layers)):
            dympn_layer = model.backbone.gtransformer_layers[layer_idx]["dympn"]
            for param in dympn_layer.parameters():
                param.requires_grad = not freeze

    freeze_bottom_layers(freeze=True)
    print("🔒 初始状态：冻结GTransformer前1层DyMPN参数，仅训练顶层与线路预测头")

    # 3. 早停机制初始化
    best_val_loss = float("inf")
    patience_counter = 0

    # 4. 微调主循环
    for epoch in range(1, epochs + 1):
        # 4.1 轮次前处理：解冻底层参数（若达到解冻轮次）
        if epoch == unfreeze_epoch:
            freeze_bottom_layers(freeze=False)
            print(f"🔓 Epoch {epoch}：解冻所有层参数，全模型微调")

        # -------------------------- 训练阶段 --------------------------
        model.train()
        train_metrics = {
            "total_loss": 0.0, "pred_loss": 0.0, "physics_loss": 0.0,
            "node_v_nrmse": 0.0, "line_p_nrmse": 0.0  # 节点电压、线路有功NRMSE
        }
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)", unit="batch") as pbar:
            for batch in pbar:
                # 数据移至设备
                node_feat = batch["node_feat"].to(device)
                adj = batch["adj"].to(device)
                gt_node = batch["gt_node"].to(device)
                gt_line = batch["gt_line"].to(device)
                line_param = [lp.to(device) for lp in batch["line_param"]]
                node_count = batch["node_count"].to(device)
                batch_size = node_feat.shape[0]

                # 前向传播：预测节点电压+线路潮流
                pred_node, pred_line = model(
                    node_feat=node_feat,
                    adj=adj,
                    node_count=node_count,
                    line_param=line_param
                )

                # 计算损失（物理知情损失，同预训练逻辑，🔶1-82）
                total_loss, pred_loss, physics_loss = loss_fn(
                    pred_node=pred_node,
                    gt_node=gt_node,
                    pred_line=pred_line,
                    gt_line=gt_line,
                    adj=adj,
                    line_param=line_param,
                    node_count=node_count
                )

                # 反向传播与参数更新
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # 计算训练指标（论文🔶1-137、🔶1-140评估逻辑）
                # 节点电压NRMSE（仅真实节点，列2=电压幅值）
                node_v_nrmse = 0.0
                # 线路有功潮流NRMSE（仅真实线路，列2=有功）
                line_p_nrmse = 0.0
                for b in range(batch_size):
                    real_node = node_count[b].item()
                    real_line = len(pred_line[b])
                    # 节点电压NRMSE
                    pred_v = pred_node[b, :real_node, 2]
                    gt_v = gt_node[b, :real_node, 2]
                    node_v_nrmse += calc_nrmse(pred_v, gt_v) / batch_size
                    # 线路有功NRMSE
                    pred_p = pred_line[b][:, 2]
                    gt_p = gt_line[b][:, 2]
                    line_p_nrmse += calc_nrmse(pred_p, gt_p) / batch_size

                # 累计训练指标
                train_metrics["total_loss"] += total_loss.item() * batch_size
                train_metrics["pred_loss"] += pred_loss.item() * batch_size
                train_metrics["physics_loss"] += physics_loss.item() * batch_size
                train_metrics["node_v_nrmse"] += node_v_nrmse * batch_size
                train_metrics["line_p_nrmse"] += line_p_nrmse * batch_size

                # 进度条更新
                pbar.set_postfix({
                    "总损失": f"{total_loss.item():.6f}",
                    "电压NRMSE": f"{node_v_nrmse:.4f}",
                    "有功NRMSE": f"{line_p_nrmse:.4f}"
                })

        # 训练指标平均化（按样本数）
        train_sample_num = len(train_loader.dataset)
        train_metrics = {k: v / train_sample_num for k, v in train_metrics.items()}
        print(f"\n📊 Epoch {epoch} 训练指标：")
        print(
            f"   总损失: {train_metrics['total_loss']:.6f} | 预测损失: {train_metrics['pred_loss']:.6f} | 物理损失: {train_metrics['physics_loss']:.6f}")
        print(
            f"   节点电压NRMSE: {train_metrics['node_v_nrmse']:.4f} | 线路有功NRMSE: {train_metrics['line_p_nrmse']:.4f}")

        # -------------------------- 验证阶段 --------------------------
        model.eval()
        val_metrics = {
            "total_loss": 0.0, "pred_loss": 0.0, "physics_loss": 0.0,
            "node_v_nrmse": 0.0, "line_p_nrmse": 0.0,
            "power_balance_satisfaction": 0.0  # 功率平衡约束满足率（🔶1-137）
        }
        with torch.no_grad():  # 禁用梯度计算，加速验证
            with tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} (Val)", unit="batch") as pbar:
                for batch in pbar:
                    # 数据移至设备
                    node_feat = batch["node_feat"].to(device)
                    adj = batch["adj"].to(device)
                    gt_node = batch["gt_node"].to(device)
                    gt_line = batch["gt_line"].to(device)
                    line_param = [lp.to(device) for lp in batch["line_param"]]
                    node_count = batch["node_count"].to(device)
                    batch_size = node_feat.shape[0]

                    # 前向传播
                    pred_node, pred_line = model(
                        node_feat=node_feat,
                        adj=adj,
                        node_count=node_count,
                        line_param=line_param
                    )

                    # 计算验证损失
                    total_loss, pred_loss, physics_loss = loss_fn(
                        pred_node=pred_node,
                        gt_node=gt_node,
                        pred_line=pred_line,
                        gt_line=gt_line,
                        adj=adj,
                        line_param=line_param,
                        node_count=node_count
                    )

                    # 计算验证指标
                    # 1. NRMSE指标（同训练阶段）
                    node_v_nrmse = 0.0
                    line_p_nrmse = 0.0
                    # 2. 功率平衡约束满足率（误差<2.5%标幺值，🔶1-137）
                    power_balance_satisfaction = 0.0
                    for b in range(batch_size):
                        real_node = node_count[b].item()
                        real_line = len(pred_line[b])
                        # 节点电压NRMSE
                        pred_v = pred_node[b, :real_node, 2]
                        gt_v = gt_node[b, :real_node, 2]
                        node_v_nrmse += calc_nrmse(pred_v, gt_v) / batch_size
                        # 线路有功NRMSE
                        pred_p = pred_line[b][:, 2]
                        gt_p = gt_line[b][:, 2]
                        line_p_nrmse += calc_nrmse(pred_p, gt_p) / batch_size
                        # 功率平衡满足率（仅非平衡节点）
                        pred_p_inj = -pred_node[b, 1:real_node, 0]  # 非平衡节点P_inj=-P_load
                        # 计算线路潮流和（简化：用真实线路潮流求和，聚焦电压预测的满足率）
                        gt_p_sum = torch.zeros(real_node, device=device)
                        line_pairs = model._get_line_node_mapping(adj[b], node_count[b])
                        for line_idx, (i, j) in enumerate(line_pairs):
                            p_ij = gt_line[b][line_idx, 2]
                            gt_p_sum[i] += p_ij
                            gt_p_sum[j] -= p_ij
                        # 功率平衡误差（非平衡节点）
                        p_err = torch.abs(pred_p_inj - gt_p_sum[1:real_node])
                        satisfaction = (p_err < 0.025).float().mean()  # 2.5%标幺值误差
                        power_balance_satisfaction += satisfaction.item() / batch_size

                    # 累计验证指标
                    val_metrics["total_loss"] += total_loss.item() * batch_size
                    val_metrics["pred_loss"] += pred_loss.item() * batch_size
                    val_metrics["physics_loss"] += physics_loss.item() * batch_size
                    val_metrics["node_v_nrmse"] += node_v_nrmse * batch_size
                    val_metrics["line_p_nrmse"] += line_p_nrmse * batch_size
                    val_metrics["power_balance_satisfaction"] += power_balance_satisfaction * batch_size

                    # 进度条更新
                    pbar.set_postfix({
                        "val总损失": f"{total_loss.item():.6f}",
                        "val电压NRMSE": f"{node_v_nrmse:.4f}",
                        "功率满足率": f"{power_balance_satisfaction:.2%}"
                    })

        # 验证指标平均化
        val_sample_num = len(val_loader.dataset)
        val_metrics = {k: v / val_sample_num for k, v in val_metrics.items()}
        print(f"📊 Epoch {epoch} 验证指标：")
        print(
            f"   总损失: {val_metrics['total_loss']:.6f} | 预测损失: {val_metrics['pred_loss']:.6f} | 物理损失: {val_metrics['physics_loss']:.6f}")
        print(f"   节点电压NRMSE: {val_metrics['node_v_nrmse']:.4f} | 线路有功NRMSE: {val_metrics['line_p_nrmse']:.4f}")
        print(f"   功率平衡约束满足率: {val_metrics['power_balance_satisfaction']:.2%}")

        # -------------------------- 早停机制与权重保存 --------------------------
        # 保存验证损失最低的模型（论文🔶1-140最优模型选择逻辑）
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存最优模型至: {save_path}（验证损失: {best_val_loss:.6f}）")
        else:
            patience_counter += 1
            print(f"⚠️  验证损失未下降（连续{patience_counter}/{patience}轮）")
            # 早停触发
            if patience_counter >= patience:
                print(f"🛑 早停机制触发：验证损失连续{patience}轮不下降，停止微调")
                break

    # -------------------------- 微调结束：测试集评估 --------------------------
    # 从验证集拆分测试集（5个场景，🔶1-140测试逻辑）
    test_indices = np.random.choice(len(val_loader.dataset), size=5, replace=False)
    test_dataset = Subset(val_loader.dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.load_state_dict(torch.load(save_path))  # 加载最优权重
    model.eval()
    test_metrics = {
        "node_v_nrmse": 0.0, "line_p_nrmse": 0.0,
        "power_balance_satisfaction": 0.0, "voltage_satisfaction": 0.0  # 电压约束满足率
    }
    with torch.no_grad():
        print("\n=== 微调结束：测试集评估（5个场景） ===")
        for batch_idx, batch in enumerate(test_loader, 1):
            node_feat = batch["node_feat"].to(device)
            adj = batch["adj"].to(device)
            gt_node = batch["gt_node"].to(device)
            gt_line = batch["gt_line"].to(device)
            line_param = [lp.to(device) for lp in batch["line_param"]]
            node_count = batch["node_count"].to(device)
            real_node = node_count[0].item()

            # 前向传播
            pred_node, pred_line = model(
                node_feat=node_feat,
                adj=adj,
                node_count=node_count,
                line_param=line_param
            )

            # 计算测试指标
            # 1. 节点电压NRMSE
            pred_v = pred_node[0, :real_node, 2]
            gt_v = gt_node[0, :real_node, 2]
            node_v_nrmse = calc_nrmse(pred_v, gt_v)
            # 2. 线路有功NRMSE
            pred_p = pred_line[0][:, 2]
            gt_p = gt_line[0][:, 2]
            line_p_nrmse = calc_nrmse(pred_p, gt_p)
            # 3. 功率平衡满足率
            pred_p_inj = -pred_node[0, 1:real_node, 0]
            gt_p_sum = torch.zeros(real_node, device=device)
            line_pairs = model._get_line_node_mapping(adj[0], node_count[0])
            for line_idx, (i, j) in enumerate(line_pairs):
                p_ij = gt_line[0][line_idx, 2]
                gt_p_sum[i] += p_ij
                gt_p_sum[j] -= p_ij
            p_err = torch.abs(pred_p_inj - gt_p_sum[1:real_node])
            power_satisfaction = (p_err < 0.025).float().mean().item()
            # 4. 电压约束满足率（标幺值0.95~1.05，🔶1-137）
            voltage_satisfaction = ((pred_v >= 0.95) & (pred_v <= 1.05)).float().mean().item()

            # 累计测试指标
            test_metrics["node_v_nrmse"] += node_v_nrmse / 5
            test_metrics["line_p_nrmse"] += line_p_nrmse / 5
            test_metrics["power_balance_satisfaction"] += power_satisfaction / 5
            test_metrics["voltage_satisfaction"] += voltage_satisfaction / 5

            print(f"场景{batch_idx}：")
            print(f"  节点电压NRMSE: {node_v_nrmse:.4f} | 线路有功NRMSE: {line_p_nrmse:.4f}")
            print(f"  功率平衡满足率: {power_satisfaction:.2%} | 电压约束满足率: {voltage_satisfaction:.2%}")

    # 打印最终测试指标
    print("\n=== 最终测试指标（5个场景平均） ===")
    print(f"节点电压NRMSE: {test_metrics['node_v_nrmse']:.4f}")
    print(f"线路有功NRMSE: {test_metrics['line_p_nrmse']:.4f}")
    print(f"功率平衡约束满足率: {test_metrics['power_balance_satisfaction']:.2%}")
    print(f"电压约束满足率: {test_metrics['voltage_satisfaction']:.2%}")
    print(f"\n🎉 微调完成！最优模型已保存至: {save_path}")

# -------------------------- 微调启动示例（完整流程） --------------------------
# if __name__ == "__main__":
#     """
#     示例：从数据拆分到微调启动的完整流程（适配用户20-50节点辐射型配电网）
#     依赖模块：data_loader（SceneDataset）、pretraining（GTransformerPretrain）、utils（评估指标）
#     """
#     # 1. 导入依赖模块
#     from data_loader import SceneDataset, get_data_loader
#     import numpy as np
#
#     # 2. 配置微调参数（论文🔶1-128、🔶1-140参数适配）
#     FINETUNE_CONFIG = {
#         "data_root": "./Dataset",               # 用户数据集路径
#         "mask_ratio": 0.3,                      # 电压缺失比例（同预训练）
#         "pretrain_path": "./pretrained_weights.pth",  # 预训练权重路径
#         "save_path": "./finetuned_weights.pth", # 微调权重保存路径
#         "batch_size": 4,                        # 微调Batch（小于预训练，避免过拟合）
#         "epochs": 30,                           # 微调轮次
#         "lr": 5e-4,                             # 学习率（预训练的1/2，保护预训练权重）
#         "unfreeze_epoch": 10,                   # 解冻底层参数轮次
#         "patience": 5,                          # 早停耐心值
#         "d_in": 4,                              # 输入特征维度
#         "d_model": 64,                          # 嵌入维度（同预训练）
#         "n_heads": 4,                           # 注意力头数（同预训练）
#         "n_layers": 2                           # GTransformer层数（同预训练）
#     }
#
#     # 3. 数据集加载与拆分（100场景：80训练+15验证+5测试）
#     print("=== 加载并拆分数据集 ===")
#     # 3.1 加载完整数据集
#     full_dataset = SceneDataset(
#         data_root=FINETUNE_CONFIG["data_root"],
#         mask_ratio=FINETUNE_CONFIG["mask_ratio"]
#     )
#     # 3.2 随机拆分索引（固定种子确保可复现）
#     np.random.seed(42)
#     total_scenes = len(full_dataset)  # 100
#     indices = np.random.permutation(total_scenes)
#     train_indices = indices[:80]     # 80训练
#     val_indices = indices[80:95]     # 15验证
#     # 3.3 创建训练/验证数据集与DataLoader
#     train_dataset = Subset(full_dataset, train_indices)
#     val_dataset = Subset(full_dataset, val_indices)
#     train_loader = get_data_loader(
#         dataset=train_dataset,
#         batch_size=FINETUNE_CONFIG["batch_size"],
#         shuffle=True,
#         num_workers=2
#     )
#     val_loader = get_data_loader(
#         dataset=val_dataset,
#         batch_size=FINETUNE_CONFIG["batch_size"],
#         shuffle=False,
#         num_workers=1
#     )
#     print(f"数据集拆分完成：训练{len(train_dataset)}个场景 | 验证{len(val_dataset)}个场景 | 测试5个场景（从验证集拆分）")
#
#     # 4. 初始化微调模型、损失函数、优化器
#     print("\n=== 初始化微调组件 ===")
#     # 4.1 微调模型（加载预训练权重）
#     model = GTransformerFinetune(
#         pretrain_path=FINETUNE_CONFIG["pretrain_path"],
#         d_in=FINETUNE_CONFIG["d_in"],
#         d_model=FINETUNE_CONFIG["d_model"],
#         n_heads=FINETUNE_CONFIG["n_heads"],
#         n_layers=FINETUNE_CONFIG["n_layers"]
#     )
#     # 4.2 物理知情损失函数（λ=0.5，同预训练）
#     loss_fn = PhysicsInformedLoss(lambda_=0.5)
#     # 4.3 Adam优化器（lr=5e-4，论文推荐）
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=FINETUNE_CONFIG["lr"],
#         weight_decay=1e-5  # 轻微权重衰减，缓解过拟合
#     )
#     print(f"模型参数总数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#
#     # 5. 启动微调
#     print("\n=== 启动微调流程 ===")
#     finetune_loop(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#         epochs=FINETUNE_CONFIG["epochs"],
#         save_path=FINETUNE_CONFIG["save_path"],
#         unfreeze_epoch=FINETUNE_CONFIG["unfreeze_epoch"],
#         patience=FINETUNE_CONFIG["patience"]
#     )