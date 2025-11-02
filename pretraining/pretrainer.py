import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
import os
from tqdm import tqdm

# 导入自定义模块
from gnn_layers import DyMPNLayer, GraphMultiHeadAttention
from utils import generate_voltage_mask

class SceneDataset(Dataset):
    """
    配电网场景数据集（适配20-50节点辐射型网络）
    每个场景包含：节点特征矩阵、线路特征矩阵、邻接矩阵
    """
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.scene_files = [f for f in os.listdir(data_root) if f.startswith("Sence_") and f.endswith(".npz")]
        self.scene_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # 按场景编号排序

    def __len__(self) -> int:
        return len(self.scene_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """加载单个场景数据，返回节点矩阵、线路矩阵、邻接矩阵和场景编号"""
        scene_file = self.scene_files[idx]
        scene_path = os.path.join(self.data_root, scene_file)
        data = np.load(scene_path, allow_pickle=False)
        
        # .npz 文件需要通过键名访问，尝试多种可能的键名
        if 'node' in data.files:
            # 如果使用命名键保存：node, line, adj
            node_matrix = data['node']
            line_matrix = data['line']
            adj_matrix = data['adj']
        elif 'arr_0' in data.files:
            # 如果使用默认键保存：arr_0, arr_1, arr_2
            node_matrix = data['arr_0']
            line_matrix = data['arr_1']
            adj_matrix = data['arr_2']
        elif len(data.files) >= 3:
            # 如果有多个文件，按字母顺序取前三个
            files = sorted(data.files)
            node_matrix = data[files[0]]
            line_matrix = data[files[1]]
            adj_matrix = data[files[2]]
        else:
            raise ValueError(f"无法从 {scene_file} 中提取3个数组，找到的文件键：{data.files}")

        # 转换为Tensor
        return {
            "node_matrix": torch.FloatTensor(node_matrix),
            "line_matrix": torch.FloatTensor(line_matrix),
            "adj_matrix": torch.FloatTensor(adj_matrix),
            "scene_idx": torch.tensor(int(scene_file.split("_")[1].split(".")[0]), dtype=torch.long),
            # 新增：当前场景的真实节点数（节点矩阵的行数）
            "node_count": torch.tensor(node_matrix.shape[0], dtype=torch.long)
        }

def get_data_loader(
        data_root: str = "./Dataset",
        dataset: Dataset = None,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 2
) -> DataLoader:
    """获取数据集加载器，支持自定义collate_fn处理变长节点数"""
    if dataset is None:
        dataset = SceneDataset(data_root)

    def _collate_fn(batch: List[Dict]) -> Dict:
        """
        自定义Batch拼接函数：处理不同节点数的场景，用0填充至Batch内最大节点数
        新增：计算每个场景的真实节点数并添加到batch中
        """
        max_nodes = max(item["node_matrix"].shape[0] for item in batch)
        max_lines = max(item["line_matrix"].shape[0] for item in batch)

        node_matrix_batch = []
        line_matrix_batch = []
        adj_matrix_batch = []
        scene_idx_batch = []
        node_count_batch = []  # 存储每个场景的真实节点数

        for item in batch:
            a = item["node_matrix"].shape[0]  # 真实节点数（当前场景）
            b = item["line_matrix"].shape[0]
            adj_shape = item["adj_matrix"].shape  # 邻接矩阵的实际形状

            # 节点矩阵填充
            node_pad = torch.zeros(max_nodes, 4, dtype=item["node_matrix"].dtype)
            node_pad[:a] = item["node_matrix"]
            node_matrix_batch.append(node_pad)

            # 线路矩阵填充
            line_pad = torch.zeros(max_lines, 4, dtype=item["line_matrix"].dtype)
            line_pad[:b] = item["line_matrix"]
            line_matrix_batch.append(line_pad)

            # 邻接矩阵填充
            adj_pad = torch.zeros(max_nodes, max_nodes, dtype=item["adj_matrix"].dtype)
            # 使用实际邻接矩阵大小和节点矩阵大小的较小值，避免维度不匹配
            adj_size = min(a, adj_shape[0], adj_shape[1])
            adj_pad[:adj_size, :adj_size] = item["adj_matrix"][:adj_size, :adj_size]
            adj_matrix_batch.append(adj_pad)

            # 收集场景编号和真实节点数
            scene_idx_batch.append(item["scene_idx"])
            node_count_batch.append(a)  # 记录当前场景的真实节点数

        return {
            "node_matrix": torch.stack(node_matrix_batch),
            "line_matrix": torch.stack(line_matrix_batch),
            "adj_matrix": torch.stack(adj_matrix_batch),
            "scene_idx": torch.tensor(scene_idx_batch, dtype=torch.long),
            "node_count": torch.tensor(node_count_batch, dtype=torch.long)  # 新增：真实节点数
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn
    )


class GTransformerPretrain(nn.Module):
    """
    基于GTransformer的预训练模型（适配20-50节点辐射型配电网）
    核心目标：从电压缺失的节点特征中预测完整节点电压（自监督学习）
    论文逻辑适配：DyMPN + GraphMultiHeadAttention + 节点预测头（🔶1-20、🔶1-75）
    """

    def __init__(
            self,
            d_in: int = 4,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2
    ):
        """
        初始化预训练模型

        Args:
            d_in: 输入节点特征维度（默认4，对应P_load、Q_load、V、θ）
            d_model: 嵌入/中间特征维度（默认64，适配20-50节点）
            n_heads: 注意力头数（默认4，需满足d_model % n_heads == 0）
            n_layers: GTransformer堆叠层数（默认2，避免深层过拟合）
        """
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # 1. 构建n_layers层GTransformer（DyMPN + GraphMultiHeadAttention）
        self.gtransformer_layers = nn.ModuleList()
        for i in range(n_layers):
            layer_dict = nn.ModuleDict({
                "dympn": DyMPNLayer(d_in=d_in if i == 0 else d_model, d_model=d_model),
                "gatt": GraphMultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.1)
            })
            self.gtransformer_layers.append(layer_dict)

        # 2. 节点特征预测头：将d_model维特征映射回4维（P_load, Q_load, V, θ）
        self.node_pred_head = nn.Linear(d_model, d_in)

    def forward(
            self,
            node_feat: torch.Tensor,
            adj: torch.Tensor,
            node_count: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：从掩码节点特征预测完整节点特征（论文🔶1-75预训练逻辑）

        Args:
            node_feat: 带掩码的输入节点特征（B, N, 4），B=Batch，N=最大节点数
            adj: 拓扑邻接矩阵（B, N, N）
            node_count: 每个场景的真实节点数（B, ）

        Returns:
            pred_node: 预测的完整节点特征（B, N, 4）
        """
        batch_size, max_node, _ = node_feat.shape
        h = node_feat

        # 步骤1：通过n_layers层GTransformer提取特征
        for layer in self.gtransformer_layers:
            # 1.1 DyMPN：提取局部拓扑特征
            h = layer["dympn"](node_feat=h, adj=adj, node_count=node_count)
            # 1.2 GraphMultiHeadAttention：捕捉全局拓扑依赖
            # 生成电压掩码（用于注意力层的掩码处理）
            mask = torch.zeros(batch_size, max_node, 2, device=node_feat.device)
            for b in range(batch_size):
                real_node = node_count[b].item()
                # 检查哪些节点的电压被掩码（node_feat中V、θ为0的位置）
                # 简单判断：如果节点特征的第2、3列（V、θ）接近0，则认为被掩码
                v_theta = node_feat[b, :real_node, 2:4]
                is_masked = (torch.abs(v_theta) < 1e-6).any(dim=-1)  # (real_node,)
                mask[b, :real_node, 0] = is_masked.float()
                mask[b, :real_node, 1] = is_masked.float()
            h = layer["gatt"](h=h, mask=mask, node_count=node_count)

        # 步骤2：节点特征预测（4维输出）
        pred_node = self.node_pred_head(h)  # (B, N, 4)

        # 步骤3：截断填充节点（避免无效数据干扰）
        # 使用掩码操作避免原地修改（保护计算图）
        node_indices = torch.arange(max_node, device=pred_node.device).unsqueeze(0).expand(batch_size, -1)  # (B, N)
        node_count_expanded = node_count.unsqueeze(1).expand(-1, max_node)  # (B, N)
        pad_mask = node_indices >= node_count_expanded  # (B, N)，True表示填充节点
        pred_node = pred_node.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return pred_node


def pretrain_loop(
        model: GTransformerPretrain,
        data_loader: DataLoader,
        loss_fn,
        optimizer: optim.Optimizer,
        epochs: int = 50,
        device: Optional[torch.device] = None,
        save_path: str = "./pretrained_weights.pth",
        mask_ratio: float = 0.3
) -> None:
    """
    配电网潮流计算任务预训练循环（自监督掩码电压预测）
    论文逻辑适配：掩码电压预测 + 物理约束监督（🔶1-20、🔶1-75、🔶1-82、🔶1-118）

    Args:
        model: GTransformerPretrain实例
        data_loader: 训练集DataLoader（100个场景）
        loss_fn: 物理知情损失函数（确保预测符合物理规律）
        optimizer: Adam优化器（lr=1e-3，论文推荐）
        epochs: 预训练轮次（默认50）
        device: 训练设备（自动检测cpu/cuda）
        save_path: 预训练权重保存路径
        mask_ratio: 电压掩码比例（默认0.3，仅非平衡节点）
    """
    # 1. 设备自动检测与模型部署
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)
    print(f"=== 开始预训练 | 设备: {device} | 总轮次: {epochs} | 掩码比例: {mask_ratio} ===")

    # 2. 预训练主循环
    for epoch in range(1, epochs + 1):
        model.train()
        train_metrics = {
            "total_loss": 0.0, "pred_loss": 0.0, "physics_loss": 0.0
        }
        total_samples = 0

        with tqdm(data_loader, desc=f"Epoch {epoch}/{epochs} (Pretrain)", unit="batch") as pbar:
            for batch in pbar:
                # 2.1 数据准备
                node_matrix = batch["node_matrix"].to(device)  # (B, N, 4)
                line_matrix = batch["line_matrix"].to(device)  # (B, L, 4)
                adj = batch["adj_matrix"].to(device)  # (B, N, N)
                node_count = batch["node_count"].to(device)  # (B,)
                batch_size = node_matrix.shape[0]

                # 2.2 生成电压掩码并应用到节点特征
                node_feat_list = []
                mask_list = []
                line_param_list = []
                gt_line_list = []

                for b in range(batch_size):
                    real_node = node_count[b].item()
                    real_line = line_matrix.shape[1]  # 线路数

                    # 生成掩码（仅非平衡节点）
                    mask = generate_voltage_mask(
                        node_count=real_node,
                        mask_ratio=mask_ratio,
                        balance_node_idx=1
                    ).to(device)

                    # 应用掩码到节点特征（mask: 1=掩码，0=保留）
                    node_feat_b = node_matrix[b, :real_node, :] * (1 - mask)
                    node_feat_list.append(node_feat_b)

                    # 收集掩码用于后续处理
                    mask_pad = torch.zeros(node_matrix.shape[1], 4, device=device)
                    mask_pad[:real_node, :] = mask
                    mask_list.append(mask_pad)

                    # 准备线路数据（用于损失计算）
                    line_param_b = line_matrix[b, :real_line, :].clone()
                    line_param_list.append(line_param_b)

                    gt_line_b = line_matrix[b, :real_line, :].clone()
                    gt_line_list.append(gt_line_b)

                # 填充节点特征到统一维度
                max_node = node_matrix.shape[1]
                node_feat = torch.zeros(batch_size, max_node, 4, device=device)
                for b, feat_b in enumerate(node_feat_list):
                    real_node = node_count[b].item()
                    node_feat[b, :real_node, :] = feat_b

                # 2.3 前向传播：预测完整节点特征
                pred_node = model(
                    node_feat=node_feat,
                    adj=adj,
                    node_count=node_count
                )

                # 2.4 从预测节点特征生成线路潮流预测（用于物理损失计算）
                # 简化处理：使用一个简单的线性层从节点特征预测线路潮流
                # 这里我们使用真实线路参数（R, X），仅预测P、Q
                pred_line_list = []
                for b in range(batch_size):
                    real_node = node_count[b].item()
                    line_param_b = line_param_list[b]  # (L, 4): R, X, P, Q
                    real_line = line_param_b.shape[0]

                    # 从邻接矩阵获取线路-节点对映射
                    adj_b = adj[b, :real_node, :real_node]
                    line_pairs = []
                    for i in range(real_node):
                        for j in range(i + 1, real_node):
                            if adj_b[i, j] != 0:
                                line_pairs.append((i, j))

                    # 简单的线路潮流预测：使用连接节点的平均特征预测P、Q
                    pred_line_b = line_param_b.clone()  # 保留R、X
                    for line_idx, (i, j) in enumerate(line_pairs[:real_line]):
                        # 使用节点特征的简单组合预测P、Q（这里用简化方法）
                        node_feat_i = pred_node[b, i, :]
                        node_feat_j = pred_node[b, j, :]
                        # 简单的预测：使用V差值和P_load差值作为P、Q的粗略估计
                        v_diff = node_feat_i[2] - node_feat_j[2]  # 电压差
                        p_diff = node_feat_i[0] - node_feat_j[0]  # 负荷差
                        # 简化的线路潮流预测（实际应该用更复杂的公式）
                        pred_line_b[line_idx, 2] = v_diff * 0.1 + p_diff  # 预测P
                        pred_line_b[line_idx, 3] = v_diff * 0.05  # 预测Q（简化）

                    pred_line_list.append(pred_line_b)

                # 2.5 计算损失（物理知情损失）
                total_loss, pred_loss, physics_loss = loss_fn(
                    pred_node=pred_node,
                    gt_node=node_matrix,
                    pred_line=pred_line_list,
                    gt_line=gt_line_list,
                    adj=adj,
                    line_param=line_param_list,
                    node_count=node_count
                )

                # 2.6 反向传播与参数更新
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # 2.7 累计指标
                train_metrics["total_loss"] += total_loss.item() * batch_size
                train_metrics["pred_loss"] += pred_loss.item() * batch_size
                train_metrics["physics_loss"] += physics_loss.item() * batch_size
                total_samples += batch_size

                # 进度条更新
                pbar.set_postfix({
                    "总损失": f"{total_loss.item():.6f}",
                    "预测损失": f"{pred_loss.item():.6f}",
                    "物理损失": f"{physics_loss.item():.6f}"
                })

        # 3. 训练指标平均化
        train_metrics = {k: v / total_samples for k, v in train_metrics.items()}
        print(f"\n📊 Epoch {epoch} 预训练指标：")
        print(
            f"   总损失: {train_metrics['total_loss']:.6f} | "
            f"预测损失: {train_metrics['pred_loss']:.6f} | "
            f"物理损失: {train_metrics['physics_loss']:.6f}"
        )

        # 4. 保存权重（每个epoch都保存，覆盖之前）
        torch.save(model.state_dict(), save_path)
        if epoch % 10 == 0:
            print(f"✅ 权重已保存至: {save_path}（Epoch {epoch}）")

    print(f"\n🎉 预训练完成！最终权重已保存至: {save_path}")