import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import numpy as np
import os

class SceneDataset(Dataset):
    """
    配电网场景数据集（适配20-50节点辐射型网络）
    每个场景包含：节点特征矩阵、线路特征矩阵、邻接矩阵
    """
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.scene_files = [f for f in os.listdir(data_root) if f.startswith("Sence_") and f.endswith(".npy")]
        self.scene_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # 按场景编号排序

    def __len__(self) -> int:
        return len(self.scene_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """加载单个场景数据，返回节点矩阵、线路矩阵、邻接矩阵和场景编号"""
        scene_file = self.scene_files[idx]
        scene_path = os.path.join(self.data_root, scene_file)
        data = np.load(scene_path, allow_pickle=False)
        node_matrix, line_matrix, adj_matrix = data[0], data[1], data[2]

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
            adj_pad[:a, :a] = item["adj_matrix"]
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