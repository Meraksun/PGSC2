import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class SceneDataset(Dataset):
    def __init__(self, data_root: str = "./Dataset", scene_range: List[int] = None):
        self.data_root = data_root
        self.scene_paths = []
        self.scene_indices = []  # 存储场景编号（1-100）

        # 初始化场景范围（默认1-100）
        if scene_range is None:
            scene_range = list(range(1, 101))
        self.scene_range = scene_range

        # 收集所有场景文件路径
        for scene_idx in scene_range:
            scene_file = os.path.join(data_root, f"Sence_{scene_idx}.npz")
            if not os.path.exists(scene_file):
                raise FileNotFoundError(f"场景文件不存在：{scene_file}")
            self.scene_paths.append(scene_file)
            self.scene_indices.append(scene_idx)

        # 预加载并校验数据格式
        self._preload_and_validate()

    def _preload_and_validate(self):
        for idx, scene_path in enumerate(self.scene_paths):
            try:
                # 加载NPZ文件并按名称读取数组
                data = np.load(scene_path, allow_pickle=False)
                required_keys = {"node_data", "line_data", "topology_matrix"}
                if not required_keys.issubset(data.files):
                    missing = required_keys - set(data.files)
                    raise ValueError(f"文件{scene_path}缺少必要的键：{missing}")

                node_mat = data["node_data"].astype(np.float32)
                line_mat = data["line_data"].astype(np.float32)
                adj_mat = data["topology_matrix"].astype(np.float32)

                # 校验节点矩阵形状（假设为 a×4）
                a = node_mat.shape[0]
                if node_mat.shape != (a, 4):
                    raise ValueError(f"文件{scene_path}的节点矩阵形状应为(a,4)，实际为{node_mat.shape}")

                # 校验线路矩阵形状（假设为 b×5）
                b = line_mat.shape[0]
                if line_mat.shape != (b, 4):
                    raise ValueError(f"文件{scene_path}的线路矩阵形状应为(b,5)，实际为{line_mat.shape}")

                # 校验邻接矩阵形状（a×a）
                if adj_mat.shape != (a, a):
                    raise ValueError(f"文件{scene_path}的邻接矩阵形状应为({a},{a})，实际为{adj_mat.shape}")

                # 校验数据类型（假设为float32）
                if node_mat.dtype != np.float32 or line_mat.dtype != np.float32 or adj_mat.dtype != np.float32:
                    raise ValueError(
                        f"文件{scene_path}的数据类型应为float32，实际为node:{node_mat.dtype}, line:{line_mat.dtype}, adj:{adj_mat.dtype}")

            except Exception as e:
                raise RuntimeError(f"校验文件{scene_path}时出错：{str(e)}") from e

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]
        scene_idx = self.scene_indices[idx]
        data = np.load(scene_path)

        # 提取数据并转为torch张量
        node_mat = torch.from_numpy(data["node_data"].astype(np.float32))
        line_mat = torch.from_numpy(data["line_data"].astype(np.float32))
        adj_mat = torch.from_numpy(data["topology_matrix"].astype(np.float32))
        node_count = node_mat.shape[0]

        # 构建掩码（示例逻辑，可根据实际需求调整）
        a = node_mat.shape[0]
        mask = torch.ones(a, dtype=torch.bool)

        return {
            "node_matrix": node_mat,
            "line_matrix": line_mat,
            "adj_matrix": adj_mat,
            "mask": mask,
            "scene_idx": scene_idx,
            "node_count": node_count
        }


def _collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义Batch拼接函数：处理不同节点数的场景，用0填充至Batch内最大节点数
    """
    max_nodes = max(item["node_matrix"].shape[0] for item in batch)
    max_lines = max(item["line_matrix"].shape[0] for item in batch)

    node_matrix_batch = []
    line_matrix_batch = []
    adj_matrix_batch = []
    mask_batch = []
    scene_idx_batch = []

    for item in batch:
        a = item["node_matrix"].shape[0]
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

        # 掩码填充
        mask_pad = torch.zeros(max_nodes, dtype=torch.bool)
        mask_pad[:a] = item["mask"]
        mask_batch.append(mask_pad)

        scene_idx_batch.append(item["scene_idx"])

    return {
        "node_matrix": torch.stack(node_matrix_batch),
        "line_matrix": torch.stack(line_matrix_batch),
        "adj_matrix": torch.stack(adj_matrix_batch),
        "mask": torch.stack(mask_batch),
        "scene_idx": torch.tensor(scene_idx_batch, dtype=torch.long)
    }


def get_data_loader(data_root: str = "./Dataset", batch_size: int = 8,
                    scene_range: List[int] = None, shuffle: bool = True,
                    num_workers: int = 10) -> DataLoader:  # 新增 num_workers 参数
    """
    创建数据加载器
    """
    dataset = SceneDataset(data_root=data_root, scene_range=scene_range)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        num_workers=num_workers  # 传递 num_workers 到 DataLoader
    )