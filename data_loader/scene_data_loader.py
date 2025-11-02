import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class SceneDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.scene_paths = self._get_scene_paths()  # 获取所有场景路径
        self.max_node = self._get_max_node()  # 计算数据集中的最大节点数（关键：用于填充）
        self._preload_and_validate()  # 保留原有的校验逻辑

    def _get_scene_paths(self):
        """获取所有场景文件路径"""
        scene_files = [f for f in os.listdir(self.data_root)
                       if f.startswith("Sence_") and f.endswith(".npz")]
        scene_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        return [os.path.join(self.data_root, f) for f in scene_files]

    def _get_max_node(self):
        """计算数据集中的最大节点数（用于统一形状）"""
        max_node = 0
        for path in self.scene_paths:
            data = np.load(path, allow_pickle=False)
            node_mat = data["node_data"]
            max_node = max(max_node, node_mat.shape[0])
        return max_node  # 例如返回50（覆盖20-50节点的范围）

    def _preload_and_validate(self):
        """保留原有的校验逻辑（无需修改）"""
        for idx, scene_path in enumerate(self.scene_paths):
            try:
                data = np.load(scene_path, allow_pickle=False)
                required_keys = {"node_data", "line_data", "topology_matrix"}
                if not required_keys.issubset(data.files):
                    missing = required_keys - set(data.files)
                    raise ValueError(f"文件{scene_path}缺少必要的键：{missing}")

                node_mat = data["node_data"].astype(np.float32)
                line_mat = data["line_data"].astype(np.float32)
                adj_mat = data["topology_matrix"].astype(np.float32)

                a = node_mat.shape[0]
                if node_mat.shape != (a, 4):
                    raise ValueError(f"文件{scene_path}的节点矩阵形状应为(a,4)，实际为{node_mat.shape}")

                b = line_mat.shape[0]
                if line_mat.shape != (b, 4):
                    raise ValueError(f"文件{scene_path}的线路矩阵形状应为(b,4)，实际为{line_mat.shape}")

                if adj_mat.shape != (a, a):
                    raise ValueError(f"文件{scene_path}的邻接矩阵形状应为({a},{a})，实际为{adj_mat.shape}")

                if node_mat.dtype != np.float32 or line_mat.dtype != np.float32 or adj_mat.dtype != np.float32:
                    raise ValueError(
                        f"文件{scene_path}的数据类型应为float32，实际为node:{node_mat.dtype}, line:{line_mat.dtype}, adj:{adj_mat.dtype}")

            except Exception as e:
                raise RuntimeError(f"校验文件{scene_path}时出错：{str(e)}") from e

    def __getitem__(self, idx):
        """加载单个场景并填充到最大节点数"""
        scene_path = self.scene_paths[idx]
        data = np.load(scene_path, allow_pickle=False)

        # 加载原始数据
        node_mat = data["node_data"].astype(np.float32)  # 形状：(N, 4)，N为当前场景节点数
        line_mat = data["line_data"].astype(np.float32)  # 形状：(B, 4)，B为线路数
        adj_mat = data["topology_matrix"].astype(np.float32)  # 形状：(N, N)

        # 关键：计算当前场景的真实节点数
        node_count = node_mat.shape[0]  # 例如41、32等

        # 1. 填充节点矩阵（到max_node行）
        # 原形状(N,4) → 填充为(max_node,4)，新增部分补0
        pad_node = np.zeros((self.max_node - node_count, 4), dtype=np.float32)
        node_mat_padded = np.concatenate([node_mat, pad_node], axis=0)

        # 2. 填充邻接矩阵（到max_node×max_node）
        # 原形状(N,N) → 填充为(max_node,max_node)，新增部分补0
        pad_adj = np.zeros((self.max_node, self.max_node), dtype=np.float32)
        pad_adj[:node_count, :node_count] = adj_mat  # 左上角保留原始邻接矩阵
        adj_mat_padded = pad_adj

        # 线路矩阵无需填充（后续处理中按node_count过滤）

        return {
            "node_matrix": node_mat_padded,  # 已填充，形状固定为(max_node,4)
            "adj_matrix": adj_mat_padded,  # 已填充，形状固定为(max_node,max_node)
            "line_matrix": line_mat,
            "node_count": node_count  # 真实节点数（用于后续过滤填充部分）
        }

    def __len__(self):
        return len(self.scene_paths)


def get_data_loader(data_root=None, dataset=None, batch_size=8, shuffle=True, num_workers=2):
    """生成DataLoader，使用高效的collate_fn"""
    if dataset is None:
        dataset = SceneDataset(data_root=data_root)

    # 改进collate_fn：直接堆叠（因为已填充为固定形状，无需列表转换）
    def collate_fn(batch):
        return {
            "node_matrix": torch.stack([torch.tensor(d["node_matrix"]) for d in batch]),
            "adj_matrix": torch.stack([torch.tensor(d["adj_matrix"]) for d in batch]),
            "line_matrix": [torch.tensor(d["line_matrix"]) for d in batch],  # 线路数不同，保留列表
            "node_count": torch.tensor([d["node_count"] for d in batch], dtype=torch.int32)
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )