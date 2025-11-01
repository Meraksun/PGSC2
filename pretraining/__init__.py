# 导出GTransformer预训练核心类与启动函数，便于主程序模块调用
from .pretrainer import GTransformerPretrain, pretrain_loop

# 明确导出内容，符合模块化编程规范，避免导入冗余
__all__ = ["GTransformerPretrain", "pretrain_loop"]