import argparse
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Subset, DataLoader

# 导入自定义模块（基于论文《GNNs Generalization Improvement...》实现）
from data_loader import SceneDataset, get_data_loader
from pretraining import GTransformerPretrain, pretrain_loop
from finetuning import GTransformerFinetune, finetune_loop
from physics_loss import PhysicsInformedLoss
from utils import calc_nrmse, calc_physics_satisfaction, generate_voltage_mask


def parse_args():
    """
    解析命令行参数（适配论文预训练、微调、推理全流程，🔶1-20、🔶1-113、🔶1-140）
    返回：解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="基于论文的配电网潮流计算GTransformer模型（20-50节点辐射型网络）")
    # 核心运行参数
    parser.add_argument("--mode", type=str, default="finetune",
                        choices=["pretrain", "finetune", "infer"],
                        help="运行模式：pretrain（预训练）、finetune（微调）、infer（推理）")
    parser.add_argument("--data_root", type=str, default="./Dataset",
                        help="数据集根路径（默认：./Dataset，需包含Sence_1~Sence_100.npy）")
    parser.add_argument("--pretrain_path", type=str, default="./pretrained_weights.pth",
                        help="预训练权重保存/加载路径（默认：./pretrained_weights.pth）")
    parser.add_argument("--finetune_path", type=str, default="./finetuned_weights.pth",
                        help="微调权重保存/加载路径（默认：./finetuned_weights.pth）")
    # 数据与训练参数
    parser.add_argument("--mask_ratio", type=float, default=0.3,
                        help="电压掩码比例（仅非平衡节点，默认0.3，🔶1-78）")
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练轮次（预训练默认50，微调默认30，🔶1-128、🔶1-141）")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch大小（默认8，适配小节点规模GPU内存）")
    # 推理模式专用参数
    parser.add_argument("--scene_idx", type=int, default=None,
                        help="推理模式指定场景编号（1~100，默认使用测试集5个场景）")

    args = parser.parse_args()
    # 参数合法性校验（贴合论文与数据集要求）
    if args.mask_ratio < 0 or args.mask_ratio > 1:
        raise ValueError(f"掩码比例mask_ratio需在[0,1]范围内，当前输入：{args.mask_ratio}（🔶1-78）")
    if args.epochs < 1:
        raise ValueError(f"训练轮次epochs需≥1，当前输入：{args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"Batch大小batch_size需≥1，当前输入：{args.batch_size}")
    if args.scene_idx is not None and (args.scene_idx < 1 or args.scene_idx > 100):
        raise ValueError(f"场景编号scene_idx需在1~100范围内，当前输入：{args.scene_idx}")
    return args


def load_data(args):
    """
    加载数据集并按模式拆分（适配论文数据使用逻辑，🔶1-118、🔶1-126）
    Args:
        args: 命令行参数对象
    Returns:
        模式对应的DataLoader或数据字典（训练/验证/测试）
    """
    print(f"=== 加载数据集（路径：{args.data_root}，模式：{args.mode}）===")
    # 1. 加载完整数据集（100个场景）
    full_dataset = SceneDataset(
        data_root=args.data_root,
        #mask_ratio=args.mask_ratio
    )
    total_scenes = len(full_dataset)
    print(f"完整数据集加载完成：共{total_scenes}个场景（20-50节点辐射型网络）")

    # 2. 按模式拆分数据集
    if args.mode == "pretrain":
        # 预训练：使用全部100个场景（无验证，🔶1-118）
        train_loader = get_data_loader(
            data_root="./Dataset",
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        print(f"预训练数据配置：Batch={args.batch_size}，场景数={total_scenes}（全量）")
        return {"train_loader": train_loader}

    elif args.mode in ["finetune", "infer"]:
        # 微调/推理：拆分80训练+15验证+5测试（🔶1-140测试逻辑）
        np.random.seed(42)  # 固定种子确保可复现
        indices = np.random.permutation(total_scenes)
        train_indices = indices[:80]
        val_indices = indices[80:95]
        test_indices = indices[95:100]

        # 构建训练/验证/测试集
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # 生成DataLoader
        train_loader = get_data_loader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = get_data_loader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1
        )
        test_loader = get_data_loader(
            dataset=test_dataset,
            batch_size=1,  # 推理时单场景处理
            shuffle=False,
            num_workers=0
        )

        print(
            f"数据集拆分完成：训练{len(train_dataset)}个场景 | 验证{len(val_dataset)}个场景 | 测试{len(test_dataset)}个场景")

        # 推理模式：若指定scene_idx，单独加载该场景
        if args.mode == "infer" and args.scene_idx is not None:
            # 加载指定场景（Sence_{scene_idx}.npy）
            scene_file = f"{args.data_root}/Sence_{args.scene_idx}.npy"
            try:
                data = np.load(scene_file, allow_pickle=False)
                node_mat, line_mat, adj_mat = data[0], data[1], data[2]
                node_count = node_mat.shape[0]
                # 生成掩码（同Dataset逻辑）
                mask = generate_voltage_mask(
                    node_count=node_count,
                    mask_ratio=args.mask_ratio,
                    balance_node_idx=1
                )
                # 转换为Tensor并添加Batch维度
                infer_data = {
                    "node_feat": torch.FloatTensor(node_mat) * (1 - mask),
                    "adj": torch.FloatTensor(adj_mat),
                    "line_param": torch.FloatTensor(line_mat),
                    "gt_node": torch.FloatTensor(node_mat),
                    "gt_line": torch.FloatTensor(line_mat),
                    "mask": mask,
                    "node_count": torch.tensor(node_count, dtype=torch.int32)
                }
                print(f"推理模式：已加载指定场景{args.scene_idx}（节点数：{node_count}）")
                return {"infer_data": infer_data, "test_loader": test_loader}
            except FileNotFoundError:
                raise FileNotFoundError(f"指定场景文件不存在：{scene_file}")

        return {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader}


def init_model(args, device):
    """
    初始化模型（根据模式适配论文的GTransformer结构，🔶1-22、🔶1-113）
    Args:
        args: 命令行参数对象
        device: 训练/推理设备（cpu/cuda）
    Returns:
        初始化后的模型对象
    """
    # 模型核心参数（贴合论文小节点适配逻辑，🔶1-128）
    model_config = {
        "d_in": 4,  # 节点特征维度（P_load, Q_load, V, θ）
        "d_model": 64,  # 嵌入维度（适配20-50节点）
        "n_heads": 4,  # 注意力头数（64//4=16，单头维度合理）
        "n_layers": 2  # GTransformer层数（避免深层过拟合）
    }
    print(f"=== 初始化模型（设备：{device}，参数：{model_config}）===")

    if args.mode == "pretrain":
        # 预训练模型：GTransformerPretrain（🔶1-20、🔶1-75）
        model = GTransformerPretrain(
            d_in=model_config["d_in"],
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            n_layers=model_config["n_layers"]
        ).to(device)
        print(f"预训练模型初始化完成（参数总数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}）")
        return model

    elif args.mode in ["finetune", "infer"]:
        # 微调/推理模型：GTransformerFinetune（加载预训练权重，🔶1-113、🔶1-141）
        try:
            model = GTransformerFinetune(
                pretrain_path=args.pretrain_path,
                d_in=model_config["d_in"],
                d_model=model_config["d_model"],
                n_heads=model_config["n_heads"],
                n_layers=model_config["n_layers"]
            ).to(device)
            # 推理模式：加载微调权重
            if args.mode == "infer":
                if not torch.isfile(args.finetune_path):
                    raise FileNotFoundError(f"微调权重文件不存在：{args.finetune_path}")
                model.load_state_dict(torch.load(args.finetune_path, map_location=device))
                model.eval()  # 推理模式切换为评估模式
                print(f"推理模型初始化完成：已加载预训练权重（{args.pretrain_path}）与微调权重（{args.finetune_path}）")
            else:
                print(f"微调模型初始化完成：已加载预训练权重（{args.pretrain_path}）")
            return model
        except FileNotFoundError as e:
            raise RuntimeError(f"模型初始化失败：{str(e)}") from e


def run_pretrain(args, data_dict, device):
    """
    执行预训练流程（基于论文自监督掩码电压预测，🔶1-23、🔶1-75）
    Args:
        args: 命令行参数对象
        data_dict: 数据加载结果（含train_loader）
        device: 训练设备
    """
    print("\n=== 启动预训练流程（论文PPGT框架简化版，🔶1-20）===")
    # 1. 初始化模型、损失函数、优化器
    model = init_model(args, device)
    loss_fn = PhysicsInformedLoss(lambda_=0.5).to(device)  # 物理约束权重λ=0.5（🔶1-82）
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 学习率1e-3（🔶1-128）

    # 2. 调整预训练轮次（默认50，🔶1-118）
    pretrain_epochs = args.epochs if args.epochs != 30 else 50
    print(f"预训练配置：轮次={pretrain_epochs}，Batch={args.batch_size}，损失函数=PhysicsInformedLoss（λ=0.5）")

    # 3. 启动预训练
    pretrain_loop(
        model=model,
        data_loader=data_dict["train_loader"],
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=pretrain_epochs,
        device=device,
        save_path=args.pretrain_path
    )
    print("=== 预训练流程完成（权重已保存至：{args.pretrain_path}）===")


def run_finetune(args, data_dict, device):
    """
    执行微调流程（基于论文预训练权重迁移，🔶1-113、🔶1-140）
    Args:
        args: 命令行参数对象
        data_dict: 数据加载结果（含train_loader、val_loader、test_loader）
        device: 训练设备
    """
    print("\n=== 启动微调流程（聚焦电压缺失下潮流计算，🔶1-141）===")
    # 1. 初始化模型、损失函数、优化器
    model = init_model(args, device)
    loss_fn = PhysicsInformedLoss(lambda_=0.5).to(device)  # 同预训练的物理约束权重
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # 学习率5e-4（弱于预训练，保护权重）

    # 2. 微调配置（早停耐心值5，解冻轮次10，🔶1-141）
    print(f"微调配置：轮次={args.epochs}，Batch={args.batch_size}，学习率=5e-4，早停耐心值=5，解冻轮次=10")

    # 3. 启动微调
    finetune_loop(
        model=model,
        train_loader=data_dict["train_loader"],
        val_loader=data_dict["val_loader"],
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        save_path=args.finetune_path,
        unfreeze_epoch=10,
        patience=5
    )
    print("=== 微调流程完成（最优权重已保存至：{args.finetune_path}）===")


def run_infer(args, data_dict, device):
    """
    执行推理流程（评估电压缺失下的潮流预测效果，🔶1-140、🔶1-186）
    Args:
        args: 命令行参数对象
        data_dict: 数据加载结果（含infer_data或test_loader）
        device: 推理设备
    """
    print("\n=== 启动推理流程（评估20-50节点辐射型网络潮流预测）===")
    # 1. 初始化模型
    model = init_model(args, device)
    # 2. 推理数据准备（指定场景或测试集）
    if "infer_data" in data_dict:
        # 推理指定场景
        infer_scenes = [data_dict["infer_data"]]
        print(f"推理数据：指定场景（节点数：{infer_scenes[0]['node_count'].item()}）")
    else:
        # 推理测试集5个场景
        infer_scenes = []
        with torch.no_grad():
            for batch in data_dict["test_loader"]:
                infer_scenes.append({
                    "node_feat": batch["node_feat"].to(device),
                    "adj": batch["adj"].to(device),
                    "line_param": [lp.to(device) for lp in batch["line_param"]],
                    "gt_node": batch["gt_node"].to(device),
                    "gt_line": batch["gt_line"].to(device),
                    "node_count": batch["node_count"].to(device)
                })
        print(f"推理数据：测试集5个场景（节点数范围：20-50）")

    # 3. 执行推理与结果分析
    total_metrics = {
        "node_v_nrmse": 0.0, "line_p_nrmse": 0.0,
        "power_satisfaction": 0.0, "voltage_satisfaction": 0.0
    }
    with torch.no_grad():
        for scene_idx, scene in enumerate(infer_scenes, 1):
            print(f"\n--- 场景{scene_idx}推理结果 ---")
            # 3.1 前向传播（预测节点电压+线路潮流）
            pred_node, pred_line = model(
                node_feat=scene["node_feat"],
                adj=scene["adj"],
                node_count=scene["node_count"],
                line_param=scene["line_param"]
            )
            # 3.2 提取真实与预测数据（截断填充节点）
            node_count = scene["node_count"].item()
            line_count = len(pred_line[0]) if isinstance(pred_line, list) else pred_line.shape[1]
            # 节点数据（电压幅值V：第2列，相角θ：第3列）
            gt_v = scene["gt_node"][0, :node_count, 2]  # 真实电压幅值
            pred_v = pred_node[0, :node_count, 2]  # 预测电压幅值
            gt_theta = scene["gt_node"][0, :node_count, 3]  # 真实电压相角
            pred_theta = pred_node[0, :node_count, 3]  # 预测电压相角
            # 线路数据（有功P：第2列，无功Q：第3列）
            gt_line_p = scene["gt_line"][0, :line_count, 2]  # 真实线路有功
            pred_line_p = pred_line[0][:, 2]  # 预测线路有功
            gt_line_q = scene["gt_line"][0, :line_count, 3]  # 真实线路无功
            pred_line_q = pred_line[0][:, 3]  # 预测线路无功

            # 3.3 打印关键预测结果（前10个节点，前5条线路）
            print("1. 节点电压预测结果（标幺值）：")
            print(f"{'节点编号':<8} {'真实V':<10} {'预测V':<10} {'真实θ(rad)':<12} {'预测θ(rad)':<12}")
            print("-" * 56)
            show_node_num = min(10, node_count)
            for i in range(show_node_num):
                print(f"{i + 1:<8} {gt_v[i]:<10.4f} {pred_v[i]:<10.4f} {gt_theta[i]:<12.4f} {pred_theta[i]:<12.4f}")

            print("\n2. 线路潮流预测结果（标幺值）：")
            print(f"{'线路编号':<8} {'真实P':<10} {'预测P':<10} {'真实Q':<10} {'预测Q':<10}")
            print("-" * 48)
            show_line_num = min(5, line_count)
            for i in range(show_line_num):
                print(
                    f"{i + 1:<8} {gt_line_p[i]:<10.4f} {pred_line_p[i]:<10.4f} {gt_line_q[i]:<10.4f} {pred_line_q[i]:<10.4f}")

            # 3.4 计算场景指标（🔶1-137、🔶1-186）
            # 节点电压NRMSE
            node_v_nrmse = calc_nrmse(
                pred=pred_v.unsqueeze(0).unsqueeze(-1),
                gt=gt_v.unsqueeze(0).unsqueeze(-1),
                node_count=torch.tensor([node_count], device=device)
            )
            # 线路有功NRMSE
            line_p_nrmse = calc_nrmse(
                pred=pred_line_p.unsqueeze(0).unsqueeze(-1),
                gt=gt_line_p.unsqueeze(0).unsqueeze(-1),
                node_count=torch.tensor([line_count], device=device)
            )
            # 功率平衡满足率（非平衡节点，误差<2.5%）
            pred_p_inj = -pred_node[0, 1:node_count, 0]  # 非平衡节点P_inj=-P_load
            gt_p_sum = torch.zeros(node_count, device=device)
            line_pairs = model._get_line_node_mapping(scene["adj"][0], scene["node_count"][0])
            for line_idx, (i, j) in enumerate(line_pairs):
                p_ij = gt_line_p[line_idx]
                gt_p_sum[i] += p_ij
                gt_p_sum[j] -= p_ij
            p_err = torch.abs(pred_p_inj - gt_p_sum[1:node_count])
            power_satisfaction = (p_err < 0.025).float().mean().item()
            # 电压约束满足率（标幺值0.95~1.05）
            voltage_satisfaction = ((pred_v >= 0.95) & (pred_v <= 1.05)).float().mean().item()

            # 3.5 打印场景指标
            print(f"\n3. 场景{scene_idx}评估指标：")
            print(f"节点电压NRMSE：{node_v_nrmse:.4f}")
            print(f"线路有功NRMSE：{line_p_nrmse:.4f}")
            print(f"功率平衡约束满足率：{power_satisfaction:.2%}")
            print(f"电压约束满足率：{voltage_satisfaction:.2%}")

            # 3.6 累计总指标
            total_metrics["node_v_nrmse"] += node_v_nrmse / len(infer_scenes)
            total_metrics["line_p_nrmse"] += line_p_nrmse / len(infer_scenes)
            total_metrics["power_satisfaction"] += power_satisfaction / len(infer_scenes)
            total_metrics["voltage_satisfaction"] += voltage_satisfaction / len(infer_scenes)

    # 4. 打印整体推理指标
    print("\n=== 推理流程完成 | 整体评估指标（所有场景平均）===")
    print(f"节点电压NRMSE：{total_metrics['node_v_nrmse']:.4f}")
    print(f"线路有功NRMSE：{total_metrics['line_p_nrmse']:.4f}")
    print(f"功率平衡约束满足率：{total_metrics['power_satisfaction']:.2%}")
    print(f"电压约束满足率：{total_metrics['voltage_satisfaction']:.2%}")
    print("=" * 60)


def main():
    """主函数：串联数据加载、模型初始化、模式分支逻辑"""
    # 1. 解析命令行参数
    args = parse_args()
    # 2. 自动检测设备（🔶1-128硬件适配逻辑）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 设备检测完成：使用{device}（若需GPU加速，请确保PyTorch与CUDA兼容）===")
    # 3. 加载数据
    data_dict = load_data(args)
    # 4. 按模式执行对应流程
    if args.mode == "pretrain":
        run_pretrain(args, data_dict, device)
    elif args.mode == "finetune":
        run_finetune(args, data_dict, device)
    elif args.mode == "infer":
        run_infer(args, data_dict, device)
    print("\n=== 所有流程执行完毕 ===")


if __name__ == "__main__":
    main()