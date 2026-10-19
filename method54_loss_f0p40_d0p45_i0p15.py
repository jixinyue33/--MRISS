#!/usr/bin/env python3
"""基于《固定的ROI区域+小波变换.py》方法54的损失权重实验。

除 ConservativePrecisionLoss 的三项权重外，方法54的模型、数据处理、
病例级五折划分、训练参数与独立测试流程均直接复用原文件：

    focal_weight = 0.40
    dice_weight  = 0.45
    iou_weight   = 0.15

实验使用独立输出目录，不覆盖原方法54结果。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_DIR = Path("/home/jxy/xiuqiu3334/A")
SOURCE_FILE = PROJECT_DIR / "固定的ROI区域+小波变换.py"
SCRIPT_FILE = Path(__file__).resolve()
MODEL_TYPE = "enhance_dualpath_unet_uaf_aspp_fpguard_condexpert_ms"
MODEL_DESC = "方法54损失权重实验：Focal=0.40, Dice=0.45, IoU=0.15"
DEFAULT_SAVE_DIR = (
    PROJECT_DIR / "path4.7_aspp_fpguard_condexpert_ms_loss_f0p40_d0p45_i0p15"
)
LOSS_WEIGHTS = {
    "focal": 0.40,
    "dice": 0.45,
    "iou": 0.15,
}
EXPECTED_TEST_SLICES = 308


def load_source_module():
    """加载指定原文件，避免复制或改写方法54的模型实现。"""
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    spec = importlib.util.spec_from_file_location(
        "fixed_roi_wavelet_method54_source",
        SOURCE_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载源文件: {SOURCE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def install_adjusted_loss(module, pos_weight: float) -> None:
    """仅替换方法54使用的损失构造器。"""
    if abs(sum(LOSS_WEIGHTS.values()) - 1.0) > 1e-12:
        raise ValueError(f"损失权重之和必须为1，实际为 {sum(LOSS_WEIGHTS.values())}")

    def build_adjusted_loss(model_type: str, requested_pos_weight: float):
        if model_type != MODEL_TYPE:
            raise ValueError(f"本实验仅支持方法54，收到 model_type={model_type}")
        return module.ConservativePrecisionLoss(
            gamma=2.0,
            pos_weight=requested_pos_weight,
            focal_weight=LOSS_WEIGHTS["focal"],
            dice_weight=LOSS_WEIGHTS["dice"],
            iou_weight=LOSS_WEIGHTS["iou"],
        )

    module.build_loss = build_adjusted_loss

    criterion = module.build_loss(MODEL_TYPE, pos_weight)
    actual = {
        "focal": float(criterion.focal_weight),
        "dice": float(criterion.dice_weight),
        "iou": float(criterion.iou_weight),
    }
    if actual != LOSS_WEIGHTS:
        raise RuntimeError(f"损失权重注入失败: expected={LOSS_WEIGHTS}, actual={actual}")


def expected_checkpoints(save_dir: Path, folds: int, relative_size: float) -> list[Path]:
    return [
        save_dir / f"best_{MODEL_TYPE}_fold{fold}_rs{relative_size}.pth"
        for fold in range(folds)
    ]


def experiment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "method54_loss_weight_adjustment",
        "model_type": MODEL_TYPE,
        "model_desc": MODEL_DESC,
        "source_file": str(SOURCE_FILE),
        "launcher_file": str(SCRIPT_FILE),
        "changed_variable_only": "loss_coefficients",
        "loss_formula": (
            "lambda_focal * focal_loss + lambda_dice * dice_loss "
            "+ lambda_iou * iou_loss"
        ),
        "loss_coefficients": LOSS_WEIGHTS,
        "loss_weight_sum": sum(LOSS_WEIGHTS.values()),
        "gamma": 2.0,
        "pos_weight": args.pos_weight,
        "relative_size": args.relative_size,
        "roi_size": list(args.roi_size),
        "background_ratio": args.background_ratio,
        "wavelet": args.wavelet,
        "dropout_rate": args.dropout,
        "epochs": args.epochs,
        "num_folds": args.folds,
        "seed": args.seed,
        "threshold": args.threshold,
        "expected_test_slices": args.expected_test_slices,
        "test_postprocess": args.apply_postprocess,
    }


def train(module, args: argparse.Namespace, save_dir: Path) -> Path:
    checkpoints = expected_checkpoints(save_dir, args.folds, args.relative_size)
    existing = [path.exists() for path in checkpoints]
    if all(existing) and not args.force_train:
        print(f"发现完整的 {args.folds} 折权重，跳过训练: {save_dir}")
        return save_dir / f"cv5_summary_{MODEL_TYPE}_rs{args.relative_size}.json"
    if any(existing) and not args.force_train:
        missing = [str(path) for path in checkpoints if not path.exists()]
        raise RuntimeError(
            "输出目录中只有部分fold权重。请检查后使用 --force-train 明确重训。"
            "\n缺失文件:\n" + "\n".join(missing)
        )

    summary = module.train_fixed_roi_model_5fold(
        relative_size=args.relative_size,
        pos_weight=args.pos_weight,
        model_type=MODEL_TYPE,
        wavelet=args.wavelet,
        dropout_rate=args.dropout,
        roi_size=tuple(args.roi_size),
        background_ratio=args.background_ratio,
        num_epochs=args.epochs,
        num_folds=args.folds,
        seed=args.seed,
        save_dir=str(save_dir),
    )
    if summary is None:
        raise RuntimeError("方法54训练没有返回结果")

    summary["model_desc"] = MODEL_DESC
    summary["source_file"] = str(SOURCE_FILE)
    summary["launcher_file"] = str(SCRIPT_FILE)
    summary["changed_variable_only"] = "loss_coefficients"
    summary["loss_formula"] = (
        "lambda_focal * focal_loss + lambda_dice * dice_loss + lambda_iou * iou_loss"
    )
    summary["loss_coefficients"] = LOSS_WEIGHTS
    summary_path = save_dir / f"cv5_summary_{MODEL_TYPE}_rs{args.relative_size}.json"
    write_json(summary_path, summary)
    return summary_path


def verify_test_set(module, args: argparse.Namespace) -> int:
    volumes_dir, masks_dir, _, suggested_threshold = module.find_test_data()
    if volumes_dir is None:
        raise FileNotFoundError("未找到方法54使用的独立测试集")
    dataset = module.build_roi_dataset(
        model_type=MODEL_TYPE,
        volumes_dir=volumes_dir,
        masks_dir=masks_dir,
        target_size=tuple(args.roi_size),
        relative_size=args.relative_size,
        mask_threshold=suggested_threshold,
        background_ratio=0.0,
        transform=None,
        seed=args.seed,
    )
    actual = len(dataset)
    if actual != args.expected_test_slices:
        raise RuntimeError(
            f"测试切片数不匹配：期望 {args.expected_test_slices}，实际 {actual}"
        )
    print(f"已核验方法54独立测试集：{actual} 个切片")
    return actual


def test(module, args: argparse.Namespace, save_dir: Path) -> Path:
    checkpoints = expected_checkpoints(save_dir, args.folds, args.relative_size)
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少测试所需的fold权重:\n" + "\n".join(missing))

    verify_test_set(module, args)

    def current_experiment_paths(
        model_type: str,
        relative_size: float = 0.2,
        num_folds: int = 5,
    ):
        if model_type != MODEL_TYPE:
            raise ValueError(f"非预期模型: {model_type}")
        return [
            str(path)
            for path in expected_checkpoints(save_dir, num_folds, relative_size)
        ]

    module.get_default_model_paths_ensemble = current_experiment_paths
    result = module.test_ensemble_and_save(
        model_type=MODEL_TYPE,
        model_desc=MODEL_DESC,
        relative_size=args.relative_size,
        threshold=args.threshold,
        batch_size=args.test_batch_size,
        num_folds=args.folds,
        wavelet=args.wavelet,
        save_dir=str(save_dir),
        device=args.device,
        apply_postprocess=args.apply_postprocess,
    )
    if result is None:
        raise RuntimeError("方法54独立测试没有返回结果")

    result_path = Path(result)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    actual_slices = payload["test_set_info"]["total_test_slices"]
    if actual_slices != args.expected_test_slices:
        raise RuntimeError(
            f"结果文件记录了 {actual_slices} 张切片，不是 {args.expected_test_slices}"
        )
    payload["source_file"] = str(SOURCE_FILE)
    payload["launcher_file"] = str(SCRIPT_FILE)
    payload["changed_variable_only"] = "loss_coefficients"
    payload["loss_coefficients"] = LOSS_WEIGHTS
    write_json(result_path, payload)
    return result_path


def smoke_test(module, args: argparse.Namespace) -> None:
    criterion = module.build_loss(MODEL_TYPE, args.pos_weight)
    model = module.build_model(
        model_type=MODEL_TYPE,
        wavelet=args.wavelet,
        dropout_rate=args.dropout,
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, args.roi_size[0], args.roi_size[1]))
    print(
        "loss weights verified: "
        f"focal={criterion.focal_weight}, "
        f"dice={criterion.dice_weight}, "
        f"iou={criterion.iou_weight}"
    )
    print(f"smoke output shape: {tuple(output.shape)}")
    print(f"params: {sum(parameter.numel() for parameter in model.parameters()):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "基于《固定的ROI区域+小波变换.py》方法54的损失权重实验："
            "Focal=0.40, Dice=0.45, IoU=0.15"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("train", "test", "both", "smoke"),
        default="train",
    )
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--relative-size", type=float, default=0.2)
    parser.add_argument("--roi-size", type=int, nargs=2, default=(96, 96))
    parser.add_argument("--background-ratio", type=float, default=0.004)
    parser.add_argument("--pos-weight", type=float, default=6.0)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--expected-test-slices", type=int, default=EXPECTED_TEST_SLICES)
    parser.add_argument("--apply-postprocess", action="store_true")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="允许在独立输出目录已有部分或完整权重时明确重训",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds != 5:
        raise ValueError("为与方法54保持一致，本实验要求 --folds 5")

    module = load_source_module()
    install_adjusted_loss(module, args.pos_weight)
    save_dir = Path(args.save_dir)
    write_json(save_dir / "experiment_config.json", experiment_manifest(args))

    print("基准来源:", SOURCE_FILE)
    print("方法54模型:", MODEL_TYPE)
    print(
        "新损失权重: "
        f"Focal={LOSS_WEIGHTS['focal']:.2f}, "
        f"Dice={LOSS_WEIGHTS['dice']:.2f}, "
        f"IoU={LOSS_WEIGHTS['iou']:.2f}"
    )
    print("独立输出目录:", save_dir)

    if args.mode == "smoke":
        smoke_test(module, args)
    elif args.mode == "train":
        train(module, args, save_dir)
    elif args.mode == "test":
        test(module, args, save_dir)
    else:
        train(module, args, save_dir)
        test(module, args, save_dir)


if __name__ == "__main__":
    main()
