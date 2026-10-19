import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import binary_closing, generate_binary_structure, label
from torch.utils.data import DataLoader, Dataset

from dataset.data import (
    MultiPlaneCenterCropDataset,
    SyncMedicalTransform,
    find_and_setup_data,
    find_test_data,
)
from modelfile.model import DWTForward, ECABlock, FUE2D, ROIUNet
from paper_metrics_utils import positive_slice_metric_dict, summarize_metric


METHOD54_MODEL_TYPE = "enhance_dualpath_unet_uaf_aspp_fpguard_condexpert_ms"
METHOD54_MODEL_DESC = "方法54：方法42 + 条件化专家融合"
DEFAULT_SAVE_DIR = "/home/jxy/xiuqiu3334/A/path4.7_aspp_fpguard_condexpert_ms"


def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TransformSubset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, mask = self.base[self.indices[i]]
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask


def build_case_folds(dataset, num_folds=5, seed=12345):
    if not hasattr(dataset, "volume_files") or not hasattr(dataset, "slice_info"):
        raise AttributeError("dataset 缺少 volume_files 或 slice_info，无法按病例划分")

    num_cases = len(dataset.volume_files)
    if num_cases < num_folds:
        raise ValueError(f"病例数 {num_cases} 小于折数 {num_folds}，无法做 {num_folds} 折")

    rng = np.random.default_rng(seed)
    case_indices = np.arange(num_cases)
    rng.shuffle(case_indices)
    split_cases = np.array_split(case_indices, num_folds)

    folds = []
    for fold_id in range(num_folds):
        val_cases = sorted(int(x) for x in split_cases[fold_id].tolist())
        train_cases = sorted(
            int(x)
            for i, arr in enumerate(split_cases)
            if i != fold_id
            for x in arr.tolist()
        )
        folds.append({"train_cases": train_cases, "val_cases": val_cases})
    return folds


def get_fold_slice_indices(dataset, folds, fold_id):
    train_cases = set(folds[fold_id]["train_cases"])
    val_cases = set(folds[fold_id]["val_cases"])

    train_idx, val_idx = [], []
    for slice_i, info in enumerate(dataset.slice_info):
        volume_idx = int(info["volume_idx"])
        if volume_idx in train_cases:
            train_idx.append(slice_i)
        elif volume_idx in val_cases:
            val_idx.append(slice_i)
    return train_idx, val_idx


def save_case_folds_json(dataset, folds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "dataset_5folds.json")
    payload = {
        "num_cases": len(dataset.volume_files),
        "cases": list(dataset.volume_files),
        "folds": [],
    }
    for fold_id, fold in enumerate(folds):
        payload["folds"].append({
            "fold": fold_id,
            "train_cases_idx": fold["train_cases"],
            "val_cases_idx": fold["val_cases"],
            "train_cases_files": [dataset.volume_files[j] for j in fold["train_cases"]],
            "val_cases_files": [dataset.volume_files[j] for j in fold["val_cases"]],
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"五折划分已保存: {out_path}")
    return out_path


def build_roi_dataset(
    volumes_dir,
    masks_dir,
    target_size,
    background_ratio,
    relative_size,
    mask_threshold,
    transform,
    seed,
):
    return MultiPlaneCenterCropDataset(
        volumes_dir=volumes_dir,
        masks_dir=masks_dir,
        target_size=target_size,
        background_ratio=background_ratio,
        relative_size=relative_size,
        mask_threshold=mask_threshold,
        transform=transform,
        seed=seed,
    )


class ASPP2D(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 2, 4, 6)):
        super().__init__()
        branches = []
        for rate in rates:
            kernel_size = 1 if rate == 1 else 3
            padding = 0 if rate == 1 else rate
            branches.append(nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=rate,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        fused_channels = out_channels * (len(rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[-2:], mode="bilinear", align_corners=True)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))


class ROIUNetASPP(ROIUNet):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128),
        dropout_rate=0.3,
        aspp_rates=(1, 2, 4, 6),
    ):
        super().__init__(in_channels, out_channels, list(features), dropout_rate)
        self.aspp = ASPP2D(features[2], features[2], rates=aspp_rates)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        bottleneck = self.aspp(bottleneck)
        bottleneck = self.bottleneck_dropout(bottleneck)
        return e1, e2, e3, bottleneck

    def forward_pyramid(self, x):
        e1, e2, e3, bottleneck = self.encode(x)

        d3 = self.upconv3(bottleneck)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        d3 = self.dec3_dropout(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        return {
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "bottleneck": bottleneck,
            "d3": d3,
            "d2": d2,
            "d1": d1,
        }


class ConditionalExpertFusion2D(nn.Module):
    """方法54的条件化专家融合模块。"""

    def __init__(self, channels, hidden=32, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        fused_channels = channels * 3
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, kernel_size=1),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fused_channels, hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(num_experts)
        ])
        self.refine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, spatial_feat, wavelet_feat):
        if wavelet_feat.shape[-2:] != spatial_feat.shape[-2:]:
            wavelet_feat = F.interpolate(
                wavelet_feat,
                size=spatial_feat.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        delta = wavelet_feat - spatial_feat
        expert_input = torch.cat([spatial_feat, wavelet_feat, torch.abs(delta)], dim=1)
        weights = torch.softmax(self.router(expert_input), dim=1)
        expert_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=1)
        fused_delta = (expert_outputs * weights.unsqueeze(2)).sum(dim=1)
        fused = spatial_feat + fused_delta
        return self.refine(fused), weights


class Method54CondExpertMSASPP(nn.Module):
    """方法54：方法42 + 条件化专家融合。"""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128),
        wavelet="db4",
        dropout_rate=0.3,
        wavelet_channels=16,
        aspp_rates=(1, 2, 4, 6),
    ):
        super().__init__()
        self.wavelet_transform = DWTForward(wavelet)

        self.wavelet_adapter = nn.Sequential(
            nn.Conv2d(in_channels * 4, wavelet_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(wavelet_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(wavelet_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.wavelet_eca = ECABlock(in_channels * 4)

        self.spatial_unet = ROIUNetASPP(in_channels, out_channels, features, dropout_rate, aspp_rates=aspp_rates)
        self.wavelet_unet = ROIUNetASPP(in_channels * 4, out_channels, features, dropout_rate, aspp_rates=aspp_rates)

        self.mid_expert_fusion = ConditionalExpertFusion2D(features[0], hidden=32, num_experts=3)
        self.spatial_final_refine = nn.Sequential(
            nn.Conv2d(features[0] * 2, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )
        self.wavelet_final_refine = nn.Sequential(
            nn.Conv2d(features[0] * 2, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )

        self.spatial_fue = FUE2D(features[0], hidden=16)
        self.wavelet_fue = FUE2D(features[0], hidden=16)
        self.final_expert_fusion = ConditionalExpertFusion2D(features[0], hidden=32, num_experts=3)
        self.final_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        spatial_pyr = self.spatial_unet.forward_pyramid(x)

        wavelet_coeffs = self.wavelet_transform(x)
        adapted_wavelet = wavelet_coeffs + self.wavelet_adapter(wavelet_coeffs)
        adapted_wavelet = self.wavelet_eca(adapted_wavelet)
        wavelet_pyr = self.wavelet_unet.forward_pyramid(adapted_wavelet)

        spatial_mid = spatial_pyr["d2"]
        wavelet_mid = wavelet_pyr["d2"]
        mid_fused, _ = self.mid_expert_fusion(spatial_mid, wavelet_mid)

        mid_up = F.interpolate(
            mid_fused,
            size=spatial_pyr["d1"].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        spatial_final = self.spatial_final_refine(torch.cat([spatial_pyr["d1"], mid_up], dim=1))
        wavelet_final = wavelet_pyr["d1"]
        if wavelet_final.shape[-2:] != spatial_final.shape[-2:]:
            wavelet_final = F.interpolate(
                wavelet_final,
                size=spatial_final.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        wavelet_final = self.wavelet_final_refine(torch.cat([wavelet_final, mid_up], dim=1))

        spatial_final, _ = self.spatial_fue(spatial_final)
        wavelet_final, _ = self.wavelet_fue(wavelet_final)

        fused_feat, _ = self.final_expert_fusion(spatial_final, wavelet_final)
        return self.final_head(fused_feat)


def build_method54_model(wavelet="db4", dropout_rate=0.3):
    return Method54CondExpertMSASPP(
        in_channels=1,
        out_channels=1,
        wavelet=wavelet,
        dropout_rate=dropout_rate,
    )


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() - intersection
        return 1.0 - (intersection + self.smooth) / (union + self.smooth)


class ConservativePrecisionLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=6.0, focal_weight=0.60, dice_weight=0.25, iou_weight=0.15):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.dice_loss = DiceLoss()
        self.iou_loss = SoftIoULoss()

    def focal_loss(self, logits, targets):
        pos_w = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_w)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        return ((1.0 - pt) ** self.gamma * bce).mean()

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        iou = self.iou_loss(logits, targets)
        return self.focal_weight * focal + self.dice_weight * dice + self.iou_weight * iou


def build_method54_loss(pos_weight):
    return ConservativePrecisionLoss(gamma=2.0, pos_weight=pos_weight)


def train_one_fold(
    dataset,
    fold_id,
    folds,
    relative_size=0.2,
    pos_weight=6.0,
    wavelet="db4",
    dropout_rate=0.3,
    num_epochs=100,
    seed=12345,
    save_dir=DEFAULT_SAVE_DIR,
    batch_size=None,
):
    set_seed(seed + fold_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========== Method54 Fold {fold_id} / {len(folds) - 1} ==========")
    print(f"使用设备: {device}")

    train_idx, val_idx = get_fold_slice_indices(dataset, folds, fold_id)
    train_dataset = TransformSubset(dataset, train_idx, transform=SyncMedicalTransform())
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    if batch_size is None:
        batch_size = 16 if device.type == "cuda" else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"训练切片数: {len(train_dataset)}")
    print(f"验证切片数: {len(val_dataset)}")
    print(f"批量大小: {batch_size}")

    model = build_method54_model(wavelet=wavelet, dropout_rate=dropout_rate).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = build_method54_loss(pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_{METHOD54_MODEL_TYPE}_fold{fold_id}_rs{relative_size}.pth")
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"  Fold {fold_id} | Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Fold {fold_id} | Epoch {epoch + 1}/{num_epochs}")
        print(f"  训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  保存最佳loss模型: {best_model_path} (val_loss={best_val_loss:.4f})")

    return {
        "fold": fold_id,
        "best_val_loss": float(best_val_loss),
        "best_model_path": best_model_path,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def train_method54_5fold(
    relative_size=0.2,
    pos_weight=6.0,
    wavelet="db4",
    dropout_rate=0.3,
    roi_size=(96, 96),
    background_ratio=0.004,
    num_epochs=100,
    num_folds=5,
    seed=12345,
    save_dir=DEFAULT_SAVE_DIR,
    batch_size=None,
):
    set_seed(seed)
    volumes_dir, masks_dir, suggested_threshold = find_and_setup_data()
    print("创建方法54固定ROI数据集...")
    dataset = build_roi_dataset(
        volumes_dir=volumes_dir,
        masks_dir=masks_dir,
        target_size=roi_size,
        background_ratio=background_ratio,
        relative_size=relative_size,
        mask_threshold=suggested_threshold,
        transform=None,
        seed=seed,
    )

    if len(dataset) == 0:
        print("没有找到有效数据")
        return None

    print(f"开始构建 {num_folds} 折病例划分...")
    folds = build_case_folds(dataset, num_folds=num_folds, seed=seed)
    save_case_folds_json(dataset, folds, save_dir)

    all_fold_results = []
    for fold_id in range(num_folds):
        result = train_one_fold(
            dataset=dataset,
            fold_id=fold_id,
            folds=folds,
            relative_size=relative_size,
            pos_weight=pos_weight,
            wavelet=wavelet,
            dropout_rate=dropout_rate,
            num_epochs=num_epochs,
            seed=seed,
            save_dir=save_dir,
            batch_size=batch_size,
        )
        all_fold_results.append(result)

    best_losses = [x["best_val_loss"] for x in all_fold_results]
    summary = {
        "model_type": METHOD54_MODEL_TYPE,
        "model_desc": METHOD54_MODEL_DESC,
        "num_folds": num_folds,
        "num_epochs": num_epochs,
        "save_metric": "loss",
        "relative_size": relative_size,
        "roi_size": list(roi_size),
        "background_ratio": background_ratio,
        "pos_weight": pos_weight,
        "wavelet": wavelet,
        "dropout_rate": dropout_rate,
        "fold_results": all_fold_results,
        "best_val_loss_mean": float(np.mean(best_losses)),
        "best_val_loss_std": float(np.std(best_losses)),
    }

    summary_path = os.path.join(save_dir, f"cv5_summary_{METHOD54_MODEL_TYPE}_rs{relative_size}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== 方法54 五折交叉验证完成 ==========")
    print(f"best_val_loss 平均值: {summary['best_val_loss_mean']:.4f}")
    print(f"best_val_loss 标准差: {summary['best_val_loss_std']:.4f}")
    print(f"结果已保存到: {summary_path}")
    return summary


def _connected_components_2d(bin_mask):
    bin_mask = (bin_mask > 0).astype(np.uint8)
    structure = generate_binary_structure(2, 1)
    labeled, num = label(bin_mask, structure=structure)
    comps = []
    for comp_id in range(1, num + 1):
        area = int(np.sum(labeled == comp_id))
        if area > 0:
            comps.append((comp_id, area))
    return labeled, comps


def _filter_components_2d(bin_mask, min_area=5, keep_topk=2):
    labeled, comps = _connected_components_2d(bin_mask)
    if not comps:
        return np.zeros_like(bin_mask, dtype=np.uint8)
    comps = [(cid, area) for cid, area in comps if area >= int(min_area)]
    if not comps:
        return np.zeros_like(bin_mask, dtype=np.uint8)
    comps.sort(key=lambda x: x[1], reverse=True)
    if keep_topk is not None and int(keep_topk) > 0:
        comps = comps[:int(keep_topk)]
    keep_ids = {cid for cid, _ in comps}
    return np.isin(labeled, list(keep_ids)).astype(np.uint8)


def postprocess_prediction_2d(bin_mask, min_area=5, keep_topk=2, do_close=True):
    pred = (bin_mask > 0).astype(np.uint8)
    if do_close:
        pred = binary_closing(pred.astype(bool), structure=np.ones((3, 3), dtype=bool)).astype(np.uint8)
    pred = _filter_components_2d(pred, min_area=min_area, keep_topk=keep_topk)
    return pred.astype(np.float32)


def get_method54_model_paths(save_dir=DEFAULT_SAVE_DIR, relative_size=0.2, num_folds=5):
    return [
        os.path.join(save_dir, f"best_{METHOD54_MODEL_TYPE}_fold{fold_id}_rs{relative_size}.pth")
        for fold_id in range(num_folds)
    ]


def load_method54_ensemble(model_paths, device, wavelet="db4", dropout_rate=0.0):
    models, loaded_paths = [], []
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"权重不存在，跳过: {model_path}")
            continue
        try:
            model = build_method54_model(wavelet=wavelet, dropout_rate=dropout_rate)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
            loaded_paths.append(model_path)
            print(f"成功加载 fold 权重: {model_path}")
        except Exception as exc:
            print(f"加载失败 {model_path}: {exc}")
    return models, loaded_paths


def evaluate_method54_ensemble(
    models,
    test_loader,
    test_files,
    device,
    threshold=0.5,
    apply_postprocess=False,
    postprocess_min_area=5,
    postprocess_keep_topk=2,
    postprocess_do_close=True,
):
    print("=== 开始方法54 5-fold Ensemble ROI 切片级测试 ===")
    for model in models:
        model.eval()
        model.to(device)

    per_slice = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks_np = masks.numpy()

            prob_sum = None
            for model in models:
                probabilities = torch.sigmoid(model(images))
                prob_sum = probabilities if prob_sum is None else prob_sum + probabilities

            avg_probabilities = prob_sum / len(models)
            predictions = (avg_probabilities > threshold).float()

            if apply_postprocess:
                processed_predictions = []
                for i in range(predictions.size(0)):
                    pred_2d = predictions[i].detach().cpu().numpy()[0]
                    pred_2d = postprocess_prediction_2d(
                        pred_2d,
                        min_area=postprocess_min_area,
                        keep_topk=postprocess_keep_topk,
                        do_close=postprocess_do_close,
                    )
                    processed_predictions.append(pred_2d)
                predictions = torch.from_numpy(np.stack(processed_predictions)[:, None, :, :]).to(
                    device=device,
                    dtype=avg_probabilities.dtype,
                )

            for i in range(images.size(0)):
                pred = predictions[i].detach().cpu().numpy()[0]
                target = masks_np[i, 0]
                per_slice.append(positive_slice_metric_dict(target, pred))

            if (batch_idx + 1) % 5 == 0:
                print(f"已处理 {batch_idx + 1} 个批次")

    summary = {}
    for key in ["dice", "iou", "precision", "recall", "f1", "hd95", "assd", "rvd"]:
        summary.update(summarize_metric(per_slice, key))
    summary["total_samples"] = len(per_slice)
    summary["test_files"] = test_files
    return summary


def test_method54(
    relative_size=0.2,
    threshold=0.5,
    batch_size=8,
    num_folds=5,
    wavelet="db4",
    save_dir=DEFAULT_SAVE_DIR,
    device="cuda",
    apply_postprocess=False,
):
    print(f"=== {METHOD54_MODEL_DESC} Ensemble 测试并保存 JSON ===")
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"使用设备: {device}")

    volumes_dir, masks_dir, test_files, suggested_threshold = find_test_data()
    if volumes_dir is None:
        print("无法找到独立测试集")
        return None

    test_dataset = build_roi_dataset(
        volumes_dir=volumes_dir,
        masks_dir=masks_dir,
        target_size=(96, 96),
        relative_size=relative_size,
        mask_threshold=suggested_threshold,
        background_ratio=0.0,
        transform=None,
        seed=12345,
    )
    if len(test_dataset) == 0:
        print("测试数据集为空，无法进行测试")
        return None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model_paths = get_method54_model_paths(save_dir=save_dir, relative_size=relative_size, num_folds=num_folds)

    print("默认 5-fold 权重路径:")
    for path in model_paths:
        print(f"- {path}")

    models, loaded_paths = load_method54_ensemble(model_paths, device=device, wavelet=wavelet, dropout_rate=0.0)
    if not models:
        print("一个 fold 权重都没加载成功，无法测试")
        return None

    summary = evaluate_method54_ensemble(
        models=models,
        test_loader=test_loader,
        test_files=test_files,
        device=device,
        threshold=threshold,
        apply_postprocess=apply_postprocess,
    )

    print("\n切片级测试指标:")
    print(f"Dice系数: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    print(f"IoU系数: {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
    print(f"精确率: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    print(f"召回率: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    print(f"F1分数: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"测试样本数: {summary['total_samples']}")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(save_dir, f"ensemble_test_results_{METHOD54_MODEL_TYPE}_{timestamp}.json")
    payload = {
        "test_timestamp": timestamp,
        "model_type": METHOD54_MODEL_TYPE,
        "model_desc": METHOD54_MODEL_DESC,
        "source_file": "/home/jxy/xiuqiu3334/A/method54_only.py",
        "threshold": threshold,
        "postprocess_applied": apply_postprocess,
        "input_channels": 1,
        "context_offsets": [0],
        "ensemble_model_paths": loaded_paths,
        "num_models": len(models),
        "test_set_info": {
            "volumes_dir": volumes_dir,
            "masks_dir": masks_dir,
            "test_files": test_files,
            "total_test_slices": len(test_dataset),
        },
        "performance_summary": {
            "dice_mean": float(summary["dice_mean"]),
            "dice_std": float(summary["dice_std"]),
            "iou_mean": float(summary["iou_mean"]),
            "iou_std": float(summary["iou_std"]),
            "precision_mean": float(summary["precision_mean"]),
            "precision_std": float(summary["precision_std"]),
            "recall_mean": float(summary["recall_mean"]),
            "recall_std": float(summary["recall_std"]),
            "f1_mean": float(summary["f1_mean"]),
            "f1_std": float(summary["f1_std"]),
            "hd95_mean": float(summary["hd95_mean"]) if summary["hd95_mean"] is not None else None,
            "hd95_std": float(summary["hd95_std"]) if summary["hd95_std"] is not None else None,
            "assd_mean": float(summary["assd_mean"]) if summary["assd_mean"] is not None else None,
            "assd_std": float(summary["assd_std"]) if summary["assd_std"] is not None else None,
            "rvd_mean": float(summary["rvd_mean"]) if summary["rvd_mean"] is not None else None,
            "rvd_std": float(summary["rvd_std"]) if summary["rvd_std"] is not None else None,
        },
        "detailed_metrics": summary,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"方法54 测试结果已保存到: {result_path}")
    return result_path


def smoke_test(wavelet="db4"):
    model = build_method54_model(wavelet=wavelet, dropout_rate=0.3)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 96, 96))
    print(f"smoke output shape: {tuple(output.shape)}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")


def parse_args():
    parser = argparse.ArgumentParser(description="方法54独立训练/测试脚本，只包含方法54。")
    parser.add_argument("--mode", choices=["train", "test", "smoke"], default="train")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--relative-size", type=float, default=0.2)
    parser.add_argument("--pos-weight", type=float, default=6.0)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--apply-postprocess", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch_size if args.batch_size > 0 else None
    if args.mode == "smoke":
        smoke_test(wavelet=args.wavelet)
    elif args.mode == "train":
        train_method54_5fold(
            relative_size=args.relative_size,
            pos_weight=args.pos_weight,
            wavelet=args.wavelet,
            dropout_rate=args.dropout_rate,
            num_epochs=args.epochs,
            num_folds=args.num_folds,
            seed=args.seed,
            save_dir=args.save_dir,
            batch_size=batch_size,
        )
    elif args.mode == "test":
        test_method54(
            relative_size=args.relative_size,
            threshold=args.threshold,
            batch_size=batch_size or 8,
            num_folds=args.num_folds,
            wavelet=args.wavelet,
            save_dir=args.save_dir,
            device=args.device,
            apply_postprocess=args.apply_postprocess,
        )


if __name__ == "__main__":
    main()
