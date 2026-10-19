import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import nibabel as nib
import json


_PROJECT_A_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEVELOPMENT_THRESHOLD_FILE = os.path.join(
    _PROJECT_A_DIR, 'data', 'development_mask_area_threshold.json'
)


def load_frozen_development_threshold(path=DEFAULT_DEVELOPMENT_THRESHOLD_FILE):
    """Load and validate a threshold frozen from the development cohort."""
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('source_cohort') != 'development_only':
        raise ValueError(f'阈值文件不是由开发队列生成: {path}')
    threshold = float(payload['threshold'])
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f'冻结阈值超出前景比例范围: {threshold}')
    return threshold


def find_and_setup_data():
    """查找开发数据，并仅由完整开发队列确定面积阈值。"""
    print("3. 查找数据文件...")

    # 直接使用你找到的数据文件夹
    data_folder = './data'
    volumes_dir = os.path.join(data_folder, 'volumes')
    masks_dir = os.path.join(data_folder, 'masks')

    if os.path.exists(volumes_dir) and os.path.exists(masks_dir):
        volume_files = [f for f in os.listdir(volumes_dir) if f.endswith('.nii.gz')]
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')]

        if volume_files and mask_files:
            print(f"✅ 找到数据: {len(volume_files)}个volumes, {len(mask_files)}个masks")

            # 优先读取已冻结配置；缺失时才由全部开发掩膜重新计算。
            if os.path.exists(DEFAULT_DEVELOPMENT_THRESHOLD_FILE):
                suggested_threshold = load_frozen_development_threshold()
            else:
                suggested_threshold = analyze_mask_statistics(masks_dir, sample_files=None)

            return volumes_dir, masks_dir, suggested_threshold

    print("❌ 数据路径设置失败")
    return None, None, 1e-6


def find_test_data(mask_threshold=None):
    """查找测试数据，并返回仅由开发队列确定的冻结面积阈值。

    ``mask_threshold`` 可由外部冻结配置显式传入。未传入时，本函数从
    ``./data/masks`` 的全部开发掩膜计算阈值，绝不读取测试掩膜统计量。
    """
    print("查找测试数据...")

    data_folder = './data/test'
    volumes_dir = os.path.join(data_folder, 'volumes')
    masks_dir = os.path.join(data_folder, 'masks')

    if not (os.path.exists(volumes_dir) and os.path.exists(masks_dir)):
        print("⚠️ 未找到独立测试集目录: ./data/test/volumes 和 ./data/test/masks")
        return None, None, [], 1e-6

    test_files = sorted([f for f in os.listdir(volumes_dir) if f.endswith('.nii.gz')])
    if not test_files:
        print("⚠️ 独立测试集目录存在，但没有找到 .nii.gz 文件")
        return volumes_dir, masks_dir, [], 1e-6

    if mask_threshold is None:
        if os.path.exists(DEFAULT_DEVELOPMENT_THRESHOLD_FILE):
            suggested_threshold = load_frozen_development_threshold()
        else:
            development_masks_dir = os.path.join('./data', 'masks')
            if not os.path.exists(development_masks_dir):
                raise FileNotFoundError(
                    '未提供冻结阈值，且未找到开发掩膜目录 ./data/masks；'
                    '拒绝使用测试标注推导评价阈值。'
                )
            suggested_threshold = analyze_mask_statistics(
                development_masks_dir,
                sample_files=None,
            )
    else:
        suggested_threshold = float(mask_threshold)
    print(f"使用开发队列冻结阈值: {suggested_threshold:.10f}")
    print(f"测试文件: {test_files}")
    return volumes_dir, masks_dir, test_files, suggested_threshold


# ==================== 数据增强的类 ====================
class SyncMedicalTransform:
    """同步处理图像和mask的transform"""

    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])

    def __call__(self, image, mask):
        # 生成随机状态
        state = torch.get_rng_state()

        # 应用相同的随机变换到image
        torch.set_rng_state(state)
        image_transformed = self.transforms(image)

        # 应用完全相同的随机变换到mask
        torch.set_rng_state(state)
        mask_transformed = self.transforms(mask)

        return image_transformed, mask_transformed


import torchvision.transforms.functional as F
import torch.nn.functional as TF
import random
from scipy.ndimage import map_coordinates, gaussian_filter
from torchvision.transforms.functional import rotate as tv_rotate


class AdvancedOlfactoryBulbAugmentation:
    """专门针对嗅球分割的增强策略，处理3D张量 [C, H, W]"""

    def __init__(self, target_size=(102, 102)):
        self.target_size = target_size

    def __call__(self, image, mask):
        # 确保输入是3D张量 [channel, height, width]
        if image.dim() == 2:
            image = image.unsqueeze(0)  # [H, W] -> [1, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]

        # 随机翻转
        if random.random() < 0.5:
            image = torch.flip(image, [2])  # 水平翻转 (width维度)
            mask = torch.flip(mask, [2])

        if random.random() < 0.3:
            image = torch.flip(image, [1])  # 垂直翻转 (height维度)
            mask = torch.flip(mask, [1])

        # 随机旋转 (-15° 到 +15°)
        if random.random() < 0.4:
            angle = random.uniform(-15, 15)
            image = self.rotate_tensor(image, angle, is_mask=False)  # 图像用双线性插值
            mask = self.rotate_tensor(mask, angle, is_mask=True)  # 掩码用最近邻插值

        # 弹性变形（对困难样本特别重要）
        if random.random() < 0.3:
            image, mask = self.elastic_deform_fast(image, mask)

        # 亮度对比度调整
        if random.random() < 0.4:
            image = self.adjust_brightness_contrast(image)

        # 添加噪声
        if random.random() < 0.3:
            image = self.add_gaussian_noise(image)

        # 确保输出保持3D张量 [C, H, W]
        return image, mask

    def rotate_tensor(self, tensor, angle, is_mask=False):
        """
        处理3D张量的旋转
        is_mask: 如果是掩码，使用最近邻插值；否则使用双线性插值
        """
        # tensor形状: [C, H, W]
        # 添加批次维度 -> [1, C, H, W]
        tensor_4d = tensor.unsqueeze(0)

        # 根据是否是掩码选择插值方法
        if is_mask:
            interpolation = F.InterpolationMode.NEAREST
            fill_value = 0  # 对于掩码，填充0
        else:
            interpolation = F.InterpolationMode.BILINEAR
            fill_value = tensor.min().item()  # 对于图像，填充最小值

        # 使用torchvision的旋转函数
        rotated = tv_rotate(tensor_4d, angle, interpolation=interpolation, fill=fill_value)

        # 移除批次维度 -> [C, H, W]
        return rotated.squeeze(0)

    def elastic_deform_fast(self, image, mask, alpha=40, sigma=6):
        """
        处理3D张量的快速弹性变形
        输入: image [C, H, W], mask [C, H, W]
        输出: 相同维度的变形后张量
        """
        # 添加批次维度 -> [1, C, H, W]
        image_4d = image.unsqueeze(0)
        mask_4d = mask.unsqueeze(0)

        device = image.device
        B, C, H, W = image_4d.shape

        # 生成随机位移场
        dx = torch.randn(B, 1, H, W, device=device)
        dy = torch.randn(B, 1, H, W, device=device)

        # 应用高斯平滑
        kernel_size = int(4 * sigma + 1) | 1  # 确保是奇数
        dx = TF.avg_pool2d(dx, kernel_size, stride=1, padding=kernel_size // 2)
        dy = TF.avg_pool2d(dy, kernel_size, stride=1, padding=kernel_size // 2)

        # 归一化并缩放
        dx = (dx - dx.mean()) / (dx.std() + 1e-8) * alpha / H
        dy = (dy - dy.mean()) / (dy.std() + 1e-8) * alpha / W

        # 创建标准化网格 [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )

        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

        # 应用位移
        grid_x = grid_x + dx
        grid_y = grid_y + dy
        grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)

        # 应用网格采样
        image_deformed = TF.grid_sample(image_4d, grid, mode='bilinear', padding_mode='border', align_corners=False)
        mask_deformed = TF.grid_sample(mask_4d, grid, mode='nearest', padding_mode='border', align_corners=False)

        # 移除批次维度 -> [C, H, W]
        return image_deformed.squeeze(0), mask_deformed.squeeze(0)

    def adjust_brightness_contrast(self, image):
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        image = image * brightness
        image = (image - image.mean()) * contrast + image.mean()
        return image

    def add_gaussian_noise(self, image, std=0.05):
        noise = torch.randn_like(image) * std
        return image + noise


# ==================== 完整的固定ROI分割流程 ====================
def analyze_mask_statistics(masks_dir, sample_files=None):
    """由开发掩膜的阳性切片前景比例第5百分位确定阈值。

    默认遍历目录中的全部掩膜。``sample_files`` 仅保留给快速调试，正式
    阈值冻结不得设置该参数，也不得把测试掩膜目录传入本函数。
    """
    normalized_parts = os.path.normpath(os.path.abspath(masks_dir)).split(os.sep)
    if any(part.lower() == 'test' for part in normalized_parts):
        # Legacy comparison scripts call this helper with the test-mask path.
        # Keep those test entry points usable without ever inspecting the test
        # annotations: a test path means "load the development-frozen value".
        threshold = load_frozen_development_threshold()
        print(
            '检测到测试掩膜目录；不读取测试标注统计，'
            f'直接使用开发队列冻结阈值: {threshold:.10f}'
        )
        return threshold

    print("分析开发队列mask统计信息...")
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')])

    if not mask_files:
        print("❌ 没有找到mask文件")
        return 1e-6  # 返回默认阈值

    nonzero_ratios = []

    selected_files = mask_files if sample_files is None else mask_files[:sample_files]
    for mask_file in selected_files:
        mask_path = os.path.join(masks_dir, mask_file)
        try:
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()

            for slice_idx in range(mask_data.shape[2]):
                mask_slice = mask_data[:, :, slice_idx]
                nonzero_ratio = np.sum(mask_slice > 0) / mask_slice.size
                if nonzero_ratio > 0:  # 只记录有内容的切片
                    nonzero_ratios.append(nonzero_ratio)

            del mask_img, mask_data
        except Exception as e:
            print(f"警告: 无法读取文件 {mask_file}: {e}")
            continue

    if nonzero_ratios:
        print(f"Mask统计信息:")
        print(f"  - 最小非零比例: {min(nonzero_ratios):.8f}")
        print(f"  - 最大非零比例: {max(nonzero_ratios):.8f}")
        print(f"  - 平均非零比例: {np.mean(nonzero_ratios):.8f}")
        print(f"  - 中位数非零比例: {np.median(nonzero_ratios):.8f}")
        print(f"  - 非零切片数量: {len(nonzero_ratios)}")

        # 计算建议阈值：取最低的5%作为阈值，确保包含所有有意义的切片
        suggested_threshold = max(1e-8, np.percentile(nonzero_ratios, 5))
        print(f"  - 建议阈值: {suggested_threshold:.10f}")

        return suggested_threshold
    else:
        print("❌ 没有找到有效的mask数据")
        return 1e-6  # 返回默认阈值


# ==================== 基于统计的固定ROI定位和裁剪 ====================
class FixedROIDetector:
    """基于统计先验的固定ROI检测器"""

    def __init__(self, relative_size=0.2):
        """
        Args:
            target_roi_size: 目标ROI尺寸
            relative_size: 相对图像尺寸的比例 (基于统计结果0.014-0.018，我们取更大的0.2)
        """
        self.relative_size = relative_size
        self.target_size = (102, 102)

    def detect_fixed_roi(self, image_slice):
        # """基于统计先验检测固定ROI区域"""
        # 水平面，512*512-->102*102
        h, w = image_slice.shape

        # 基于统计结果：平均位置在中心(0.494, 0.497)，我们固定在中心
        center_x, center_y = w // 2, h // 2

        # 基于统计结果：平均相对尺寸(0.014, 0.018)，我们使用更大的固定尺寸
        roi_width = int(w * self.relative_size)
        roi_height = int(h * self.relative_size)

        # 确保最小尺寸?
        # 不需要，直接固定就行

        # 计算边界框
        min_col = max(0, center_x - roi_width // 2)
        max_col = min(w, center_x + roi_width // 2)
        min_row = max(0, center_y - roi_height // 2)
        max_row = min(h, center_y + roi_height // 2)

        # 调整确保方形区域（便于后续处理）
        current_width = max_col - min_col
        current_height = max_row - min_row
        if current_width != current_height:
            # 取较小的一边，然后调整另一边
            size = min(current_width, current_height)
            center_x_adj = (min_col + max_col) // 2
            center_y_adj = (min_row + max_row) // 2

            min_col = center_x_adj - size // 2
            max_col = center_x_adj + size // 2
            min_row = center_y_adj - size // 2
            max_row = center_y_adj + size // 2

            # 再次确保在图像范围内
            min_col = max(0, min_col)
            max_col = min(w, max_col)
            min_row = max(0, min_row)
            max_row = min(h, max_row)

        return (min_col, min_row, max_col, max_row)

    def crop_roi(self, image_slice, roi_box):
        # """根据ROI框裁剪图像"""水平面
        min_col, min_row, max_col, max_row = roi_box
        cropped = image_slice[min_row:max_row, min_col:max_col]
        return cropped


# ==================== 基于固定ROI的数据集 ====================
class FixedROIMRIDataset(Dataset):
    """基于固定ROI的MRI数据集"""

    def __init__(self, volumes_dir, masks_dir, target_roi_size=(102, 102),
                 transform=None, background_ratio=0.1, relative_size=0.2,
                 mask_threshold=1e-6, seed=12345):
        self.volumes_dir = volumes_dir
        self.masks_dir = masks_dir
        self.target_roi_size = target_roi_size
        self.transform = transform
        # 阶段2：添加强度变换
        # self.transform = AdvancedOlfactoryBulbAugmentation()
        self.background_ratio = background_ratio
        self.mask_threshold = mask_threshold
        self.rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed)

        # 使用固定ROI检测器
        self.roi_detector = FixedROIDetector(
            relative_size=relative_size
        )

        # 获取文件列表
        self.volume_files = sorted([f for f in os.listdir(volumes_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')])

        assert len(self.volume_files) == len(self.mask_files), "文件数量不匹配"

        # 预计算切片和ROI信息
        self.slice_info = []
        self.positive_count = 0
        self.negative_count = 0
        self.rejected_count = 0

        print("扫描数据并应用固定ROI区域...")
        print(f"使用mask阈值：{mask_threshold:.10f}")
        for i, volume_file in enumerate(self.volume_files):
            volume_path = os.path.join(volumes_dir, volume_file)
            mask_path = os.path.join(masks_dir, self.mask_files[i])

            try:
                volume_img = nib.load(volume_path)
                mask_img = nib.load(mask_path)

                depth = volume_img.shape[2]

                for slice_idx in range(depth):
                    mask_slice = mask_img.dataobj[:, :, slice_idx]
                    mask_slice = np.array(mask_slice, dtype=np.float32)

                    # 检查是否有目标
                    nonzero_ratio = np.sum(mask_slice>0) / mask_slice.size

                    has_meaningful_target = nonzero_ratio >= self.mask_threshold
                    has_any_target = np.sum(mask_slice>0) > 0

                    # 决定是否包含这个切片
                    # 1. 如果有意义的目标，包含
                    # 2. 如果有任何目标但不够有意义，按概率包含（用于学习困难样本）
                    # 3. 如果没有目标，按background_ratio概率包含
                    if has_meaningful_target or self.rng.random() < self.background_ratio:
                        # 对于所有切片都使用固定的ROI区域
                        volume_slice = volume_img.dataobj[:, :, slice_idx]
                        volume_slice = np.array(volume_slice, dtype=np.float32)
                        roi_box = self.roi_detector.detect_fixed_roi(volume_slice)

                        self.slice_info.append({
                            'volume_idx': i,
                            'slice_idx': slice_idx,
                            'roi_box': roi_box,
                            'has_target': has_meaningful_target
                        })

                        if has_meaningful_target:
                            self.positive_count += 1
                        else:
                            self.negative_count += 1

                del volume_img, mask_img

            except Exception as e:
                print(f"警告: 无法读取文件 {volume_file}: {e}")
                continue

        print(f"固定ROI数据集统计:")
        print(f"  - 总文件: {len(self.volume_files)}")
        print(f"  - 总切片: {len(self.slice_info)}")
        print(f"  - 正样本: {self.positive_count}")
        print(f"  - 负样本: {self.negative_count}")
        print(f"  - 正负比例: {self.positive_count/max(1, self.negative_count):.3f}")
        print(f"  - ROI尺寸: {target_roi_size}")
        print(f"  - 相对尺寸: {relative_size}")

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        info = self.slice_info[idx]
        volume_idx = info['volume_idx']
        slice_idx = info['slice_idx']
        roi_box = info['roi_box']

        volume_file = self.volume_files[volume_idx]
        mask_file = self.mask_files[volume_idx]

        try:
            # 加载数据
            volume_path = os.path.join(self.volumes_dir, volume_file)
            mask_path = os.path.join(self.masks_dir, mask_file)

            volume_img = nib.load(volume_path)
            mask_img = nib.load(mask_path)

            volume_slice = volume_img.dataobj[:, :, slice_idx]
            mask_slice = mask_img.dataobj[:, :, slice_idx]

            volume_slice = np.array(volume_slice, dtype=np.float32)
            mask_slice = np.array(mask_slice, dtype=np.float32)

            del volume_img, mask_img

            # 归一化volume
            if volume_slice.std() > 0:
                volume_slice = (volume_slice - volume_slice.mean()) / volume_slice.std()

            # 二值化mask
            mask_slice = (mask_slice > 0).astype(np.float32)

            # 裁剪固定ROI区域
            volume_roi = self.roi_detector.crop_roi(volume_slice, roi_box)
            # 调整到目标尺寸
            if volume_roi.shape != self.target_roi_size:
                from skimage.transform import resize
                volume_roi = resize(volume_roi, self.target_roi_size, order=1,
                                    preserve_range=True, anti_aliasing=True)

            mask_roi = self.roi_detector.crop_roi(mask_slice, roi_box)
            if mask_roi.shape != self.target_roi_size:
                from skimage.transform import resize
                mask_roi = resize(mask_roi, self.target_roi_size, order=0,
                                  preserve_range=True, anti_aliasing=True)

            # 转换为tensor
            volume_tensor = torch.FloatTensor(volume_roi).unsqueeze(0)
            mask_tensor = torch.FloatTensor(mask_roi).unsqueeze(0)

            # 数据增强
            if self.transform:
                # volume_tensor = self.transform(volume_tensor)
                # mask_tensor = self.transform(mask_tensor)
                volume_tensor, mask_tensor = self.transform(volume_tensor, mask_tensor)
            # print(f"增强后图像范围：[{volume_tensor.min():.3f},{volume_tensor.max():.3f}]")
            # print(f"增强后图像形状：{volume_tensor.shape}")

            return volume_tensor, mask_tensor

        except Exception as e:
            print(f"加载切片失败: {e}")
            # 返回零张量
            return torch.zeros(1, *self.target_roi_size), torch.zeros(1, *self.target_roi_size)


class CenterCropWithPadding:
    """中心裁剪并填充到目标尺寸"""

    def __init__(self, target_size=(102, 102), crop_ratio=0.2):
        self.target_size = target_size
        self.crop_ratio = crop_ratio

    def __call__(self, slice_data, plane_type='axial'):
        """
        根据平面类型进行不同的裁剪和填充策略

        Args:
            slice_data: 输入切片数据
            plane_type: 'axial', 'sagittal', 'coronal'
        """
        h, w = slice_data.shape

        if plane_type == 'axial':
            # 轴向面：中心裁剪到102x102
            return self._center_crop_axial(slice_data)
        elif plane_type == 'sagittal':
            # 矢状面：裁剪到102x20然后填充到102x102
            return self._process_sagittal_coronal(slice_data)
        else:  # coronal
            # 冠状面：裁剪到20x102然后填充到102x102
            return self._process_sagittal_coronal(slice_data, is_coronal=True)

    def _center_crop_axial(self, slice_data):
        """处理轴向面：中心裁剪"""
        h, w = slice_data.shape

        # 计算裁剪区域
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)

        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2

        # 中心裁剪
        cropped = slice_data[start_h:start_h + crop_h, start_w:start_w + crop_w]

        # 调整到目标尺寸
        from skimage.transform import resize
        resized = resize(cropped, self.target_size, order=1,
                         preserve_range=True, anti_aliasing=True)

        return resized

    def _process_sagittal_coronal(self, slice_data, is_coronal=False):
        """处理矢状面或冠状面：裁剪+填充"""
        h, w = slice_data.shape

        if is_coronal:
            # 冠状面：h=512, w=20 → 裁剪到102x20，然后填充到102x102
            crop_h = int(h * self.crop_ratio)  # 512*0.2=102.4≈102
            crop_w = w  # 保持20不变

            start_h = (h - crop_h) // 2
            start_w = 0  # 矢状面宽度很小，不裁剪宽度

            # 中心裁剪高度
            cropped = slice_data[start_h:start_h + crop_h, start_w:start_w + crop_w]

            # 创建102x102的输出，用0填充
            output = np.zeros(self.target_size, dtype=slice_data.dtype)

            # 计算放置位置（水平居中）
            start_w_output = (self.target_size[1] - crop_w) // 2
            output[:crop_h, start_w_output:start_w_output + crop_w] = cropped

            # 调整到目标尺寸
            from skimage.transform import resize
            resized = resize(output, self.target_size, order=1,
                             preserve_range=True, anti_aliasing=True)

        else:
            # 矢状面：h=20, w=512 → 裁剪到20x102，然后填充到102x102
            crop_h = int(h * self.crop_ratio)  # 512*0.2=102.4≈102
            crop_w = w  # 保持20不变

            start_h = (h - crop_h) // 2
            start_w = 0  # 矢状面宽度很小，不裁剪宽度

            # 中心裁剪高度
            cropped = slice_data[start_h:start_h + crop_h, start_w:start_w + crop_w]

            # 创建102x102的输出，用0填充
            output = np.zeros((crop_h, crop_h), dtype=slice_data.dtype)

            # 计算放置位置（水平居中）
            start_w_output = (crop_h - crop_w) // 2
            output[:crop_h, start_w_output:start_w_output + crop_w] = cropped

            # 调整到目标尺寸
            from skimage.transform import resize
            resized = resize(output, self.target_size, order=1,
                             preserve_range=True, anti_aliasing=True)
        return resized


# 综合的三个面的数据集
class MultiPlaneCenterCropDataset(Dataset):
    """使用中心裁剪+填充的多平面数据集"""

    def __init__(self, volumes_dir, masks_dir, target_size=(102, 102),
                 relative_size=0.2, transform=None, background_ratio=0.004, mask_threshold=1e-6,
                 seed=12345):
        self.volumes_dir = volumes_dir
        self.masks_dir = masks_dir
        self.target_size = target_size
        self.relative_size = relative_size
        self.transform = transform
        self.background_ratio = background_ratio
        self.mask_threshold = mask_threshold
        self.rng = np.random.default_rng(seed)

        # 创建裁剪处理器
        self.cropper = CenterCropWithPadding(target_size=target_size, crop_ratio=relative_size)

        # 获取文件列表
        self.volume_files = sorted([f for f in os.listdir(volumes_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')])

        # 预计算切片信息
        self.slice_info = []
        self._precompute_all_planes()

        print(f"数据集统计:")
        print(f"  - 总文件: {len(self.volume_files)}")
        print(f"  - 总切片: {len(self.slice_info)}")
        print(f"  - 目标尺寸: {target_size}")
        print(f"  - 裁剪比例: {relative_size}")

    def _precompute_all_planes(self):
        """预计算三个平面的所有切片"""
        for i, volume_file in enumerate(self.volume_files):
            volume_path = os.path.join(self.volumes_dir, volume_file)
            mask_path = os.path.join(self.masks_dir, self.mask_files[i])

            try:
                volume_img = nib.load(volume_path)
                mask_img = nib.load(mask_path)

                volume_data = volume_img.get_fdata().astype(np.float32)
                mask_data = mask_img.get_fdata().astype(np.float32)

                # 轴向面
                for slice_idx in range(volume_data.shape[2]):
                    mask_slice = mask_data[:, :, slice_idx]
                    if self._should_include_meaningful_slice(mask_slice):
                        self.slice_info.append({
                            'volume_idx': i,
                            'slice_idx': slice_idx,
                            'plane': 'axial'
                        })

                # # 矢状面
                # for slice_idx in range(volume_data.shape[0]):
                #     mask_slice = mask_data[slice_idx, :, :]
                #     if self._should_include_meaningful_slice(mask_slice):
                #         self.slice_info.append({
                #             'volume_idx': i,
                #             'slice_idx': slice_idx,
                #             'plane': 'sagittal'
                #         })

                # # 冠状面
                # for slice_idx in range(volume_data.shape[1]):
                #     mask_slice = mask_data[:, slice_idx, :]
                #     if self._should_include_slice(mask_slice):
                #         self.slice_info.append({
                #             'volume_idx': i,
                #             'slice_idx': slice_idx,
                #             'plane': 'coronal'
                #         })

            except Exception as e:
                print(f"处理文件 {volume_file} 时出错: {e}")

    def _should_include_slice(self, mask_slice):
        """决定是否包含该切片"""
        has_target = np.sum(mask_slice > 0) > 0
        return has_target or self.rng.random() < self.background_ratio

    def _should_include_meaningful_slice(self, mask_slice):
        """决定是否包含该切片"""
        has_meaningful_target = np.sum(mask_slice > 0) / mask_slice.size > self.mask_threshold
        return has_meaningful_target or self.rng.random() < self.background_ratio

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        info = self.slice_info[idx]
        volume_idx = info['volume_idx']
        slice_idx = info['slice_idx']
        plane = info['plane']

        volume_path = os.path.join(self.volumes_dir, self.volume_files[volume_idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[volume_idx])

        volume_img = nib.load(volume_path)
        mask_img = nib.load(mask_path)

        volume_data = volume_img.get_fdata().astype(np.float32)
        mask_data = mask_img.get_fdata().astype(np.float32)

        # 提取对应平面的切片
        if plane == 'axial':
            volume_slice = volume_data[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx]
        elif plane == 'sagittal':
            volume_slice = volume_data[slice_idx, :, :]
            mask_slice = mask_data[slice_idx, :, :]
        else:  # coronal
            volume_slice = volume_data[:, slice_idx, :]
            mask_slice = mask_data[:, slice_idx, :]

        # 使用中心裁剪+填充处理
        volume_processed = self.cropper(volume_slice, plane)
        mask_processed = self.cropper(mask_slice, plane)

        # 归一化
        if volume_processed.std() > 0:
            volume_processed = (volume_processed - volume_processed.mean()) / volume_processed.std()

        # 二值化mask
        mask_processed = (mask_processed > 0).astype(np.float32)

        # 转换为tensor
        volume_tensor = torch.FloatTensor(volume_processed).unsqueeze(0)
        mask_tensor = torch.FloatTensor(mask_processed).unsqueeze(0)

        # 数据增强
        if self.transform:
            volume_tensor, mask_tensor = self.transform(volume_tensor, mask_tensor)

        return volume_tensor, mask_tensor


class TestROIDataset(Dataset):
    """独立的测试数据集，确保与训练数据无重叠"""

    def __init__(self, volumes_dir, masks_dir, test_files=None, target_roi_size=(128, 128),
                 mask_threshold=1e-6):
        self.volumes_dir = volumes_dir
        self.masks_dir = masks_dir
        self.target_roi_size = target_roi_size
        self.mask_threshold = mask_threshold

        # 使用指定的测试文件，或所有文件
        if test_files is None:
            self.volume_files = sorted([f for f in os.listdir(volumes_dir) if f.endswith('.nii.gz')])
        else:
            self.volume_files = [f for f in test_files if f.endswith('.nii.gz')]

        self.mask_files = self.volume_files  # 假设文件名相同

        self.roi_detector = FixedROIDetector()

        # 预计算切片信息（只包含有目标的切片以提高测试效率）
        self.slice_info = []

        print(f"加载测试数据: {len(self.volume_files)} 个volumes")
        for i, volume_file in enumerate(self.volume_files):
            volume_path = os.path.join(volumes_dir, volume_file)
            mask_path = os.path.join(masks_dir, volume_file)

            try:
                volume_img = nib.load(volume_path)
                mask_img = nib.load(mask_path)

                depth = volume_img.shape[2]

                for slice_idx in range(depth):
                    mask_slice = mask_img.dataobj[:, :, slice_idx]
                    mask_slice = np.array(mask_slice, dtype=np.float32)

                    # 只包含有目标的切片进行测试
                    # 检查是否有目标
                    nonzero_ratio = np.sum(mask_slice > 0) / mask_slice.size

                    has_meaningful_target = nonzero_ratio >= self.mask_threshold
                    has_any_target = np.sum(mask_slice > 0) > 0

                    # 决定是否包含这个切片
                    # 1. 如果有意义的目标，包含
                    # 2. 如果有任何目标但不够有意义，按概率包含（用于学习困难样本）
                    # 3. 如果没有目标，按background_ratio概率包含
                    if has_meaningful_target:
                        # 对于所有切片都使用固定的ROI区域
                        volume_slice = volume_img.dataobj[:, :, slice_idx]
                        volume_slice = np.array(volume_slice, dtype=np.float32)
                        roi_box = self.roi_detector.detect_fixed_roi(volume_slice)

                    # if np.sum(mask_slice > 0) > 0:
                    #     volume_slice = volume_img.dataobj[:, :, slice_idx]
                    #     volume_slice = np.array(volume_slice, dtype=np.float32)
                    #
                    #     # 使用固定ROI检测
                    #     roi_box = self.roi_detector.detect_fixed_roi(volume_slice)

                        self.slice_info.append({
                            'volume_idx': i,
                            'slice_idx': slice_idx,
                            'roi_box': roi_box,
                            'volume_file': volume_file,
                            'mask_file': volume_file
                        })

                del volume_img, mask_img

            except Exception as e:
                print(f"警告: 无法读取测试文件 {volume_file}: {e}")
                continue

        print(f"测试集统计:")
        print(f"  - 测试volumes: {len(self.volume_files)}")
        print(f"  - 测试切片: {len(self.slice_info)} (只包含有目标的切片)")

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        info = self.slice_info[idx]
        volume_file = info['volume_file']
        mask_file = info['mask_file']
        slice_idx = info['slice_idx']
        roi_box = info['roi_box']

        try:
            # 加载数据
            volume_path = os.path.join(self.volumes_dir, volume_file)
            mask_path = os.path.join(self.masks_dir, mask_file)

            volume_img = nib.load(volume_path)
            mask_img = nib.load(mask_path)

            volume_slice = volume_img.dataobj[:, :, slice_idx]
            mask_slice = mask_img.dataobj[:, :, slice_idx]

            volume_slice = np.array(volume_slice, dtype=np.float32)
            mask_slice = np.array(mask_slice, dtype=np.float32)

            del volume_img, mask_img

            # 归一化
            if volume_slice.std() > 0:
                volume_slice = (volume_slice - volume_slice.mean()) / volume_slice.std()

            # 二值化mask
            mask_slice = (mask_slice > 0).astype(np.float32)

            # 裁剪ROI区域
            volume_roi = self.roi_detector.crop_roi(volume_slice, roi_box)
            # 调整到目标尺寸
            if volume_roi.shape != self.target_roi_size:
                from skimage.transform import resize
                volume_roi = resize(volume_roi, self.target_roi_size, order=1,
                                 preserve_range=True, anti_aliasing=True)

            mask_roi = self.roi_detector.crop_roi(mask_slice, roi_box)
            if mask_roi.shape != self.target_roi_size:
                from skimage.transform import resize
                mask_roi = resize(mask_roi, self.target_roi_size, order=0,
                                 preserve_range=True, anti_aliasing=True)

            # 转换为tensor
            volume_tensor = torch.FloatTensor(volume_roi).unsqueeze(0)
            mask_tensor = torch.FloatTensor(mask_roi).unsqueeze(0)

            return volume_tensor, mask_tensor

        except Exception as e:
            print(f"加载测试切片失败: {e}")
            return torch.zeros(1, *self.target_roi_size), torch.zeros(1, *self.target_roi_size)
