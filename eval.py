import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示进度条
import torch
from torch.utils.data import TensorDataset, DataLoader
from siren import SIREN

BATCH_SIZE = 8192   # 批量大小，可以自己根据环境调节

img_filepath = 'data/plant.jpg'   # 图像文件路径
img_raw = np.array(Image.open(img_filepath))   # 读取图像并转换为NumPy数组
img_ground_truth = torch.from_numpy(img_raw).float()   # 将NumPy数组转换为FloatTensor

rows, cols, channels = img_ground_truth.shape   # 图像的行数、列数和通道数
pixel_count = rows * cols   # 图像像素总数


def build_eval_tensors():
    img_mask_x = np.arange(0, rows)   # 构建横坐标掩码
    img_mask_y = np.arange(0, cols)   # 构建纵坐标掩码

    img_mask_x, img_mask_y = np.meshgrid(img_mask_x, img_mask_y, indexing='ij')   # 利用掩码坐标生成网格
    img_mask_x = torch.from_numpy(img_mask_x)   # 转换为张量
    img_mask_y = torch.from_numpy(img_mask_y)   # 转换为张量

    img_mask_x = img_mask_x.float() / rows   # 归一化横坐标
    img_mask_y = img_mask_y.float() / cols   # 归一化纵坐标

    img_mask = torch.stack([img_mask_x, img_mask_y], dim=-1)   # 合并横纵坐标
    img_mask = img_mask.reshape(-1, 2)   # 调整形状为二维张量
    img_eval = img_ground_truth.reshape(-1, 3)   # 调整形状为二维张量

    return img_mask, img_eval

img_mask, img_eval = build_eval_tensors()   # 构建输入和目标张量


test_dataset = TensorDataset(img_mask, img_eval)   # 构建测试数据集
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)   # 构建数据加载器

# 构建模型
layers = [256, 256, 256, 256, 256]
in_features = 2
out_features = 3
initializer = 'siren'
w0 = 1.0
w0_initial = 30.0
c = 6
model = SIREN(
    layers, in_features, out_features, w0, w0_initial,
    initializer=initializer, c=c)

# 恢复模型
checkpoint_path = 'checkpoints/siren/inpainting/model'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError('Checkpoint not found at {}'.format(
        checkpoint_path))

# 加载检查点（模型权重）
print('Loading model checkpoint from {}'.format(checkpoint_path))
ckpt = torch.load(checkpoint_path)
model.load_state_dict(ckpt['network'])   # 加载模型权重
model.eval()   # 设置为评估模式

iterator = tqdm(test_dataloader)   # 创建迭代器并显示进度条

predictions = []   # 存储预测结果

for batch in iterator:
    inputs, _ = batch   # 获取输入

    with torch.no_grad():
        prediction = model(inputs)   # 模型预测

    predictions.append(prediction)   # 将预测结果添加到列表中

predicted_image = torch.cat(predictions).cpu().numpy()   # 将预测的图像转换为NumPy数组
predicted_image = predicted_image.reshape((rows, cols, channels)) / 255   # 调整形状并进行归一化
predicted_image = predicted_image.clip(0.0, 1.0)   # 将像素值裁剪到0-1范围内

img_save_path = 'images/celtic_spiral_knot.jpg'   # 保存图像的路径
os.makedirs(os.path.dirname(img_save_path), exist_ok=True)   # 确保保存路径存在

fig, axes = plt.subplots(1, 2)   # 创建图形窗口和子图
plt.sca(axes[0])
plt.imshow(img_ground_truth.numpy() / 255)   # 显示原始图像
plt.title("Ground Truth Image")

plt.sca(axes[1])
plt.imshow(predicted_image)   # 显示预测图像
plt.title("Predicted Image")

fig.tight_layout()   # 调整子图布局
plt.savefig(img_save_path, bbox_inches='tight')   # 保存图像
plt.show()   # 显示图像