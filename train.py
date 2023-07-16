import os
from datetime import datetime
from PIL import Image
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from siren import SIREN
from utils import set_logger

SAMPLING_RATIO = 0.1  # 采样比例
BATCH_SIZE = 8192  # 批量大小
EPOCHS = 5000  # 训练轮数
LEARNING_RATE = 0.0005  # 学习率

img_filepath = 'data/celtic_spiral_knot.jpg'  # 图片文件路径
img_raw = np.array(Image.open(img_filepath))  # 读取图片并转换为numpy数组
img_ground_truth = torch.from_numpy(img_raw).float()  # 将numpy数组转换为torch张量

rows, cols, channels = img_ground_truth.shape  # 获取图片的行数、列数和通道数
pixel_count = rows * cols  # 像素总数
sampled_pixel_count = int(pixel_count * SAMPLING_RATIO)  # 采样像素数

def build_train_tensors():
    img_mask_x = torch.from_numpy(
        np.random.randint(0, rows, sampled_pixel_count))  # 从0到rows之间随机生成sampled_pixel_count个整数，构建img_mask_x
    img_mask_y = torch.from_numpy(
        np.random.randint(0, cols, sampled_pixel_count))  # 从0到cols之间随机生成sampled_pixel_count个整数，构建img_mask_y

    img_train = img_ground_truth[img_mask_x, img_mask_y]  # 根据img_mask_x和img_mask_y从img_ground_truth中获取对应的像素值，构建img_train

    img_mask_x = img_mask_x.float() / rows  # 将img_mask_x归一化到[0, 1]范围
    img_mask_y = img_mask_y.float() / cols  # 将img_mask_y归一化到[0, 1]范围

    img_mask = torch.stack([img_mask_x, img_mask_y], dim=-1)  # 将img_mask_x和img_mask_y堆叠在一起，构建img_mask

    return img_mask, img_train

img_mask, img_train = build_train_tensors()  # 构建训练集的输入和目标张量

train_dataset = TensorDataset(img_mask, img_train)  # 构建训练集的数据集
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)  # 构建训练集的数据加载器

layers = [256, 256, 256, 256, 256]  # 模型的隐藏层大小列表
in_features = 2  # 输入特征数
out_features = 3  # 输出特征数
initializer = 'siren'  # 初始化器类型
w0 = 1.0  # 初始权重
w0_initial = 30.0  # 初始权重（对于第一层）
c = 6  # 均匀分布的上下界常数
model = SIREN(
    layers, in_features, out_features, w0, w0_initial,
    initializer=initializer, c=c)  # 构建SIREN模型

model.train()  # 将模型设置为训练模式

BATCH_SIZE = min(BATCH_SIZE, len(img_mask))  # 批量大小不超过样本数
num_steps = int(len(img_mask) * EPOCHS / BATCH_SIZE)  # 计算总的训练步数
print("Total training steps : ", num_steps)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 使用Adam优化器
criterion = torch.nn.MSELoss()  # 使用均方误差损失函数

checkpoint_dir = 'checkpoints/siren/inpainting/'  # 检查点保存的目录
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')  # 当前时间戳
logdir = os.path.join('logs/siren/inpainting/', timestamp)  # 日志保存的目录

if not os.path.exists(logdir):
    os.makedirs(logdir)

set_logger(os.path.join(logdir, 'train.log'))  # 设置日志记录器

best_loss = np.inf  # 最佳损失初始化为无穷大

for epoch in range(EPOCHS):  # 迭代训练轮数
    iterator = tqdm(train_dataloader, dynamic_ncols=True)  # 构建进度条迭代器

    losses = []  # 存储每个batch的损失值

    for batch in iterator:
        inputs, targets = batch  # 获取输入和目标张量
        predictions = model(inputs)  # 模型前向传播得到预测值
        loss = criterion(predictions, targets)  # 计算损失值
        losses.append(loss.reshape(-1))  # 将损失值添加到损失列表中

        optimizer.zero_grad()  # 梯度置零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        iterator.set_description(
            "Epoch: {} | Loss {:.4f}".format(epoch, loss), refresh=True)  # 更新进度条显示信息

    avg_loss = torch.mean(torch.cat(losses)).item()  # 计算平均损失值
    logging.info("Epoch: {} | Avg. Loss {:.4f}".format(epoch, avg_loss))  # 输出平均损失值到日志

    if avg_loss < best_loss:  # 如果平均损失值优于最佳损失值
        logging.info('Loss improved from {:.4f} to {:.4f}'.format(
            best_loss, avg_loss))  # 输出改善的信息到日志
        best_loss = avg_loss  # 更新最佳损失值
        torch.save(
            {'network': model.state_dict()},
            os.path.join(checkpoint_dir + 'model'))  # 保存模型的状态字典到文件
