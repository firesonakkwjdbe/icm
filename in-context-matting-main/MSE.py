import os
import cv2  # 用于读取图像
import numpy as np

# 定义文件夹路径
predicted_folder = r"C:\selfinfor\alphapredic\in-context-matting-main\result\results3"  # 存储预测alpha蒙版的文件夹
ground_truth_folder = r"C:\selfinfor\alphapredic\in-context-matting-main\datasets\ICM57\alpha"  # 存储真实alpha蒙版的文件夹

# 定义计算MSE的函数
def calculate_mse(predicted, ground_truth):
    """
    计算两个图像之间的均方误差 (MSE)
    :param predicted: 预测的alpha蒙版 (numpy array)
    :param ground_truth: 真实的alpha蒙版 (numpy array)
    :return: MSE值
    """
    return np.mean((predicted - ground_truth) ** 2)

# 遍历文件夹中的文件
predicted_files = sorted(os.listdir(predicted_folder))  # 按名称排序，确保文件匹配
ground_truth_files = sorted(os.listdir(ground_truth_folder))

# 检查预测和真实文件数量是否一致
if len(predicted_files) != len(ground_truth_files):
    raise ValueError("预测文件和真实文件的数量不一致，请检查文件夹内容！")

# 初始化MSE列表
mse_list = []

# 逐一读取文件并计算MSE
for pred_file, gt_file in zip(predicted_files, ground_truth_files):
    # 构建文件路径
    pred_path = os.path.join(predicted_folder, pred_file)
    gt_path = os.path.join(ground_truth_folder, gt_file)

    # 读取预测和真实alpha蒙版
    pred_alpha = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    gt_alpha = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)      # 读取为灰度图

    # 确保图像大小一致
    if pred_alpha.shape != gt_alpha.shape:
        raise ValueError(f"图像尺寸不一致：{pred_file} 和 {gt_file}")

    # 归一化图像到[0, 1]范围（假设输入值为0-255）
    pred_alpha = pred_alpha / 255.0
    gt_alpha = gt_alpha / 255.0

    # 计算MSE
    mse = calculate_mse(pred_alpha, gt_alpha)
    mse_list.append(mse)

    # 打印每张图像的MSE
    print(f"图像 {pred_file} 的 MSE: {mse:.6f}")

# 计算所有图像的平均MSE
average_mse = np.mean(mse_list)
print(f"\n测试集上的平均MSE: {average_mse:.6f}")
