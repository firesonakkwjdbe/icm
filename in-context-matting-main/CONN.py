import os
import cv2
import numpy as np
from scipy.ndimage import label

# 定义文件夹路径
predicted_folder = r"C:\selfinfor\alphapredic\in-context-matting-main\result\results3"  # 存储预测alpha蒙版的文件夹
ground_truth_folder = r"C:\selfinfor\alphapredic\in-context-matting-main\datasets\ICM57\alpha"  # 存储真实alpha蒙版的文件夹

# 定义CONN计算函数
def calculate_conn(predicted, ground_truth, threshold=0.5):
    """
    计算两个图像之间的CONN（Connectivity Error）
    :param predicted: 预测的alpha蒙版 (numpy array)
    :param ground_truth: 真实的alpha蒙版 (numpy array)
    :param threshold: 二值化阈值 (默认0.5)
    :return: CONN值
    """
    # 二值化处理（前景为1，背景为0）
    pred_binary = (predicted >= threshold).astype(np.uint8)
    gt_binary = (ground_truth >= threshold).astype(np.uint8)

    # 使用 8-连通性标记前景区域
    pred_labeled, pred_num_features = label(pred_binary, structure=np.ones((3, 3)))
    gt_labeled, gt_num_features = label(gt_binary, structure=np.ones((3, 3)))

    # 统计预测和真实的连通区域得分
    pred_scores = np.zeros(pred_num_features + 1)
    gt_scores = np.zeros(gt_num_features + 1)

    for i in range(1, pred_num_features + 1):
        pred_scores[i] = np.sum(pred_binary[pred_labeled == i])
    for i in range(1, gt_num_features + 1):
        gt_scores[i] = np.sum(gt_binary[gt_labeled == i])

    # 计算连接性误差
    conn_error = 0.0
    for i in range(1, gt_num_features + 1):
        max_overlap = 0
        for j in range(1, pred_num_features + 1):
            overlap = np.sum((gt_labeled == i) & (pred_labeled == j))
            max_overlap = max(max_overlap, overlap)
        conn_error += gt_scores[i] - max_overlap

    return conn_error / np.sum(gt_scores)

# 遍历文件夹中的文件
predicted_files = sorted(os.listdir(predicted_folder))  # 按名称排序，确保文件匹配
ground_truth_files = sorted(os.listdir(ground_truth_folder))

# 检查预测和真实文件数量是否一致
if len(predicted_files) != len(ground_truth_files):
    raise ValueError("预测文件和真实文件的数量不一致，请检查文件夹内容！")

# 初始化CONN列表
conn_list = []

# 逐一读取文件并计算CONN
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

    # 计算CONN
    conn = calculate_conn(pred_alpha, gt_alpha)
    conn_list.append(conn)

    # 打印每张图像的CONN
    print(f"图像 {pred_file} 的 CONN: {conn:.6f}")

# 计算所有图像的平均CONN
average_conn = np.mean(conn_list)
print(f"\n测试集上的平均CONN: {average_conn:.6f}")
