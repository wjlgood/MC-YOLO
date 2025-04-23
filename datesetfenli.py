import os
import shutil
import random


def split_dataset(src_images_dir, src_labels_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 检查数据集文件夹是否存在
    if not os.path.exists(src_images_dir) or not os.path.exists(src_labels_dir):
        print("图片或标注文件夹不存在！")
        return

    # 获取所有标注文件
    label_files = [f for f in os.listdir(src_labels_dir) if f.endswith('.txt')]
    random.shuffle(label_files)  # 随机打乱文件顺序

    # 计算各数据集的大小
    total_files = len(label_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size  # 剩余的分配给测试集

    # 创建目标文件夹
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    test_dir = os.path.join(dest_dir, 'test')

    for dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
            os.makedirs(os.path.join(dir, 'images'))
            os.makedirs(os.path.join(dir, 'labels'))

    # 定义划分
    train_files = label_files[:train_size]
    val_files = label_files[train_size:train_size + val_size]
    test_files = label_files[train_size + val_size:]

    # 拷贝文件到目标文件夹
    def copy_files(file_list, dest_images_dir, dest_labels_dir):
        for label_file in file_list:
            # 获取图片文件路径
            label_path = os.path.join(src_labels_dir, label_file)
            img_file = label_file.replace('.txt', '.jpg')  # 假设图片格式为.jpg
            img_path = os.path.join(src_images_dir, img_file)

            if os.path.exists(img_path):  # 如果图片文件存在
                # 拷贝标注文件
                shutil.copy(label_path, dest_labels_dir)
                # 拷贝图片文件
                shutil.copy(img_path, dest_images_dir)

    # 拷贝训练集、验证集、测试集文件
    copy_files(train_files, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
    copy_files(val_files, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))
    copy_files(test_files, os.path.join(test_dir, 'images'), os.path.join(test_dir, 'labels'))

    print("数据集划分完成：")
    print(f"训练集：{len(train_files)}")
    print(f"验证集：{len(val_files)}")
    print(f"测试集：{len(test_files)}")


# 使用示例
src_images_dir = 'D:/ultralytics-main/datesets/images'  # 图片文件夹路径
src_labels_dir = 'D:/ultralytics-main/datesets/labels'  # YOLO标签文件夹路径
dest_dir = '/datasets/insulator'  # 目标文件夹路径

split_dataset(src_images_dir, src_labels_dir, dest_dir)



