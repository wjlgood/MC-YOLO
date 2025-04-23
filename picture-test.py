from ultralytics import YOLO
import os
import cv2

model_path = 'runs/detect/train98/weights/best.pt'  # 模型路径（.pt 文件）
image_path = 'image_folder'  # 单张图片或文件夹路径
conf_thres = 0.2  # 置信度阈值
imgsz = 640  # 输入图片尺寸

# ========== 加载模型 ==========
model = YOLO(model_path)

# ========== 执行推理 ==========
results = model.predict(
    source=image_path,
    conf=conf_thres,
    imgsz=imgsz,
    save=True,       # 保存预测图像
    save_txt=True,   # 保存标签（txt）
    save_conf=True   # 保存置信度
)

for result in results:
    print(f"\n图片: {result.path}")
    if result.boxes is None or len(result.boxes) == 0:
        print("⚠️ 没有检测到任何目标")
    else:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"目标 {i+1}：类别 {cls_id}, 置信度 {conf:.2f}, 坐标 {xyxy}")

print("\n✅ 推理完成，结果保存在 runs/detect/predict/")
