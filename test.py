from ultralytics import YOLO


#加载模型
model=YOLO('yolov9s.pt')
#model = YOLO('yolov8-MobileNetV3.yaml').load('yolov8s.pt')  # 从配置文件初始化模型
#训练模型
model.train(data='yolo-insulator.yaml', workers=0,epochs=100,batch=16)


# 训练模型
