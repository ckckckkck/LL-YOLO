from ultralytics import YOLO
model_yaml_path = "ultralytics/cfg/models/11/yolo11n.yaml"
data_yaml_path = 'RDDChina-low-light.yaml'

if __name__ == '__main__':

    model = YOLO (model_yaml_path)
    #model.load(pre_model_name)  # loading pretrain weights
    # 训练模型
    model.train(data=data_yaml_path,
                          epochs=500,
                          batch=16,
                          device=0,
                          optimizer = 'SGD',
                          project = 'runs/添加噪声实验3',
                          patience = 100,
                          imgsz = 640,
)
