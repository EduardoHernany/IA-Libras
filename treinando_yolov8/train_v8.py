from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# para marcar as imagens
# https://www.makesense.ai/

def main():
    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8m-cls.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="../datasets", epochs=150, device=0, imgsz=64, batch=64, hsv_h=0.1)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print("path", path)


if __name__ == '__main__':
    # freeze_support()
    main()
