import torch
from mmdet.apis import init_detector, inference_detector


config_file = '/home/an/project/mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
torch.onnx.export(model, (torch.zeros(1, 3, 416, 416),), "yolov3.onnx", opset_version=11)
