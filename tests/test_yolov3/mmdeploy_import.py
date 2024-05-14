from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK


img = 'mmdetection/demo/demo.jpg'
work_dir = 'mmdeploy514/tests/test_yolov3//work_dir/onnx/yolov3'
save_file = 'mmdeploy514/tests/test_yolov3//end2end.onnx'
deploy_cfg = 'mmdeploy514/configs/mmdet/detection/detection_onnxruntime_dynamic.py'
model_cfg = 'mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
model_checkpoint = 'mmdeploy514/tests/test_yolov3/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg, model_checkpoint, device)

# 2. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)
