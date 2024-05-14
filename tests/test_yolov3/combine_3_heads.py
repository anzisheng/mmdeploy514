import torch
from itertools import repeat
from mmdet.apis import init_detector, inference_detector

config_file = '/home/an/project/mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'


class YOLOV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_detector(config_file, checkpoint_file, device='cpu')
        self.class_num = 80
        self.base_sizes = [[(116, 90), (156, 198), (373, 326)],
                           [(30, 61), (62, 45), (59, 119)],
                           [(10, 13), (16, 30), (33, 23)]]
        self.stride = [32, 16, 8]
        self.strides = [tuple(repeat(x, 2)) for x in self.stride]
        self.centers = [(x[0] / 2., x[1] / 2.) for x in self.strides]
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = self.centers[i]
            x_center, y_center = center
            base_anchors = []
            for base_size in base_sizes_per_level:
                w, h = base_size
                base_anchor = torch.Tensor(
                    [x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w, y_center + 0.5 * h])
                base_anchors.append(base_anchor)
            base_anchors = torch.stack(base_anchors, dim=0)
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def _meshgrid(self, x, y):
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        return xx, yy

    def grid_priors(self, featmap_sizes):
        multi_level_anchors = []
        for i in range(len(featmap_sizes)):
            base_anchors = self.base_anchors[i]
            feat_h, feat_w = featmap_sizes[i]
            stride_w, stride_h = self.strides[i]
            shift_x = torch.arange(0, feat_w) * stride_w
            shift_y = torch.arange(0, feat_h) * stride_h
            shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
            shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
            anchors = base_anchors[None, :, :] + shifts[:, None, :]
            anchors = anchors.view(-1, 4)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def decode(self, bboxes, pred_bboxes, stride):
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (pred_bboxes[..., :2] - 0.5) * stride
        whs = (bboxes[..., 2:] - bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
        decoded_bboxes = torch.stack((xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] - whs[..., 1],
                                      xy_centers[..., 0] + whs[..., 0], xy_centers[..., 1] + whs[..., 1]), dim=-1)
        return decoded_bboxes

    def forward(self, x):
        x = self.model.backbone(x)
        x = self.model.neck(x)
        pred_maps = self.model.bbox_head(x)

        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps[0], self.stride):
            pred = pred.permute(0, 2, 3, 1).reshape(1, -1, 5 + self.class_num)
            pred[..., :2] = pred[..., :2].sigmoid()
            flatten_preds.append(pred)
            flatten_strides.append(pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_preds[..., 4] = flatten_objectness
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
        flatten_preds[..., 5:] = flatten_cls_scores

        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps[0]]
        mlvl_anchors = self.grid_priors(featmap_sizes)
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)

        flatten_bboxes = self.decode(flatten_anchors, flatten_bbox_preds, flatten_strides.unsqueeze(-1))
        flatten_preds[..., :4] = flatten_bboxes

        return flatten_preds


model = YOLOV3().eval()
input = torch.zeros(1, 3, 416, 416, device='cpu')
torch.onnx.export(model, input, "yolov3_combined.onnx", opset_version=11)
