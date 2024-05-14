#pose-detection_rtmo_rknn_dynamic.py  mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py mmpose/weight/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth mmpose/data/test/multi-person.jpeg --work-dir results-rtmo_test --dump-info  --show --device cpu
_base_ = ['./pose-detection_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(
    output_names=['dets', 'keypoints'],
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'dets': {
            0: 'batch',
        },
        'keypoints': {
            0: 'batch'
        }
    })

codebase_config = dict(
    post_processing=dict(
        score_threshold=0.05,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=2000,
        keep_top_k=50,
        background_label_id=-1,
    ))
