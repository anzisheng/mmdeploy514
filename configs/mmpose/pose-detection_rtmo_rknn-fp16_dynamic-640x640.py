_base_ = ['./pose-detection_static.py', '../_base_/backends/rknn.py']

#onnx_config = dict(input_shape=[320, 320])

onnx_config = dict(
    input_shape=[640, 640],
    output_names=['dets', 'keypoints'],
    ###
    # dynamic_axes={
    #     'input': {
    #         0: 'batch',
    #     },
    #     'dets': {
    #         0: 'batch',
    #     },
    #     'keypoints': {
    #         0: 'batch'
    #     }
    # }###
    )

backend_config = dict(
    #common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640])))
    ])

codebase_config = dict(
    post_processing=dict(
        score_threshold=0.05,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=2000,
        keep_top_k=50,
        background_label_id=-1,
    ))
#backend_config = dict(input_size_list=[[3, 640, 640]])


#need to change star and end
partition_config = dict(
  type='rknn',  # the partition policy name
  apply_marks=True,  # should always be set to True
  partition_cfg=[
      dict(
          save_file='model.onnx',  # name to save the partitioned onnx
          start=['pose_detector_forward:input'],  # [mark_name:input, ...]
          end=['yolo_head:input'],  # [mark_name:output, ...]
          output_names=[f'pred_maps.{i}' for i in range(3)]) # output names
  ])