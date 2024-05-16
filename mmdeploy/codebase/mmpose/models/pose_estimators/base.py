# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape
import torch
from mmdet.structures import DetDataSample

from typing import Dict, List, Optional, Tuple, Union
from mmpose.structures import PoseDataSample

from mmdeploy.core import FUNCTION_REWRITER, mark
# Type hint of data samples
SampleList = List[PoseDataSample]
OptSampleList = Optional[SampleList]


@mark(    'pose_detector_forward', inputs=['input'], outputs=['dets', 'labels', 'masks'])
def __forward_impl(self, batch_inputs, data_samples):
    """Rewrite and adding mark for `forward`.

    Encapsulate this function for rewriting `forward` of BasePoseEstimator.
    1. Add mark for BasePoseEstimator.
    2. Support both dynamic and static export to onnx.
    """

    x = self.extract_feat(batch_inputs)

    #output = self.bbox_head.predict(x, data_samples, rescale=False)
    output = self.head.predict(x, data_samples, {}) #anzisheng, rescale=False)
    #output = self.RTMOHead.predict(x,  rescale=False)
    #output = self.head.predict(x, None)#,rescale=False)




@torch.fx.wrap
def _set_metainfo(data_samples, img_shape):
    """Set the metainfo.

    Code in this function cannot be traced by fx.
    """
    if data_samples is None:
        data_samples = [DetDataSample()]

    # note that we can not use `set_metainfo`, deepcopy would crash the
    # onnx trace.
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')

    return data_samples

@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.pose_estimators.base.BasePoseEstimator.forward')
def base_pose_estimator__forward(self, inputs, *args, **kwargs):
    """Rewrite `forward` of TopDown for default backend.'.

    1.directly call _forward of subclass.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (BasePoseEstimator): The instance of the class Object
            BasePoseEstimator.
        inputs (torch.Tensor[NxCxHxW]): Input images.

    Returns:
        torch.Tensor: The predicted heatmaps.
    """
    print("oooooooooooooooooo")
    #
    # ctx = FUNCTION_REWRITER.get_context()
    #
    # deploy_cfg = ctx.cfg
    #
    # # get origin input shape as tensor to support onnx dynamic shape
    # is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # img_shape = torch._shape_as_tensor(inputs)[2:]
    # if not is_dynamic_flag:
    #     img_shape = [int(val) for val in img_shape]
    #
    # # set the metainfo
    # data_samples = _set_metainfo(data_samples, img_shape)
    #
    # return __forward_impl(self, inputs, data_samples=data_samples)
    #
    #




    return self._forward(inputs)
