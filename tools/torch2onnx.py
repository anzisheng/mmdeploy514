# Copyright (c) OpenMMLab. All rights reserved.
#partition for yolov3
#configs/mmdet/detection/yolov3_partition_onnxruntime_static.py ../mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py  https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth  /home/an/project/mmdetection/demo/demo.jpg --work-dir ./work-dirs/mmdet/yolov3/ort/partition

import argparse
import logging
import os
import os.path as osp

from mmdeploy.apis import (extract_model, get_predefined_partition_cfg,
                           torch2onnx)
from mmdeploy.utils import (get_ir_config, get_partition_config,
                            get_root_logger, load_config)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img', help='image used to convert model model')
    parser.add_argument(
        '--work-dir',
        default='./work-dir',
        help='Directory to save output files.')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger = get_root_logger(log_level=args.log_level)

    logger.info(f'torch2onnx: \n\tmodel_cfg: {args.model_cfg} '
                f'\n\tdeploy_cfg: {args.deploy_cfg}')

    os.makedirs(args.work_dir, exist_ok=True)
    # load deploy_cfg
    deploy_cfg = load_config(args.deploy_cfg)[0]
    save_file = get_ir_config(deploy_cfg)['save_file']

    torch2onnx(
        args.img,
        args.work_dir,
        save_file,
        deploy_cfg=args.deploy_cfg,
        model_cfg=args.model_cfg,
        model_checkpoint=args.checkpoint,
        device=args.device)

    # partition model
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:
        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = osp.join(args.work_dir, save_file)
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)
    logger.info(f'torch2onnx finished. Results saved to {args.work_dir}')


if __name__ == '__main__':
    main()
