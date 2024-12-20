# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
#from .single_stage import SingleStageDetector
from .singlestage_heatmap import SingleStageDetectorHP


@DETECTORS.register_module()
#class GFL(SingleStageDetector):
class GFL(SingleStageDetectorHP):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GFL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
