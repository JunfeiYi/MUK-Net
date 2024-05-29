_base_ = [
    '../../configs/01kd-f/gfl_r18_fpn_1x_coco.py'
]
# model settings
find_unused_parameters=True
temp=0.5
alpha_fgd=0.001
beta_fgd=0.0005
gamma_fgd=0.0005
lambda_fgd=0.000005
distiller = dict(
    type='DetectionDistiller',
    #teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth',
    teacher_pretrained = 'checkpoints/Fu-epoch_12.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/01kd-f/gfl_r18_fpn_1x_coco.py'
teacher_cfg = 'configs/01kd-f/gfl_r101_fpn_1x_coco.py'
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='CutOut', n_holes=(3), cutout_ratio=(0.01,0.02,0.04,0.06,0.08,0.1)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

classes = ( 'baliser_ok', 'baliser_aok', 'baliser_nok', 'insulator_ok', 'insulator_nok', 'bird_nest', 'stockbridge_ok', 'stockbridge_nok', 'spacer_ok', 'spacer_nok', 'insulator_unk')


#classes = ('bicycle', 'car', 'motorcycle',)
#classes = ( 'pinsub','pinsug','pinsumiss','plinkgrp','pnest','psusp','pvib','pvibmiss')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        img_prefix='data/coco-Fu/train2017/',
        classes=classes,
        ann_file='data/coco-Fu/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='data/coco-Fu/val2017/',
        classes=classes,
        ann_file='data/coco-Fu/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='data/coco-Fu/test2017/',
        classes=classes,
        ann_file='data/coco-Fu/annotations/instances_test2017.json'))
