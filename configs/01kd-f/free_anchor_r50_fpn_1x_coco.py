_base_ = './retinanet_r18_fpn_1x_coco.py'
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='FreeAnchorRetinaHead',
        num_classes=11,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75)))


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
#load_from = 'checkpoints/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'

dataset_type = 'CocoDataset'
classes = ( 'baliser_ok', 'baliser_aok', 'baliser_nok', 'insulator_ok', 'insulator_nok', 'bird_nest', 'stockbridge_ok', 'stockbridge_nok', 'spacer_ok', 'spacer_nok', 'insulator_unk')


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
        img_prefix='data/coco-Fu/val2017/',
        classes=classes,
        ann_file='data/coco-Fu/annotations/instances_val2017.json'))
