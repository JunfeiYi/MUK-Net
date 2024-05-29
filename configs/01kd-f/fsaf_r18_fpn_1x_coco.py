_base_ = './retinanet_r18_fpn_1x_coco.py'
# model settings
model = dict(
    type='FSAF',
    bbox_head=dict(
        type='FSAFHead',
        num_classes=11,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        # Only anchor-free branch is implemented. The anchor generator only
        #  generates 1 anchor at each feature point, as a substitute of the
        #  grid of features.
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(_delete_=True, type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(
            _delete_=True,
            type='IoULoss',
            eps=1e-6,
            loss_weight=1.0,
            reduction='none')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='CenterRegionAssigner',
            pos_scale=0.2,
            neg_scale=0.2,
            min_pos_iof=0.01),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
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
    
