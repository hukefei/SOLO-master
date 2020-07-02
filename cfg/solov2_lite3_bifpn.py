# fp16 settings
# fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='SOLO',
    pretrained='/versa/dyy/pretrained_models/tf_efficientnet_lite3_tuned.pth',
    backbone=dict(
        type='EfficientNet_Lite',
        model_name='efficientnet-b3',
        num_stages=7,
        out_indices=(1, 2, 4, 6),  # C2, C3, C4, C5
        frozen_stages=7),
    neck=dict(
        type='BiFPN_Lite',  # P2 ~ P6
        compound_coef=3,
        num_repeats=1,
        out_channels=160),
    bbox_head=dict(
        type='SOLOV2Head',
        num_classes=5,
        in_channels=160,
        stacked_convs=2,
        seg_feat_channels=160,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        with_deform=False,
        # fp16_training=True,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
    ))
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/versa/dyy/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(
    #     type='InstaBoost',
    #     action_candidate=('normal', 'horizontal', 'skip'),
    #     action_prob=(1, 0, 0),
    #     scale=(0.8, 1.2),
    #     dx=15,
    #     dy=15,
    #     theta=(-1, 1),
    #     color_prob=0.5,
    #     hflag=False,
    #     aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resizer', img_scale=[832, 768, 704, 640, 576, 512]),
    dict(type='Resize',
         img_scale=[(832, 512), (832, 448), (832, 384)],
         # img_scale=[(832, 832), (768, 768), (704, 704),
         #            (640, 640), (576, 576), (512, 512)],
         multiscale_mode='value',
         keep_ratio=False),
    # dict(type='RandomCrop', crop_size=(384, 384)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(832, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            # dict(type='Resizer'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=24,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix='/versa/dataset/COCO2017/coco/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/solov2_lite3_bifpn_bn'
load_from = None
resume_from = None
workflow = [('train', 1)]
