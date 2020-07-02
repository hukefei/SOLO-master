# model settings
model = dict(
    type='SOLO',
    pretrained='/versa/dyy/pretrained_models/RegNetX-800MF_dds_8gpu.pth',
    backbone=dict(
        type='RegNet',
        arch='regnetx_800mf',
        base_channels=32,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 288, 672],
        out_channels=128,
        num_outs=5),
    # neck=dict(
    #     type='BiFPN_Lite',  # P2 ~ P6
    #     is_efficientnet=False,
    #     num_repeats=1,
    #     in_channels=[64, 128, 288, 672],
    #     out_channels=128,
    #     freeze_params=False),
    bbox_head=dict(
        type='SOLOAttentionHead',
        num_classes=2,
        in_channels=128,
        stacked_convs=4,
        seg_feat_channels=128,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        # loss_mask=dict(
        #     type='BCELoss',
        #     loss_weight=2.0),
        # loss_ssim=dict(
        #     type='SSIMLoss',
        #     window_size=11,
        #     size_average=True,
        #     loss_weight=2.0),
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
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(832, 512), (832, 448), (832, 384)],
         multiscale_mode='value',
         keep_ratio=False),
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
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_td.json',
        img_prefix=data_root + 'train_td/',  # '/versa/dataset/COCO2017/coco/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_td.json',
        img_prefix=data_root + 'val_td/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_td.json',
        img_prefix=data_root + 'val_td/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
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
work_dir = './work_dirs/solo_attention_reg800/'
load_from = None  # './work_dirs/solo_attention_bce_td/epoch_12.pth'  # './solo2-lite3_bifpn.pth'
resume_from = None
workflow = [('train', 1)]
