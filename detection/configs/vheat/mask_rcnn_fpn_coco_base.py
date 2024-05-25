_base_ = [
    './swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MMDET_VHEAT',
        post_norm=True,
        img_size=512,
        layer_scale=1.e-5, 
        drop_path_rate=0.5,
        depths=(2, 2, 18, 2),
        dims=128,
        out_indices=(0, 1, 2, 3),
        pretrained="path/to/pth",
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)
train_dataloader = dict(batch_size=2) # as gpus=8
val_dataloader = dict(batch_size=2)

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

