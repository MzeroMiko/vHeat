_base_ = [
    './swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MMDET_VHEAT',
        post_norm=True,
        img_size=512,
        layer_scale=1.e-5, 
        drop_path_rate=0.3,
        depths=(2, 2, 18, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="path/to/pth",
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
)
train_dataloader = dict(batch_size=2) # as gpus=8
val_dataloader = dict(batch_size=2)

