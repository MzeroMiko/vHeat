_base_ = [
    './swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MMDET_VHEAT',
        drop_path=0.1,
        post_norm=False,
        depths=(2, 2, 6, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        img_size=512,
        pretrained="path/to/pth",
    ),
)

train_dataloader = dict(batch_size=2) # as gpus=8
val_dataloader = dict(batch_size=2)
