_base_ = [
    './configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MMSEG_VHEAT',
        img_size=512,
        post_norm=False, 
        drop_path_rate=0.3,
        depths=(2, 2, 6, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="path/to/pth",
    ),
    decode_head=dict(num_classes=150, in_channels=[96, 192, 384, 768]),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

train_dataloader = dict(batch_size=2) # as gpus=8
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

