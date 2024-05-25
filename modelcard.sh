######################
# classification
######################
torchrun --nproc_per_node 16 main.py --cfg configs/vHeat/vHeat_tiny_224.yaml --batch-size 128 --data-path /path/to/dataset --output output/vHeat_tiny
torchrun --nproc_per_node 16 main.py --cfg configs/vHeat/vHeat_small_224.yaml --batch-size 128 --data-path /path/to/dataset --output output/vHeat_small
torchrun --nproc_per_node 16 main.py --cfg configs/vHeat/vHeat_base_224.yaml --batch-size 128 --data-path /path/to/dataset --output output/vHeat_base

######################
# detection
######################
bash tools/dist_train.sh configs/vheat/mask_rcnn_fpn_coco_tiny.py 8
bash tools/dist_train.sh configs/vheat/mask_rcnn_fpn_coco_small.py 8
bash tools/dist_train.sh configs/vheat/mask_rcnn_fpn_coco_base.py 8
bash tools/dist_train.sh configs/vheat/mask_rcnn_fpn_coco_tiny_ms_3x.py 8
bash tools/dist_train.sh configs/vheat/mask_rcnn_fpn_coco_small_ms_3x.py 8

######################
# segmentation
######################
bash tools/dist_train.sh configs/vheat/upernet_vheat_160k_ade20k_512x512_tiny.py 8
bash tools/dist_train.sh configs/vheat/upernet_vheat_160k_ade20k_512x512_small.py 8
bash tools/dist_train.sh configs/vheat/upernet_vheat_160k_ade20k_512x512_base.py 8


