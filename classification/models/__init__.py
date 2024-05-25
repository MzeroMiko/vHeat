import os
import torch
import torch.nn.functional as F

from functools import partial

from .vHeat import vHeat


def build_vHeat_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    
    if model_type in ["vHeat"]:
        model = vHeat(
            in_chans=config.MODEL.VHEAT.IN_CHANS, 
            patch_size=config.MODEL.VHEAT.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VHEAT.DEPTHS, 
            dims=config.MODEL.VHEAT.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.VHEAT.MLP_RATIO,
            post_norm=config.MODEL.VHEAT.POST_NORM,
            layer_scale=config.MODEL.VHEAT.LAYER_SCALE,
            img_size=config.DATA.IMG_SIZE,
            infer_mode=config.EVAL_MODE or config.THROUGHPUT_MODE,
        )
        if config.THROUGHPUT_MODE:
            model.infer_init()
        return model
    
    
def build_model(config, is_pretrain=False):
    model = build_vHeat_model(config, is_pretrain)
    return model
