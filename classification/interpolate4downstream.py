import os
import torch
import torch.nn.functional as F
import argparse


def parse_option():
    parser = argparse.ArgumentParser('Interpolate pretrained checkpoint for downstream tasks.', add_help=False)
    parser.add_argument('--pt_pth', type=str, required=True, help='path to pretrained pth', )
    parser.add_argument('--pt_size', type=int, default=224, )
    parser.add_argument('--tg_pth', type=str, required=True, help='path to save target pth', )
    parser.add_argument('--tg_size', type=int, default=512, )
    args, unparsed = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_option()
    
    pt_pth = torch.load(args.pt_pth, map_location='cpu')
    
    for i in range(4):
        print("Layer {}:".format(i))
        interpolate_size = (args.tg_size - args.pt_size) // (2**(i + 2))
        tmp = pt_pth['model']['freq_embed.{}'.format(i)]
        print("Shape before interpolation:", tmp.shape)
        tmp = F.pad(tmp.permute(2, 0, 1), (0, interpolate_size, 0, interpolate_size), 'constant', 0)
        pt_pth['model']['freq_embed.{}'.format(i)] = tmp.permute(1, 2, 0)
        print("Shape after interpolation:", tmp.permute(1, 2, 0).shape)
    
    torch.save(pt_pth, args.tg_pth)
    print("Finished! Saved to {}.".format(args.tg_pth))