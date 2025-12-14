import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.Rain_Dataloader import TestData_for_Rain100L
from models import *
from utils import AverageMeter
from utils.utils import calculate_psnr_torch


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SFHformer_l', type=str,
                    help='模型名称，例如 SFHformer_l / SFHformer_m / SFHformer_s')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='训练好的 .pth 文件路径')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Rain100L 测试集根目录，对应 val_data_dir')
parser.add_argument('--save_dir', type=str, default='./results_rain100L',
                    help='去雨结果保存目录')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()


def build_and_load_model():
    """
    构建模型并加载权重：
    1）不使用 DataParallel；
    2）自动去掉 state_dict 中可能存在的 'module.' 前缀。
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ① 构建“裸”的模型（不包 DataParallel）
    net = eval(args.model.replace('-', '_'))().to(device)

    # ② 读取 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        # 万一你保存的时候就是直接 save 的 state_dict
        state_dict = ckpt

    # ③ 统一处理 key：有 'module.' 去掉，没有就保持
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    # ④ 加载
    # 这里先 strict=False，这样如果有多余的 key 不会直接报错，方便你先跑通；
    # 如果想严格检查，可以改成 strict=True 看看有没有真正不匹配的参数。
    incompatible = net.load_state_dict(new_state_dict, strict=False)

    try:
        if len(incompatible.missing_keys) > 0:
            print('[Warning] Missing keys:', incompatible.missing_keys)
        if len(incompatible.unexpected_keys) > 0:
            print('[Warning] Unexpected keys:', incompatible.unexpected_keys)
    except AttributeError:
        pass

    net.eval()
    return net, device


def main():
    os.makedirs(args.save_dir, exist_ok=True)

    net, device = build_and_load_model()

    # 2. 构建测试集 DataLoader
    test_dataset = TestData_for_Rain100L(local_size=8,
                                         val_data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    # 3. 循环测试：推理 + 指标 + 保存图片
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            rain = batch['source'].to(device)
            gt = batch['target'].to(device)
            # Rain_Dataloader 里我给你加过 filename 字段的话就用它；否则用 idx 命名
            if 'filename' in batch:
                name = batch['filename'][0]
            else:
                name = f'{idx:04d}.png'

            out = net(rain).clamp_(0, 1)

            psnr, ssim = calculate_psnr_torch(gt, out)
            psnr_meter.update(psnr.item(), 1)
            ssim_meter.update(ssim.item(), 1)

            # 保存结果图片
            save_path = os.path.join(args.save_dir,
                                     name.replace('.jpg', '.png').replace('.JPG', '.png'))
            save_image(out, save_path)

            if (idx + 1) % 10 == 0:
                print(f'[{idx + 1}/{len(test_loader)}] '
                      f'PSNR_now={psnr_meter.avg:.2f}, SSIM_now={ssim_meter.avg:.4f}')

    print('================ Rain100L Test Done ================')
    print(f'Final PSNR: {psnr_meter.avg:.2f} dB')
    print(f'Final SSIM: {ssim_meter.avg:.4f}')


if __name__ == '__main__':
    main()
