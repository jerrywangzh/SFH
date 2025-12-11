# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from datasets.Rain_Dataloader import TestData_for_Rain100L
from models import *
from utils.utils import calculate_psnr_torch


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SFHformer_m', type=str, help='model name')
parser.add_argument('--weights', required=True, type=str, help='path to checkpoint (.pth)')
parser.add_argument('--data_dir', required=True, type=str, help='path to Rain100L test root (rain_data_test_Light)')
parser.add_argument('--save_dir', default='./results/rain100L', type=str, help='path to save restored images')
parser.add_argument('--num_workers', default=4, type=int, help='dataloader workers')
parser.add_argument('--local_size', default=8, type=int, help='patch align size')
args = parser.parse_args()


def main():
    network = eval(args.model.replace('-', '_'))()
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            network = nn.DataParallel(network, device_ids=list(range(device_count))).cuda()
        else:
            network = network.cuda()
        checkpoint = torch.load(args.weights, map_location='cuda')
    else:
        checkpoint = torch.load(args.weights, map_location='cpu')

    network.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    network.eval()

    test_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    dataset = TestData_for_Rain100L(args.local_size, test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    os.makedirs(args.save_dir, exist_ok=True)
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing Rain100L'):
            source = batch['source'].cuda() if torch.cuda.is_available() else batch['source']
            target = batch['target'].cuda() if torch.cuda.is_available() else batch['target']
            filename = batch['filename'][0]

            output = network(source).clamp_(0, 1)
            psnr, ssim = calculate_psnr_torch(target, output)
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            count += 1

            save_path = os.path.join(args.save_dir, filename.replace('.jpg', '').replace('.png', '') + '_restored.png')
            save_image(output.cpu(), save_path)

    avg_psnr = total_psnr / max(count, 1)
    avg_ssim = total_ssim / max(count, 1)
    print(f'Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}, Samples: {count}')


if __name__ == '__main__':
    main()
