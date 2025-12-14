# finetune_for_myreal.py
import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import *  # 里面有 SFHformer_l
from datasets.Rain_Dataloader import TrainData_for_Rain100L  # 你可以复制一个改名为 TrainData_for_MyReal
from utils.utils import calculate_psnr_torch, calculate_ssim_torch  # 根据你项目里实际名字来

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SFHformer_l', type=str)
parser.add_argument('--pretrained', type=str, required=True, help='Rain100L 上训好的 best.pth')
parser.add_argument('--train_dir', type=str, required=True, help='data/MyRealRain/train 根目录')
parser.add_argument('--val_dir', type=str, required=True, help='data/MyRealRain/test 根目录')
parser.add_argument('--save_dir', type=str, default='./checkpoints_finetune')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10)  # 短期微调
parser.add_argument('--lr', type=float, default=1e-4)  # 比原来小
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

def build_model_and_load_pretrained():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = eval(args.model.replace('-', '_'))().to(device)

    ckpt = torch.load(args.pretrained, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    # 去掉可能存在的 'module.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    net.load_state_dict(new_state_dict, strict=True)

    # ===== 关键：冻结一部分层，只训练高层 =====
    for name, param in net.named_parameters():
        if name.startswith('patch_embed') or name.startswith('layer1') or name.startswith('layer2'):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 只把 requires_grad=True 的参数扔给优化器
    params_to_update = [p for p in net.parameters() if p.requires_grad]
    print(f'Trainable params: {sum(p.numel() for p in params_to_update)/1e6:.2f} M')

    optimizer = optim.AdamW(params_to_update, lr=args.lr, weight_decay=1e-4)
    return net, optimizer, device

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    net, optimizer, device = build_model_and_load_pretrained()

    # ====== 这里你可以写一个新的 TrainData_for_MyReal 类，更通用些 ======
    train_dataset = TrainData_for_Rain100L(train_data_dir=args.train_dir,
                                           patch_size=args.patch_size)
    val_dataset   = TrainData_for_Rain100L(train_data_dir=args.val_dir,
                                           patch_size=args.patch_size)  # 或者写一个 TestData_for_MyReal

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True)

    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        net.train()
        for i, batch in enumerate(train_loader):
            rain = batch['source'].to(device)
            gt   = batch['target'].to(device)

            out = net(rain)

            loss_l1 = F.l1_loss(out, gt)
            loss = loss_l1  # 如果原来有频域 loss，这里可以把那部分也复制过来

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f'Epoch[{epoch}/{args.epochs}] Step[{i+1}/{len(train_loader)}] '
                      f'loss={loss.item():.4f}')

        # ====== 每个 epoch 做一次验证，统计在真实验证集上的 PSNR/SSIM ======
        net.eval()
        psnr_sum, ssim_sum, cnt = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                rain = batch['source'].to(device)
                gt   = batch['target'].to(device)
                out  = net(rain).clamp_(0, 1)

                psnr = calculate_psnr_torch(gt, out)
                ssim = calculate_ssim_torch(gt, out)
                psnr_sum += psnr.item()
                ssim_sum += ssim.item()
                cnt += 1
        psnr_avg = psnr_sum / cnt
        ssim_avg = ssim_sum / cnt
        print(f'===> Epoch[{epoch}] Val PSNR={psnr_avg:.2f} SSIM={ssim_avg:.4f}')

        # 保存 best
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            save_path = os.path.join(args.save_dir, f'{args.model}_myreal_finetune_best.pth')
            torch.save({'state_dict': net.state_dict(),
                        'psnr': best_psnr,
                        'epoch': epoch}, save_path)
            print(f'*** New best model saved to {save_path}')

if __name__ == '__main__':
    main()
