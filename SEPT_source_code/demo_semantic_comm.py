"""demo_semantic_comm.py

ModelNet40 语义通信演示脚本。
如果存在训练权重 semantic_ae_airplane.pth，将直接加载；否则会在 airplane 类上训练 50 个 epoch。
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


try:
    from openpoints.dataset.modelnet.modelnet40_ply_2048_loader import ModelNet40Ply2048
except ModuleNotFoundError as exc:
    raise ImportError(
        '无法导入 openpoints。请确保已激活 `openpoints` 环境，或者将项目根目录添加到 PYTHONPATH。'
    ) from exc

try:
    from octree.octree import calculate_metrics_d1_d2
except ModuleNotFoundError as exc:
    raise ImportError(
        '无法导入 octree。请确认 `octree` 模块已安装或位于项目路径中。'
    ) from exc

from models.PCT_trans_ptv4_cls import get_model, calc_cd


def parse_args():
    parser = argparse.ArgumentParser('ModelNet40 Semantic Communication Demo')
    parser.add_argument('--data_dir', type=str, default=None, help='ModelNet40 dataset directory')
    parser.add_argument('--model_path', type=str, default='semantic_ae_airplane.pth', help='模型权重文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='Airplane-only training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='训练 batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='训练学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='优化器权重衰减')
    parser.add_argument('--snr', type=float, default=10.0, help='AWGN 信道 SNR')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备: cuda 或 cpu')
    return parser.parse_args()


def ensure_dataset(data_dir: str):
    if not os.path.exists(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')):
        raise RuntimeError(
            f'未找到 ModelNet40 数据集目录: {os.path.join(data_dir, "modelnet40_ply_hdf5_2048")}.\n'
            '请先从 https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip 下载并解压，'
            '或将本地备份链接到该目录。'
        )


def build_airplane_loader(dataset, batch_size: int):
    airplane_label = dataset.classes.index('airplane')
    airplane_indices = np.where(dataset.label == airplane_label)[0].tolist()
    if len(airplane_indices) == 0:
        raise RuntimeError('在训练集中没有找到 airplane 类样本。')
    subset = Subset(dataset, airplane_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def train_airplane_model(model, data_loader, device, epochs: int, lr: float, weight_decay: float, snr: float, save_path: str):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_samples = 0
        start_time = time.time()
        for batch in data_loader:
            if 'x' in batch:
                inp = {'pos': batch['pos'].float().to(device), 'x': batch['x'].float().to(device)}
            else:
                inp = batch['pos'].float().to(device)
            recon, cd_loss, _ = model(inp, snr=snr)
            loss = torch.mean(cd_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (batch['pos'].shape[0] if hasattr(batch['pos'], 'shape') else points.size(0))
            num_samples += (batch['pos'].shape[0] if hasattr(batch['pos'], 'shape') else points.size(0))
        epoch_loss = total_loss / max(num_samples, 1)
        print(f'[Train] epoch={epoch:02d}/{epochs} loss={epoch_loss:.6f} time={time.time()-start_time:.1f}s')
    torch.save(model.state_dict(), save_path)
    print(f'已保存训练权重到 {save_path}')


def visualize_airplane(original: np.ndarray, reconstructed: np.ndarray, out_path: str):
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    cmap = plt.get_cmap('Spectral')
    colors_in = cmap((original[:, 2] - original[:, 2].min()) / (original[:, 2].ptp() + 1e-9))
    colors_out = cmap((reconstructed[:, 2] - reconstructed[:, 2].min()) / (reconstructed[:, 2].ptp() + 1e-9))

    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c=colors_in, s=6, marker='.', alpha=0.9)
    ax1.set_title('Input Airplane')
    ax1.set_axis_off()
    ax1.set_box_aspect([1, 1, 1])

    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], c=colors_out, s=6, marker='.', alpha=0.9)
    ax2.set_title('Reconstructed Airplane')
    ax2.set_axis_off()
    ax2.set_box_aspect([1, 1, 1])

    all_points = np.vstack([original, reconstructed])
    limits = np.array([all_points.min(axis=0), all_points.max(axis=0)])
    for ax in (ax1, ax2):
        ax.set_xlim(limits[0, 0], limits[1, 0])
        ax.set_ylim(limits[0, 1], limits[1, 1])
        ax.set_zlim(limits[0, 2], limits[1, 2])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存重建对比图: {out_path}')


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    data_dir = os.path.abspath(args.data_dir) if args.data_dir is not None else os.path.join(ROOT_DIR, 'data', 'ModelNet40Ply2048')
    print(f'Using ModelNet40 data_dir: {data_dir}')
    ensure_dataset(data_dir)

    weights_path = os.path.abspath(args.model_path)
    model = get_model(bottleneck_size=300, recon_points=2048, jscc_hidden_dim=512).to(device)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f'加载预训练权重: {weights_path}')
    else:
        print('未检测到预训练权重，开始在 airplane 类上训练 50 epoch...')
        train_dataset = ModelNet40Ply2048(num_points=2048, data_dir=data_dir, split='train')
        airplane_loader = build_airplane_loader(train_dataset, batch_size=args.batch_size)
        train_airplane_model(model, airplane_loader, device, args.epochs, args.lr, args.weight_decay, args.snr, weights_path)

    print('开始在 ModelNet40 测试集上选取 airplane 样本进行重建演示...')
    test_dataset = ModelNet40Ply2048(num_points=2048, data_dir=data_dir, split='test')
    airplane_label = test_dataset.classes.index('airplane')
    airplane_indices = np.where(test_dataset.label == airplane_label)[0].tolist()
    if len(airplane_indices) == 0:
        raise RuntimeError('测试集中没有找到 airplane 类样本。')

    sample = test_dataset[airplane_indices[0]]
    input_points = sample['pos'].astype(np.float32)
    input_x = sample.get('x', None)
    with torch.no_grad():
        model.eval()
        pos_t = torch.from_numpy(input_points).float().unsqueeze(0).to(device)
        if input_x is not None:
            x_t = torch.from_numpy(input_x.astype(np.float32)).float().unsqueeze(0).to(device)
            inp = {'pos': pos_t, 'x': x_t}
        else:
            inp = pos_t
        recon_points, cd_loss, coarse_cd = model(inp, snr=args.snr)

    recon_np = recon_points[0].cpu().numpy()
    _, d1_psnr, d2_psnr = calculate_metrics_d1_d2(input_points, recon_np)
    print(f'Chamfer Distance: {cd_loss.item():.6f}')
    print(f'D1PSNR: {d1_psnr:.2f} dB')
    print(f'D2PSNR: {d2_psnr:.2f} dB')

    save_png = os.path.join(BASE_DIR, 'airplane_recon.png')
    visualize_airplane(input_points, recon_np, save_png)


if __name__ == '__main__':
    main()
