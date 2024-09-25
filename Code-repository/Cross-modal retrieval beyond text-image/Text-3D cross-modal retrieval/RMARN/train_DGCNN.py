import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from data.data import split_dataset
from models.Encoder import SceneFeatureExtractor

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = split_dataset(args.data_root, 'Area_6')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    net = SceneFeatureExtractor(args).to(device)
    net.train()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.train_lr,
        weight_decay=args.weight_decay
    )

    start_epoch = 0
    for epoch in range(start_epoch, args.num_epoch):
        for batch_id, (points, room_names) in enumerate(train_loader):
            points = points.to(device)
            print("points:", points.shape)
            points = points.transpose(2, 1).contiguous()

            scene_feature = net(points)
            print("scene_feature:", scene_feature.shape)
            # loss =
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/your_path/to/FPS_S3DIS')
    # parser.add_argument('--output_root', type=str, default='/your_path/to/FPS_S3DIS')
    parser.add_argument('--log_root', type=str, default='/your_path/to/logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--k', type=int, default=64)
    # parser.add_argument('--input_dim', type=int, default=6)

    args = parser.parse_args()

    train(args)
