import os
import numpy as np
import torch
import argparse
from data.data_FPS import split_dataset


def save_sampled_points(dataset, output_root):
    for i in range(len(dataset)):
        try:
            sampled_points, room_name = dataset[i]
            area, area_num, room = room_name.split('_', 2)
            area = area + '_' + area_num
            room_dir = os.path.join(output_root, area, room)
            os.makedirs(room_dir, exist_ok=True)
            output_file = os.path.join(room_dir, f"{room}.txt")
            np.savetxt(output_file, sampled_points.cpu().numpy(), fmt='%.6f')
            print(output_file, "has been saved!")
        except Exception as e:
            print(f"Error processing {room_name}: {e}")


def process(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 'Area_6'为测试集
    train_dataset, test_dataset = split_dataset(args.data_root, 'Area_6', device)
    save_sampled_points(train_dataset, args.output_root)
    save_sampled_points(test_dataset, args.output_root)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='autodl-tmp/Stanford3dDataset_v1.2_Aligned_Version')
    parser.add_argument('--output_root', type=str, default='autodl-tmp/FPS_S3DIS')
    args = parser.parse_args()

    process(args)
