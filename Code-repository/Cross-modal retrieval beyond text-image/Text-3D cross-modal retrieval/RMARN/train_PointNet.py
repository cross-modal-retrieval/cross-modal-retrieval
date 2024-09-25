import os
import torch
import argparse
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from data.data import split_dataset, ElephantDataset
from data.dpt_loader import load_dpt
from models.PointNet import PointNetEncoder
from CLIP_encoder import encode_text, encode_text_single
import re
import torch.nn as nn
import CLIP_Image
import json


# Training example
json_path = '/your_path/to/Area_1_processed.json'
args = argparse.Namespace(
    data_root='/your_path/to/Pointcloud/FPS_S3DIS',
    log_root='./logs',
    batch_size=16,
    num_workers=16,
    num_epoch=20,
    train_lr=1e-3,
    weight_decay=0,
    input_dim=6,
    output_dim=512,
    feature_transform=False
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(args):
    train_dataset, test_dataset = split_dataset(args.data_root, 'Area_5')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    net = PointNetEncoder(args).to(device)
    net.train()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.train_lr,
        weight_decay=args.weight_decay
    )

    Area = None
    rooms = []
    start_epoch = 0
    with torch.no_grad():
        for batch_id, (points, room_names) in enumerate(train_loader):
            points = points.to(device)
            points = points.transpose(2, 1).contiguous()
            scene_feature = net(points).detach()
            if Area is None:
                Area = scene_feature
            else:
                Area = torch.cat((Area, scene_feature), dim=0)

    return Area

caption_root = '/your_path/to/caption'

def get_txt_3d():
    point_feature = train(args).detach()

    areas_dpt, gt_id = load_dpt(json_path)
    text_features = encode_text(areas_dpt)

    return text_features, point_feature

def extract_room_name(filename):
    match = re.search(r'_([a-zA-Z]+_\d+)_', filename)
    return match.group(1) if match else None

def get_imgdir():
    img_root = "/your_path/to/2D_3D_S_pano"
    pattern = re.compile(r'.*_jpg$')
    jpg_folders = sorted([os.path.join(img_root, folder) for folder in os.listdir(img_root)
                          if os.path.isdir(os.path.join(img_root, folder)) and pattern.match(folder)])
    img_pairs = []
    for area in jpg_folders:
        img_dirs = sorted([img_dir for img_dir in os.listdir(area) if img_dir.endswith('.jpg')])
        for img_dir in img_dirs:
            img_path = os.path.join(area, img_dir)
            room_name = area[-10:-3] + extract_room_name(img_dir)
            img_pairs.append([img_path, room_name])
    img_pairs.sort(key=lambda x: x[1])
    return img_pairs

def get_cld_dir():
    train_dataset, test_dataset = split_dataset(args.data_root, 'Area_5')
    return train_dataset

def save_I3D_links():
    net = PointNetEncoder(args).to(device)
    net.eval()
    links = []
    save_cld_path = "/your_path/to/Cld_4Img.pt"
    save_img_path = "/your_path/to/Img_features/Img_4Cld.pt"
    cld_features = None
    img_features = None
    train_dataset = get_cld_dir()
    img_pairs = get_imgdir()
    j = 0
    for i in range(len(train_dataset)):
        points, room_name = train_dataset[i]
        points = points.unsqueeze(0).to(device)
        points = points.transpose(2, 1).contiguous()
        scene_feature = net(points).detach()
        if cld_features is None:
            cld_features = scene_feature
        else:
            cld_features = torch.cat((cld_features, scene_feature), dim=0)
        for img_pair in img_pairs:
            if room_name == img_pair[1]:
                j += 1
                img_feature = CLIP_Image.get_patch_tokens(img_pair[0])
                if img_features is None:
                    img_features = img_feature
                else:
                    img_features = torch.cat((img_features, img_feature), dim=0)
                links.append([[save_img_path, j-1], [save_cld_path, i]])
    torch.save(cld_features, save_cld_path)
    torch.save(img_features, save_img_path)
    link_path = "/your_path/to/links_cld100_img49.json"
    with open(link_path, "w", encoding='utf-8') as f:
        json.dump(links, f)
        print(f'successfully saved link.json')

def pre_elephant_t3d_link():
    cap_path = "/your_path/to/cap.json"
    with open(cap_path, 'r', encoding='utf-8') as f:
        cap = json.load(f)
    cap = cap['result']
    sorted_cap = sorted(cap.items())
    pld_root = "/your_path/to/pcd"
    sorted_pld = sorted([[obj, pld_root+"/"+obj+"/model.ply"] for obj in os.listdir(pld_root)])
    pld_dataset = ElephantDataset(sorted_pld)
    save_ele_t3d_links(sorted_cap, pld_dataset)

def save_ele_t3d_links(sorted_cap, pld_dataset):
    net = PointNetEncoder(args).to(device)
    net.eval()
    links = []
    save_cld_path = "/your_path/to/Ele_Cld_4Txt.pt"
    save_txt_path = "/your_path/to/Ele_Txt_4Cld.pt"
    cld_features = None
    txt_features = None
    flag = 1
    for i in range(2387):
        if sorted_cap[i][0] != pld_dataset[i][1]:
            print(f"mismatch at {i}")
    j = 0
    for i, (obj_cap, txt) in enumerate(sorted_cap):
        txt_feature = encode_text_single(txt)
        for points, obj_cld in pld_dataset:
            if obj_cap == obj_cld:
                j += 1
                if txt_features is None:
                    txt_features = txt_feature
                else:
                    txt_features = torch.cat((txt_features, txt_feature), dim=0)
                points = points.unsqueeze(0).to(device)
                points = points.transpose(2, 1).contiguous()
                scene_feature = net(points).detach()
                if cld_features is None:
                    cld_features = scene_feature
                else:
                    cld_features = torch.cat((cld_features, scene_feature), dim=0)
                links.append([[save_txt_path, i], [save_cld_path, j-1]])

    torch.save(cld_features, save_cld_path)
    torch.save(txt_features, save_txt_path)
    link_path = "/your_path/to/ele_links_txt77_cld100.json"
    with open(link_path, "w", encoding='utf-8') as f:
        json.dump(links, f)
        print(f'successfully saved link.json')

def redo_ele_link():
    link_path = "/your_path/to/redo_ele_link_811.json"
    save_cld_path = "/your_path/to/Ele_Cld_4Txt.pt"
    save_txt_path = "/your_path/to/Ele_Txt_4Cld.pt"
    cld = torch.load(save_cld_path)
    txt = torch.load(save_txt_path)
    links = []
    for i in range(cld.size(0)):
        links.append([[save_txt_path, i], [save_cld_path, i]])
    with open(link_path, "w", encoding='utf-8') as f:
        json.dump(links, f)
    print("success")

if __name__ == '__main__':
    redo_ele_link()
    # save_elephant_t3d_link()
    # pre_elephant_t3d_link()
    # cld_names = get_cld_dir()
    # get_imgdir()
    # train(args)
    # save_I3D_links()
