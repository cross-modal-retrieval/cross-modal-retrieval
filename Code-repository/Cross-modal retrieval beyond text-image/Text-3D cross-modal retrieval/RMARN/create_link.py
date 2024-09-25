from torch.utils.data import Dataset
import json
import torch
import random
import os
import glob
import data.dpt_loader
from data.data import Comp_cld_Dataset
from models.PointNet2 import PointNet2
from CLIP_encoder import encode_text_single_with_bert

device = torch.device("cuda:0")

def create_file_list(root_dir):
    files = []
    areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    for area in areas:
        if area.startswith('.'):
            continue
        area_dir = os.path.join(root_dir, area)
        for room in os.listdir(area_dir):
            if room.startswith('.'):
                continue
            room_dir = os.path.join(area_dir, room)
            if not os.path.isdir(room_dir):
                continue
            for file in os.listdir(room_dir):
                if file.startswith('.'):
                    continue
                if file.endswith('.txt'):
                    files.append([f"{area}_{room}", os.path.join(room_dir, file)])
    files = sorted(files)
    return files

def find_processed_files(root_dir):
    processed_files = []
    for file_path in glob.iglob(os.path.join(root_dir, '**', '*processed*'), recursive=True):
        if os.path.isfile(file_path):
            processed_files.append(os.path.abspath(file_path))
    processed_files = sorted(processed_files)
    return processed_files

def load_dpt(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    gt_list = [p for p in data['result']]
    gt_list.sort()
    areas_dpt = []
    gt_id = []
    for gt in gt_list:
        tmp = []
        gt_id.append(gt_list.index(gt))
        for dpt in data['result'][gt]:
            tmp.append(dpt)
        areas_dpt.append(tmp)
    return areas_dpt, gt_list

def load_area_dpt_pair(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    area_pair = []
    gt_list = [p for p in data['result']]
    gt_list.sort()
    for room in gt_list:
        for dpt in data['result'][room]:
            area_pair.append([room, dpt])
    return area_pair

def save_link_origin():
    pc_dir = '/your_path/to/Pointcloud/FPS_S3DIS'
    json_dir = '/your_path/to/discription/caption'
    cloud_pt_path = '/your_path/to/Pointcloud/extracted_pc.pt'
    links = []
    clouds_pair = create_file_list(pc_dir)
    json_files = find_processed_files(json_dir)
    for json_file in json_files:
        pt_file = json_file.replace('_processed.json', '_text_features.pt')
        _, room_names = load_dpt(json_file)
        for room in room_names:
            if room not in clouds:
                print(f'Room {room} not found in clouds, skipping...')
                continue
            txt_index = room_names.index(room)
            pc_index = clouds.index(room)
            link = [[pt_file, txt_index], [cloud_pt_path, pc_index]]
            links.append(link)
    with open('/your_path/to/link.json', 'w', encoding='utf-8') as f:
        json.dump(links, f)
        print(f'successfully saved link.json')

def creat_txt_pair(json_files):
    txt_pairs = []
    for json_file in json_files:
        _, room_names = load_dpt(json_file)
        for room in room_names:
            if room not in clouds:
                print(f'Room {room} not found in clouds, skipping...')
                continue
            txt_index = room_names.index(room)
            pc_index = clouds.index(room)
            link = [[pt_file, txt_index], [cloud_pt_path, pc_index]]
            links.append(link)

def save_link_bert_PNplus():
    pc_dir = '/your_path/to/Pointcloud/FPS_S3DIS'
    json_dir = '/your_path/to/discription/caption'
    links = []
    cld_pairs = create_file_list(pc_dir)
    json_files = find_processed_files(json_dir)
    text_feature_path = "/your_path/to/discription/txt_bert_PNplus.pt"
    cld_feature_path = "/your_path/to/Pointcloud/cld_bert_PNplus.pt"
    txt_pairs = []
    for json_file in json_files:
        area_pair = load_area_dpt_pair(json_file)
        txt_pairs.extend(area_pair)
    print(f"len txt pair: {len(txt_pairs)}")

    txt_features = None
    cld_features = None
    cld_pair_set = Comp_cld_Dataset(cld_pairs)
    cld_model = PointNet2(input_dim=6).to(device)
    cld_model.eval()
    j = 0
    for i, (points, cld_name) in enumerate(cld_pair_set):
        points = points.unsqueeze(0).to(device)
        points = points.transpose(2, 1).contiguous()
        scene_feature = cld_model(points).detach()
        if cld_features is None:
            cld_features = scene_feature
        else:
            cld_features = torch.cat((cld_features, scene_feature), dim=0)
        for txt_name, dpt in txt_pairs:
            if cld_name == txt_name:
                j += 1
                dpt_feature = encode_text_single_with_bert(dpt)
                if txt_features is None:
                    txt_features = dpt_feature
                else:
                    txt_features = torch.cat((txt_features, dpt_feature), dim=0)
                links.append([[text_feature_path, j-1], [cld_feature_path, i]])
                print(f"links[{j-1}]:{links[j - 1]}")

    torch.save(txt_features, text_feature_path)
    torch.save(cld_features, cld_feature_path)
    link_path = "/your_path/to/bert_pn++_links_txt5_cld5.json"
    print(f"cls shape:{cld_features.size()} {cld_features.dtype}\n"
          f"txt shape:{txt_features.size()} {txt_features.dtype}\n"
          f"links len:{len(links)}")
    with open(link_path, "w", encoding='utf-8') as f:
        json.dump(links, f)
        print(f'successfully saved link.json')

if __name__ == '__main__':
    save_link_bert_PNplus()
