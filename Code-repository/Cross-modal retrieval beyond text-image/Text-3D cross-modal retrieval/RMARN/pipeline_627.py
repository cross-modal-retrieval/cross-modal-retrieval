import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Dataset
import json
import os

import cnn
import sim
from recall import get_rank, get_sim, evaluation
from transformer import DecoderBlock, Config

import tqdm

import argparse
import gc

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
# device = torch.device("cuda:1")


class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()

    def forward(self, u_sim_t):
        ground_truth = torch.arange(u_sim_t.size(0), dtype=torch.long, device=u_sim_t.device)
        u_sim_cloud = u_sim_t.transpose(0, 1)
        loss_text = F.cross_entropy(u_sim_t, ground_truth)
        loss_point_clouds = F.cross_entropy(u_sim_cloud, ground_truth)

        return (loss_text + loss_point_clouds) / 2.0


class PairDataset(Dataset):
    def __init__(self, txt_root, cloud_root, link_pth, device, cld_first=False, cache_num=None):
        """
            *** examples: ***
        :param txt_root:   "/usr/txt/"
        :param cloud_root:  "/usr/cld/"
        :param link_pth:    "/usr/link.json"
        :param cache_num:    3 / None
        """
        self.txt_root = txt_root
        self.cloud_root = cloud_root
        self.device = device
        with open(link_pth, "r", encoding='utf-8') as f:
            self.link = json.load(f)

        if cache_num is not None:
            self.cache = [["", None] for _ in range(cache_num)]
        else:
            self.cache = None
        self.cld_first = cld_first

    def __len__(self):
        return len(self.link)

    def get_block(self, pth) -> torch.tensor:
         if self.cache is not None:
            for ca in self.cache:
                pt, blk = ca
                if pt == pth:
                    return blk

            idx = random.randint(0, len(self.cache) - 1)
            self.cache[idx] = [
                pth,
                torch.load(pth)
            ]
            return self.cache[idx][1]
         else:
            blk = torch.load(pth)
            return blk

    def get_line(self, pth, idx) -> torch.tensor:
        blk = self.get_block(pth)
        return blk[idx]

    def __getitem__(self, item):
        txt, cld = self.link[item]
        pth, idx = txt
        pth = self.txt_root + pth
        txt = self.get_line(pth, idx)

        pth, idx = cld
        pth = self.cloud_root + pth
        cld = self.get_line(pth, idx)

        assert len(txt.shape) == len(cld.shape), \
            f"Shape mismatch: txt shape {txt.shape} vs cld shape {cld.shape}"
        # return txt.to(self.device), cld.to(self.device)
        if self.cld_first:
            return cld.to(self.device), txt.to(self.device)
        else:
            return txt.to(self.device), cld.to(self.device)


    def get_all_T3D(self):
        txt = torch.load("/your_path/all_in1.pt")
        cld = torch.load("/your_path/extracted_pc.pt")
        link = [[txt[1], cld[1]] for txt, cld in self.link]
        return txt, cld, link

    def get_all_ele(self, txt_pth, cld_pth):
        txt = torch.load(txt_pth)
        cld = torch.load(cld_pth)
        link = [[txt[1], cld[1]] for txt, cld in self.link]
        return txt, cld, link

    def get_all_I3D(self):
        img = torch.load("/your_path/Img_4Cld.pt")
        cld = torch.load("/your_path/Cld_4Img.pt")
        link = [[img[1], cld[1]] for img, cld in self.link]
        return img, cld, link


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        query = self.query_projection(x).transpose(0, 1)
        key = self.key_projection(x).transpose(0, 1)
        value = self.value_projection(x).transpose(0, 1)

        attn_output, attn_output_weights = self.multihead_attn(query, key, value, attn_mask=attention_mask)
        attn_output = attn_output.transpose(0, 1)

        attn_output = self.dropout(attn_output)
        attn_output = self.layernorm(attn_output + x)

        return attn_output


class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src):
        src = self.input_proj(src)
        src = src.transpose(0, 1)  # Transformer expects (seq_len, batch_size, embed_dim)
        src = self.encoder_layer(src)
        src = self.norm(src)
        src = src.transpose(0, 1)  # Back to (batch_size, seq_len, embed_dim)
        return src



class MiniModel(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1,
                 rank: int = 32, num_layers: int = 6):
        super(MiniModel, self).__init__()
        self.global_sim = sim.GlobalCos(d_model, rank, pool='mean')
        # self.local_sim = sim.LocalSimilarity(in_size=d_model, k=64,
        #                                      max_txt_len=77, max_cld_len=300, n_head=n_head, dropout=dropout)
        # self.Conv = cnn.ConvGlobalSimilarity(n_head=n_head, base_block=cnn.ResBlock)
        self.text_self_attention_layers = nn.ModuleList(
            [SelfAttentionEncoder(d_model, d_model, n_head,
                                  ff_dim=d_model, dropout=dropout) for _ in range(num_layers)])
        self.point_cloud_self_attention_layers = nn.ModuleList(
            [SelfAttentionEncoder(d_model, d_model, n_head,
                                  ff_dim=d_model, dropout=dropout) for _ in range(num_layers)])

        self.down_t_sample = nn.MaxPool1d(kernel_size=20, stride=20, padding=0)
        self.down_pc_sample = nn.MaxPool1d(kernel_size=20, stride=20, padding=0)

    def forward(self, txt: torch.Tensor, point_cloud: torch.Tensor):
        with torch.no_grad():
            txt = txt.transpose(2, 1)
            txt = self.down_t_sample(txt)  # (batch_size, 512, N/k)
            txt = txt.transpose(2, 1)

            point_cloud = point_cloud.transpose(2, 1)
            point_cloud = self.down_pc_sample(point_cloud)  # (batch_size, 512, N/k)
            point_cloud = point_cloud.transpose(2, 1)


        for layer in self.text_self_attention_layers:
            txt = layer(txt)

        for layer in self.point_cloud_self_attention_layers:
            point_cloud = layer(point_cloud)

        g_sim = self.global_sim(txt, point_cloud, cross_sim=True)
        return g_sim


class RetrievalModel(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1,
                 rank: int = 32, num_layers: int = 6, is_i3d: bool = False):
        super(RetrievalModel, self).__init__()
        self.down_pc_sample = nn.MaxPool1d(kernel_size=10, stride=10, padding=0)
        self.down_t_sample = nn.MaxPool1d(kernel_size=5, stride=5, padding=0)
        
        self.global_sim = sim.GlobalSimilarity(d_model, rank, pool='mean')
        self.local_sim = sim.LocalSimilarity(in_size=d_model, k=64,
                                             max_txt_len=77, max_cld_len=300, n_head=n_head, dropout=dropout)
        self.Conv = cnn.ConvGlobalSimilarity(n_head=n_head, base_block=cnn.ResBlock)
        self.is_i3d = is_i3d
        if self.is_i3d:
            self.transform = nn.Linear(768, d_model)

        self.text_self_attention_layers = nn.ModuleList(
            [SelfAttentionEncoder(d_model, d_model, n_head,
                                  ff_dim=d_model, dropout=dropout) for _ in range(num_layers)])
        self.point_cloud_self_attention_layers = nn.ModuleList(
            [SelfAttentionEncoder(d_model, d_model, n_head,
                                  ff_dim=d_model, dropout=dropout) for _ in range(num_layers)])
   
    def forward(self, txt: torch.Tensor, point_cloud: torch.Tensor):
        # with torch.no_grad():
        #     txt = txt.transpose(2, 1)
        #     txt = self.down_t_sample(txt)  # (batch_size, 512, N/k)
        #     txt = txt.transpose(2, 1)
        #
        #     point_cloud = point_cloud.transpose(2, 1)
        #     point_cloud = self.down_pc_sample(point_cloud)  # (batch_size, 512, N/k)
        #     point_cloud = point_cloud.transpose(2, 1)
            
        if self.is_i3d:
                txt = self.transform(txt)

        for layer in self.text_self_attention_layers:
            txt = layer(txt)

        for layer in self.point_cloud_self_attention_layers:
            point_cloud = layer(point_cloud)

        l_sim = self.local_sim(txt, point_cloud, cross_sim=True)
        head_shape = l_sim.shape[:2]
        tail_shape = l_sim.shape[2:]
        l_sim = l_sim.reshape(-1, *tail_shape)
        conv_l = self.Conv(l_sim)
        conv_l = conv_l.reshape(*head_shape)
        g_sim = self.global_sim(txt, point_cloud, cross_sim=True)
        ultimate_sim = conv_l + g_sim
        return ultimate_sim

      
class RetrievalConfig(Config):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, device,
                 rank, is_i3d):
        super().__init__(vocab_size, block_size, n_layer, n_head, n_embd, dropout, device)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.device = device
        self.rank = rank
        self.is_i3d = is_i3d
    

class RetrievalModel_t(nn.Module):
    def __init__(self, config):
        super(RetrievalModel_t, self).__init__()
        self.global_sim = sim.GlobalSimilarity(config.n_embd, config.rank, pool='mean')
        self.local_sim = sim.LocalSimilarity(in_size=config.n_embd, k=64, max_txt_len=77, max_cld_len=300,
                                             n_head=config.n_head, dropout=config.dropout)
        self.Conv = cnn.ConvGlobalSimilarity(n_head=config.n_head, base_block=cnn.ResBlock)
        self.is_i3d = config.is_i3d
        if self.is_i3d:
            self.transform = nn.Linear(768, config.n_embd)

        self.text_self_attention_layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layer)])
        self.point_cloud_self_attention_layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layer)])
        self.sim_weight = nn.Parameter(torch.tensor(0.5, device=config.device))

    def forward(self, txt: torch.Tensor, point_cloud: torch.Tensor):
        if self.is_i3d:
            txt = self.transform(txt)

        for layer in self.text_self_attention_layers:
            txt = layer(txt)

        for layer in self.point_cloud_self_attention_layers:
            point_cloud = layer(point_cloud)

        l_sim = self.local_sim(txt, point_cloud, cross_sim=True)
        head_shape = l_sim.shape[:2]
        tail_shape = l_sim.shape[2:]
        l_sim = l_sim.reshape(-1, *tail_shape)
        conv_l = self.Conv(l_sim)
        conv_l = conv_l.reshape(*head_shape)
        g_sim = self.global_sim(txt, point_cloud, cross_sim=True)
        ultimate_sim = self.sim_weight * conv_l + (1 - self.sim_weight) * g_sim
        return ultimate_sim



def train(model, dataloader, criterion, optimizer, txt, cld, link,
          save_model_path, checkpoint_path, num_epochs=80):

    best_recall_at_5 = 0
    best_model_wts = None
    model = model.to(device)
    txt, cld = txt.to(device), cld.to(device)
    for epoch in tqdm.trange(num_epochs):
        model.train()
        epoch_losses = []
        for text, point_clouds in dataloader:
            text, point_clouds = text.to(device), point_clouds.to(device)
            optimizer.zero_grad()

            u_sim = model(text, point_clouds)
            # u_sim = u_sim.to(device)

            loss = criterion(u_sim)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)

        model.eval()
        recalls, mr, mrr = evaluation(txt, cld, link, sim_func=model, t2c=True, recall_idx=[1, 5, 10])
        current_recall_at_5 = recalls[1]
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}," 
              f"Recall@1: {recalls[0]:.4f},  Recall@5: {recalls[1]:.4f}, Recall@10: {recalls[2]:.4f}")

        if current_recall_at_5 > best_recall_at_5:
            best_recall_at_5 = current_recall_at_5
            best_model_wts = model.state_dict()


    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    if best_model_wts is not None:
        torch.save(best_model_wts, save_model_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_recall_at_5 = checkpoint['best_recall_at_5']
        best_model_wts = checkpoint['best_model_wts']
        print(f"Checkpoint loaded, starting from epoch {start_epoch}")
        return start_epoch, best_recall_at_5, best_model_wts
    else:
        print("No checkpoint found, starting from scratch")
        return 0, 0, None


def train_save_checkpoint(model, dataloader, criterion, optimizer, txt, cld, link,
                          save_model_path, checkpoint_path, num_epochs=100):
    start_epoch, best_recall_at_5, best_model_wts = load_checkpoint(model, optimizer, checkpoint_path)
    model = model.to(device)
    txt, cld = txt.to(device), cld.to(device)
    for epoch in tqdm.trange(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        for text, point_clouds in dataloader:
            # text, point_clouds = text.to(device), point_clouds.to(device)
            optimizer.zero_grad()

            u_sim = model(text, point_clouds)

            loss = criterion(u_sim)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())


            del text, point_clouds, u_sim, loss
            torch.cuda.empty_cache()
            gc.collect()


        epoch_loss = np.mean(epoch_losses)

        model.eval()
        recalls, mr, mrr = evaluation(txt, cld, link, sim_func=model, t2c=True, recall_idx=[1, 5, 10])
        current_recall_at_5 = recalls[1]
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f},"
              f"Recall@1: {recalls[0]:.4f},  Recall@5: {recalls[1]:.4f}, Recall@10: {recalls[2]:.4f}")

        if current_recall_at_5 > best_recall_at_5:
            best_recall_at_5 = current_recall_at_5
            best_model_wts = model.state_dict()

        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall_at_5': best_recall_at_5,
                'best_model_wts': best_model_wts
            }
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")


            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()


        del recalls, mr, mrr
        torch.cuda.empty_cache()
        gc.collect()

    print(f"best_R@5:{best_recall_at_5}")

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    if best_model_wts is not None:
        torch.save(best_model_wts, save_model_path)

def parse_command_line():
    parser = argparse.ArgumentParser(description='Train a model and save parameters.')
    parser.add_argument('--save_model_path', "-s", default="./best_model/best_miEle813b.pth",
                        type=str, help='The path to save model parameters')
    parser.add_argument('--checkpoint_path', "-c", default="./checkpoint/miEle813b.pth",
                        type=str, help='The path to save checkpoint')
    parser.add_argument('--link_path', "-l", default="../7.27_link.json",
                        type=str, help='link file path')
    parser.add_argument('--batch_size', "-b", default=32,
                        type=int, help='batch size')
    parser.add_argument('--nhead', "-nh", default=16,
                        type=int, help='attention head numbers')
    parser.add_argument('--dropout', "-dp", default=0.1,
                        type=float, help='dropout rate')
    parser.add_argument('--num_layers', "-layers", default=6,
                        type=int, help='attention encoder numbers')
    parser.add_argument('--num_epochs', "-epo", default=100,
                        type=int, help='epoch numbers')
    parser.add_argument('--learning_rate', "-lr", default=0.002,
                        type=float, help='epoch numbers')
    parser.add_argument('--rank', "-rk", default=256,
                        type=int, help='rank')
    parser.add_argument('--cld_first_link', "-cf", default=False,
                        type=bool, help='link order')
    parser.add_argument('--is_i3d', default=False,
                        type=bool, help='i3d or t3d')

    args = parser.parse_args()
    # print(args)
    return args


def t_Lmodel():
    args = parse_command_line()
    # batch_size = 64
    link_path = '/YOUR_PATH/7.27_link.json'
    dataset = PairDataset("", "", link_path, device=device)
    txt, cld, link = dataset.get_all_T3D()
    print(f'txt size: {txt.size()}\n cld size: {cld.size()}\n link size: {len(link)}')
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    d_model = 512


    model = RetrievalModel(d_model, args.nhead, args.dropout, args.rank, num_layers=args.num_layers, is_i3d=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.91, 0.9993))  # 3*1e-3
    criterion = CLIPLoss()

    train_save_checkpoint(model, train_loader, criterion, optimizer, txt, cld, link,
                          args.save_model_path, args.checkpoint_path, args.num_epochs)


def t_mini_on_data():
    args = parse_command_line()
    # batch_size = 64
    link_path = '/your_path/redo_ele_link_811.json'  # ele dataset
    dataset = PairDataset("", "", link_path, device=device)
    txt_fpth = "/your_path/Ele_Txt_4Cld.pt"
    cld_fpth = "/your_path/Ele_Cld_4Txt.pt"
    txt, cld, link = dataset.get_all_ele(txt_fpth, cld_fpth)

    print(f'txt size: {txt.size()}\n cld size: {cld.size()}\n link size: {len(link)}')
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    d_model = 512


    model = MiniModel(d_model=d_model, n_head=args.nhead,  dropout=args.dropout, rank=32, num_layers=args.num_layers)
    print(f"model:{model}")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3, betas=(0.91, 0.9993))  # 3*1e-3
    criterion = CLIPLoss()


    train_save_checkpoint(model, train_loader, criterion, optimizer, txt, cld, link,
                          args.save_model_path, args.checkpoint_path, args.num_epochs)


def t_EleLarge_on_data():
    args = parse_command_line()
    # batch_size = 64
    cld_path = "/your_path/Ele_Cld_4Txt.pt"
    txt_path = "/your_path/Ele_Txt_4Cld.pt"
    link_path = '/your_path/redo_ele_link_811.json'  # ele dataset
    dataset = PairDataset("", "", link_path, device=device)
    txt, cld, link = dataset.get_all_ele(txt_path, cld_path)

    if args.cld_first_link:
        link = [[pair[1], pair[0]] for pair in link]
    print(f"link[0]: {link[0]}")
    print(f'txt size: {txt.size()}\ncld size: {cld.size()}\nlink size: {len(link)}')
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    d_model = 512
    conf = RetrievalConfig(
        vocab_size=5500,
        block_size=512, 
        n_layer=args.num_layers,
        n_head=args.nhead,
        n_embd=d_model,
        dropout=args.dropout,
        device=device,
        rank=args.rank,
        is_i3d=args.is_i3d
    )

    model = RetrievalModel_t(conf)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)  
    # 3*1e-3
    criterion = CLIPLoss()


    train_save_checkpoint(model, train_loader, criterion, optimizer, txt, cld, link,
                          args.save_model_path, args.checkpoint_path, args.num_epochs)


if __name__ == '__main__':
    # parse_command_line()
    # t_EleLarge_on_data()
    # t_I3DLarge_on_data()
    # t_pair_dataset()
    t_mini_on_data()
    # device = torch.device("cuda:1")
    # t_mini_on_data()
    # batch_size = 2
    # t_cliploss()
    # # baseline 7.9 test1

    # dataset = PairDataset("", "", link_path, device=device)
    # txt, cld, link = dataset.get_all()
    # print(f'txt size: {txt.size()}\n cld size: {cld.size()}\n link size: {len(link)}')
    # train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)


