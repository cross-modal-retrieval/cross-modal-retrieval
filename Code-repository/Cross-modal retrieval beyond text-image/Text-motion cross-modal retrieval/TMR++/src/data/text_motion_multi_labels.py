import os
import codecs as cs
import orjson  # loading faster than json
import json
import logging
import random

import torch
import numpy as np
from .text_motion import TextMotionDataset
from tqdm import tqdm

from .collate import collate_text_motion_multiple_texts



def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionMultiLabelsDataset(TextMotionDataset):
    def __init__(
        self,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
    ):
        if tiny:
            split = split + "_tiny"

        self.collate_fn = collate_text_motion_multiple_texts
        self.split = split
        self.keyids = read_split(path, split)

        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = "train" in split
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]

        self.nfeats = self.motion_loader.nfeats

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue  
        
    def load_keyid(self, keyid, device=None, text_idx=None, sent_emb_mode="first"):
        annotations = self.annotations[keyid]                    

        index = 0

        path = annotations["path"]
        annotation = annotations["annotations"][index]
        start = annotation["start"]
        end = annotation["end"]

        texts = [ann["text"] for ann in annotations["annotations"]]

        text_x_dicts = self.text_to_token_emb(texts) # [{"x": ..., "length": ...}, {"x": ..., "length"}: ..., ... ]
        motion_x_dict = self.motion_loader(
            path=path,
            start=start,
            end=end,
        )

        if sent_emb_mode == "first":
            sent_emb = self.text_to_sent_emb(texts[0])
        elif sent_emb_mode == "average":
            sent_emb = torch.stack([self.text_to_sent_emb(text) for text in texts])
            sent_emb = torch.mean(sent_emb, axis=0)
            sent_emb = torch.nn.functional.normalize(sent_emb, dim=0)

        output = {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dicts,
            "text": texts,
            "keyid": keyid,
            "sent_emb": sent_emb,
        }
        
        if device is not None:
            output["motion_x_dict"]["x"] = output["motion_x_dict"]["x"].to(device)
            for text_x_dict in output["text_x_dict"]:
                text_x_dict["x"] = text_x_dict["x"].to(device)

        return output


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))

