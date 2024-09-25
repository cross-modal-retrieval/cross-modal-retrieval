import copy
import logging
import numpy as np
import random
import torch
from tqdm import tqdm

from .collate import collate_text_motion_multiple_texts
from .text_motion import load_annotations, TextMotionDataset


class AugmentedTextMotionDataset(TextMotionDataset):
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
        paraphrase_filename: str = None,
        summary_filename: str = None,
        paraphrase_prob: float = 0.2,
        summary_prob: float = 0.2,
        averaging_prob: float = 0.4,
        text_sampling_nbr: int = 4
    ):
        super().__init__(path, motion_loader, text_to_sent_emb, text_to_token_emb,
                         split=split, min_seconds=min_seconds, max_seconds=max_seconds,
                         preload=False, tiny=tiny)

        self.collate_fn = collate_text_motion_multiple_texts

        assert paraphrase_prob == 0 or paraphrase_filename is not None
        assert summary_prob == 0 or summary_filename is not None

        self.text_sampling_nbr = text_sampling_nbr
        self.paraphrase_prob = 0
        if split=="train" and paraphrase_filename is not None:
            self.annotations_paraphrased = load_annotations(path, name=paraphrase_filename)
            self.paraphrase_prob = paraphrase_prob
        self.summary_prob = 0
        if split=="train" and summary_filename is not None:
            self.annotations_summary = load_annotations(path, name=summary_filename)
            self.summary_prob = summary_prob
        self.averaging_prob = 0
        if split=="train" and paraphrase_filename is not None:
            self.averaging_prob = averaging_prob

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            if "train" in split and paraphrase_filename is not None:
                self.annotations_paraphrased = self.filter_annotations(self.annotations_paraphrased)
            if "train" in split and summary_filename is not None:
                self.annotations_summary = self.filter_annotations(self.annotations_summary)

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def load_keyid(self, keyid, text_idx=None, sent_emb_mode="first"):

        p = random.random() # Probability that will determine if we use data from augmentation, and with which config
        averaging = False
        if self.is_training and p < self.paraphrase_prob:
            annotations = self.annotations_paraphrased[keyid]
        elif self.is_training and p < self.summary_prob + self.paraphrase_prob:
            if keyid in self.annotations_summary:
                annotations = self.annotations_summary[keyid]
            else:
                annotations = self.annotations_paraphrased[keyid] # For Babel that has no summary
        elif self.is_training and p < self.averaging_prob + self.summary_prob + self.paraphrase_prob:
            annotations = copy.deepcopy(self.annotations[keyid])
            if hasattr(self, "annotations_paraphrased") and keyid in self.annotations_paraphrased:
                annotations["annotations"] += self.annotations_paraphrased[keyid]["annotations"]
            if hasattr(self, "annotations_summary") and keyid in self.annotations_summary:
                annotations["annotations"] += self.annotations_summary[keyid]["annotations"]
            averaging = True
        else:
            annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if averaging:
            if isinstance(self.text_sampling_nbr, int): # If number of samples if provided
                n = min(self.text_sampling_nbr, len(annotations["annotations"]))
            else: # If number of sample not provided, it's chosen randomly
                n = random.randint(2, len(annotations["annotations"]))
            index = random.sample(range(0, len(annotations["annotations"])), n)
        elif text_idx is not None:
            index = text_idx % len(annotations["annotations"])
        elif self.is_training:
            index = np.random.randint(len(annotations["annotations"]))

        if isinstance(index, int):
            index = [index]

        annotation_list = [annotations["annotations"][i] for i in index]
        text = [ann["text"] for ann in annotation_list]
        annotation0 = annotations["annotations"][index[0]]

        text_x_dict = [self.text_to_token_emb(t) for t in text]

        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation0["start"],
            end=annotation0["end"],
        )

        if sent_emb_mode == "first":
            sent_emb = self.text_to_sent_emb(text[0])
        elif sent_emb_mode == "average":
            sent_emb = torch.stack([self.text_to_sent_emb(t) for t in text])
            sent_emb = torch.mean(sent_emb, axis=0)
            sent_emb = torch.nn.functional.normalize(sent_emb, dim=0)

        output = {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dict,
            "text": text,
            "keyid": keyid,
            "sent_emb": sent_emb,
        }

        # TODO
        #if device is not None:
        #    output["motion_x_dict"]["x"] = output["motion_x_dict"]["x"].to(device)
        #    for i in range(len(output["text_x_dict"][i])):
        #        output["text_x_dict"][i]["x"] = output["text_x_dict"][i]["x"].to(device)

        return output
